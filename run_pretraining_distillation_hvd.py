# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
#    http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Pretraining for Bort. This is taken from:
    https://github.com/dmlc/gluon-nlp/blob/v0.9.x/scripts/bert/run_pretraining.py
    but modified to work with Bort
"""

import os
import random
import warnings
import logging
import functools
import time
import sys

import mxnet as mx
import gluonnlp as nlp

from utils.fp16_utils import FP16Trainer
from utils.pretraining_distillation_utils import get_model_loss, get_pretrain_data_npz
from utils.pretraining_distillation_utils import split_and_load, log, evaluate, forward, get_argparser
from utils.pretraining_distillation_utils import save_parameters, save_states
from utils.pretraining_distillation_utils import get_teacher_model_loss, LogTB, profile

from bort import bort

from tqdm import tqdm
from gluonnlp.metric import MaskedAccuracy

# parser
parser = get_argparser()
args = parser.parse_args()

# logging
level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=level,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(level)
logging.info(args)
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'

try:
    import horovod.mxnet as hvd
except ImportError:
    logging.info('horovod must be installed.')
    exit()
hvd.init()
store = None
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()
is_master_node = (rank == local_rank)
if not args.use_avg_len and hvd.size() > 1:
    logging.info('Specifying --use-avg-len and setting --batch_size with the '
                 'target number of tokens would help improve training throughput.')
logging.info('Using effective batch size = batch_size * accumulate * np = %d',
             args.batch_size * args.accumulate * num_workers)


def train(data_train, dataset_eval, model, teacher_model, mlm_loss, teacher_ce_loss, teacher_mse_loss,
          vocab_size, ctx, teacher_ce_weight, distillation_temperature, mlm_weight, log_tb):
    """Training function."""
    params = model.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    mlm_metric = MaskedAccuracy()
    mlm_metric.reset()

    logging.debug('Creating distributed trainer...')
    lr = args.lr
    optim_params = {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
    if args.dtype == 'float16':
        optim_params['multi_precision'] = True

    dynamic_loss_scale = args.dtype == 'float16'
    if dynamic_loss_scale:
        loss_scale_param = {'scale_window': 2000 / num_workers}
    else:
        loss_scale_param = None
    trainer = hvd.DistributedTrainer(params, 'bertadam', optim_params)

    if args.dtype == 'float16':
        fp16_trainer = FP16Trainer(trainer, dynamic_loss_scale=dynamic_loss_scale,
                                   loss_scaler_params=loss_scale_param)
        trainer_step = lambda: fp16_trainer.step(1, max_norm=1 * num_workers)
    else:
        trainer_step = lambda: trainer.step(1)


    if args.start_step:
        out_dir = os.path.join(args.ckpt_dir, f"checkpoint_{args.start_step}")
        state_path = os.path.join(
            out_dir, '%07d.states.%02d' % (args.start_step, local_rank))
        logging.info('Loading trainer state from %s', state_path)
        nlp.utils.load_states(trainer, state_path)

    accumulate = args.accumulate
    num_train_steps = args.num_steps
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    params = [p for p in model.collect_params().values()
              if p.grad_req != 'null']
    param_dict = model.collect_params()

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    if accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    train_begin_time = time.time()
    begin_time = time.time()
    running_mlm_loss, running_teacher_ce_loss, running_teacher_mse_loss = 0, 0, 0
    running_num_tks = 0
    batch_num = 0
    step_num = args.start_step

    logging.debug('Training started')

    pbar = tqdm(total=num_train_steps, desc="Training:")

    while step_num < num_train_steps:
        for raw_batch_num, data_batch in enumerate(data_train):
            sys.stdout.flush()
            if step_num >= num_train_steps:
                break
            if batch_num % accumulate == 0:
                step_num += 1
                # if accumulate > 1, grad_req is set to 'add', and zero_grad is
                # required
                if accumulate > 1:
                    param_dict.zero_grad()
                # update learning rate
                if step_num <= num_warmup_steps:
                    new_lr = lr * step_num / num_warmup_steps
                else:
                    offset = lr * step_num / num_train_steps
                    new_lr = lr - offset
                trainer.set_learning_rate(new_lr)
                if args.profile:
                    profile(step_num, 10, 14,
                            profile_name=args.profile + str(rank))

            # load data
            if args.use_avg_len:
                data_list = [[[s.as_in_context(context) for s in seq] for seq in shard]
                             for context, shard in zip([ctx], data_batch)]
            else:
                data_list = list(split_and_load(data_batch, [ctx]))
            #data = data_list[0]
            data = data_list

            # forward
            with mx.autograd.record():
                (loss_val, ns_label, classified, masked_id, decoded, masked_weight, mlm_loss_val, teacher_ce_loss_val,
                 teacher_mse_loss_val, valid_len) = forward(data, model, mlm_loss, vocab_size,
                                                            args.dtype,
                                                            mlm_weight=mlm_weight,
                                                            teacher_ce_loss=teacher_ce_loss,
                                                            teacher_mse_loss=teacher_mse_loss,
                                                            teacher_model=teacher_model,
                                                            teacher_ce_weight=teacher_ce_weight,
                                                            distillation_temperature=distillation_temperature)
                loss_val = loss_val / accumulate
                # backward
                if args.dtype == 'float16':
                    fp16_trainer.backward(loss_val)
                else:
                    loss_val.backward()

            running_mlm_loss += mlm_loss_val.as_in_context(mx.cpu())
            running_teacher_ce_loss += teacher_ce_loss_val.as_in_context(
                mx.cpu())
            running_teacher_mse_loss += teacher_mse_loss_val.as_in_context(
                mx.cpu())
            running_num_tks += valid_len.sum().as_in_context(mx.cpu())

            # update
            if (batch_num + 1) % accumulate == 0:
                # step() performs 3 things:
                # 1. allreduce gradients from all workers
                # 2. checking the global_norm of gradients and clip them if necessary
                # 3. averaging the gradients and apply updates
                trainer_step()

            mlm_metric.update([masked_id], [decoded], [masked_weight])

            # logging
            if step_num % args.log_interval == 0 and batch_num % accumulate == 0:
                log("train ",
                    begin_time,
                    running_num_tks,
                    running_mlm_loss / accumulate,
                    running_teacher_ce_loss / accumulate,
                    running_teacher_mse_loss / accumulate,
                    step_num,
                    mlm_metric,
                    trainer,
                    args.log_interval,
                    model=model,
                    log_tb=log_tb,
                    is_master_node=is_master_node)
                begin_time = time.time()
                running_mlm_loss = running_teacher_ce_loss = running_teacher_mse_loss = running_num_tks = 0
                mlm_metric.reset_local()

            # saving checkpoints
            if step_num % args.ckpt_interval == 0 and batch_num % accumulate == 0:
                if is_master_node:
                    out_dir = os.path.join(args.ckpt_dir, f"checkpoint_{step_num}")
                    if not os.path.isdir(out_dir):
                        nlp.utils.mkdir(out_dir)
                    save_states(step_num, trainer, out_dir, local_rank)
                    if local_rank == 0:
                        save_parameters(step_num, model, out_dir)
                if data_eval:
                    dataset_eval = get_pretrain_data_npz(
                        data_eval, args.batch_size_eval, 1, False, False, 1)
                    evaluate(dataset_eval, model, mlm_loss, len(vocab), [ctx], args.log_interval, args.dtype,
                             mlm_weight=mlm_weight,
                             teacher_ce_loss=teacher_ce_loss,
                             teacher_mse_loss=teacher_mse_loss,
                             teacher_model=teacher_model,
                             teacher_ce_weight=teacher_ce_weight,
                             distillation_temperature=distillation_temperature,
                             log_tb=log_tb)

            batch_num += 1
        pbar.update(1)
        del data_batch
    if is_master_node:
        out_dir = os.path.join(args.ckpt_dir, "checkpoint_last")
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        save_states(step_num, trainer, out_dir, local_rank)
        if local_rank == 0:
            save_parameters(step_num, model, args.ckpt_dir)
    mx.nd.waitall()
    train_end_time = time.time()
    pbar.close()
    logging.info('Train cost={:.1f}s'.format(
        train_end_time - train_begin_time))

if __name__ == '__main__':
    random_seed = random.randint(0, 1000)
    nlp.utils.mkdir(args.ckpt_dir)

    if args.cpu_only:
        ctx = mx.cpu(local_rank)
    else:
        ctx = mx.gpu(local_rank)

    dataset_name, vocab = args.dataset_name, None

    model, vocab = bort.get_bort_model(args.model, dataset_name=dataset_name, vocab=vocab,
                                       pretrained=args.pretrained, ctx=ctx)
    model, mlm_loss = get_model_loss([ctx], model, args.pretrained, args.dtype,
                                     ckpt_dir=args.ckpt_dir, start_step=args.start_step)
    if args.teacher_model:
        teacher_model, teacher_ce_loss, teacher_mse_loss, _ = get_teacher_model_loss([ctx], args.teacher_model,
                                                                                     dataset_name, vocab, args.dtype,
                                                                                     ckpt_dir=args.teacher_ckpt_dir)
    else:
        teacher_model = None
        teacher_ce_loss = None
        teacher_mse_loss = None
    log_tb = LogTB(args)
    logging.debug('Model created')
    data_eval = args.data_eval

    logging.debug('Random seed set to %d', random_seed)
    mx.random.seed(random_seed)

    num_parts = num_workers
    part_idx = rank

    get_dataset_fn = get_pretrain_data_npz

    dataset_eval = get_dataset_fn(data_eval, args.batch_size_eval, 1, True, args.use_avg_len, args.num_buckets,
                                  num_parts=num_parts, part_idx=part_idx)
    if args.data:
        data_train = get_dataset_fn(args.data, args.batch_size, 1, True,
                                    args.use_avg_len, args.num_buckets,
                                    num_parts=num_parts, part_idx=part_idx)
        train(data_train, dataset_eval, model, teacher_model, mlm_loss, teacher_ce_loss, teacher_mse_loss,
              len(vocab), ctx, args.teacher_ce_weight, args.distillation_temperature, args.mlm_weight, log_tb=log_tb)
    if data_eval:
        # eval data is always based on a fixed npz file.
        logging.info("evaluation of student model:")
        dataset_eval = get_pretrain_data_npz(
            data_eval, args.batch_size_eval, 1, False, False, 1)
        evaluate(dataset_eval, model, mlm_loss, len(vocab), [ctx], args.log_interval, args.dtype,
                 mlm_weight=args.mlm_weight,
                 teacher_ce_loss=teacher_ce_loss,
                 teacher_mse_loss=teacher_mse_loss,
                 teacher_model=teacher_model,
                 teacher_ce_weight=args.teacher_ce_weight,
                 distillation_temperature=args.distillation_temperature,
                 log_tb=log_tb)

        logging.info("evaluation of teacher model:")
        dataset_eval = get_pretrain_data_npz(
            data_eval, args.batch_size_eval, 1, False, False, 1)
        evaluate(dataset_eval, teacher_model, mlm_loss, len(vocab), [ctx], args.log_interval, args.dtype,
                 mlm_weight=args.mlm_weight,
                 teacher_ce_loss=teacher_ce_loss,
                 teacher_mse_loss=teacher_mse_loss,
                 teacher_model=teacher_model,
                 teacher_ce_weight=args.teacher_ce_weight,
                 distillation_temperature=args.distillation_temperature,
                 log_tb=log_tb)
