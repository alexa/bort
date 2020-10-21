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
Main classification script, heavily modified from:
https://github.com/dmlc/gluon-nlp/blob/v0.9.x/scripts/bert/finetune_classifier.py
"""

import io
import os
import time
import json
import argparse
import random
import logging
import warnings
import collections

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.contrib.amp import amp
from mxnet.gluon import HybridBlock, nn
import gluonnlp as nlp

from utils.finetune_utils import ALL_TASKS_DIC, EXTRA_METRICS
from bort.bort import BortClassifier, get_bort_model
from utils.finetune_utils import preprocess_data, do_log


def get_parser():
    parser = argparse.ArgumentParser(description='Bort fine-tune examples for various NLU tasks.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs.')
    parser.add_argument('--ramp_up_epochs', type=int,
                        default=3, help='number of ramp up epochs.')
    parser.add_argument('--training_steps', type=int,
                        help='The total training steps. Note that if specified, epochs will be ignored.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size. Number of examples per gpu in a minibatch.')
    parser.add_argument('--dev_batch_size', type=int, default=8,
                        help='Batch size for dev set and test set')

    parser.add_argument('--init', type=str, default='uniform', choices=['gaussian', 'uniform', 'orthogonal', 'xavier'],
                        help='Initialization distribution.')
    parser.add_argument('--prob', type=float, default=0.5,
                        help='The probability around which to center the distribution.')

    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Initial learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                        help='Small value to avoid division by 0')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum length of the sentence pairs')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Which GPU to use for fine-tuning.')

    parser.add_argument('--use_scheduler', default=True, type=bool)
    parser.add_argument('--accumulate', type=int, default=None,
                        help='The number of batches for gradients accumulation to simulate large batch size. ')

    parser.add_argument('--log_interval', type=int,
                        default=10, help='report interval')
    parser.add_argument('--no_distributed', default=False, type=bool)

    parser.add_argument('--task_name', type=str, choices=ALL_TASKS_DIC.keys())

    parser.add_argument('--dataset', type=str, default='openwebtext_ccnews_stories_books_cased',
                        choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                                 'openwebtext_book_corpus_wiki_en_uncased', 'wiki_multilingual_uncased',
                                 'wiki_multilingual_cased', 'wiki_cn_cased',
                                 'openwebtext_ccnews_stories_books_cased'])

    parser.add_argument('--pretrained_parameters', type=str,
                        default=None, help='Pre-trained Bort model parameter file.')
    parser.add_argument('--model_parameters', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_dir',
                        help='The output directory to where the model params will be written.')
    parser.add_argument('--only_inference', action='store_true',
                        help='If set, we skip training and only perform inference on dev and test data.')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16'], help='The data type for training.')
    parser.add_argument('--early_stop', type=int, default=None,
                        help='Whether to perform early stopping based on the metric on dev set.')

    parser.add_argument('--multirc_test_location', type=str, required=False, default="/home/ec2-user/.mxnet/datasets/superglue_multirc/test.jsonl",
                        help='Location of the MultiRC test set, in case it is not in the default location.')
    parser.add_argument('--record_test_location', type=str, required=False, default="/home/ec2-user/.mxnet/datasets/superglue_record/test.jsonl",
                        help='Location of the ReCoRD test set, in case it is not in the default location.')
    parser.add_argument('--record_dev_location', type=str, required=False, default="/home/ec2-user/.mxnet/datasets/superglue_record/val.jsonl",
                        help='Location of the ReCoRD dev set, in case it is not in the default location.')
    parser.add_argument('--race_dataset_location', type=str, required=False, default=None,
                        help='Location of the RACE dataset, in case it is not in the default location.')

    return parser


def load_and_setup_model(task, args):

    pretrained_parameters = args.pretrained_parameters
    model_parameters = args.model_parameters
    dataset = args.dataset

    if only_inference and not model_parameters:
        warnings.warn(
            'model_parameters is not set. Randomly initialized model will be used for inference.')

    get_pretrained = not (
        pretrained_parameters is not None or model_parameters is not None)

    # STS-B is a regression task and STSBTask().class_labels returns None
    do_regression = not task.class_labels
    if do_regression:
        num_classes = 1
        loss_function = gluon.loss.L2Loss()
    else:
        num_classes = len(task.class_labels)
        loss_function = gluon.loss.SoftmaxCELoss()

    bort, vocabulary = get_bort_model("bort_4_8_768_1024", dataset_name=dataset,
                                      pretrained=get_pretrained, ctx=ctx,
                                      use_pooler=True, use_decoder=False, use_classifier=False)
    # TODO: CoLA uses a different classifier!
    model = BortClassifier(bort, dropout=args.dropout, num_classes=num_classes)

    if args.init == "gaussian":
        initializer = mx.init.Normal(args.prob)
    if args.init == "uniform":
        initializer = mx.init.Uniform(args.prob)
    if args.init == "orthogonal":
        initializer = mx.init.Orthogonal(scale=args.prob)
    if args.init == "xavier":
        initializer = mx.init.Xavier()

    if not model_parameters:
        model.classifier.initialize(init=initializer, ctx=ctx)

    # load checkpointing
    if pretrained_parameters:
        logging.info('loading Bort params from %s', pretrained_parameters)
        nlp.utils.load_parameters(
            model.bort, pretrained_parameters, ctx=ctx, ignore_extra=True, cast_dtype=True)

    if model_parameters:
        logging.info('loading model params from %s', model_parameters)
        nlp.utils.load_parameters(
            model, model_parameters, ctx=ctx, cast_dtype=True)

    # data processing
    nlp.utils.mkdir(output_dir)
    do_lower_case = 'uncased' in dataset

    tokenizer = nlp.data.GPT2BPETokenizer()

    return model, tokenizer, loss_function, vocabulary


def setup_logger(args):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logging.captureWarnings(True)
    fh = logging.FileHandler('log_{0}.txt'.format(task_name))
    formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                                  datefmt='%H:%M:%S')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    log.addHandler(console)
    log.addHandler(fh)


def train(metric):
    """Training function."""
    if not only_inference:
        logging.info(
            'Now we are doing Bort classification training on %s!', ctx)

    all_model_params = model.collect_params()
    optimizer_params = {'learning_rate': lr,
                        'epsilon': epsilon, 'wd': args.weight_decay}
    trainer = gluon.Trainer(all_model_params, "bertadam", optimizer_params,
                            update_on_kvstore=False)
    if args.dtype == 'float16':
        amp.init_trainer(trainer)

    epoch_number = args.ramp_up_epochs
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * args.ramp_up_epochs)
    if args.training_steps:
        num_train_steps = args.training_steps
        epoch_number = 9999

    logging.info('training steps=%d', num_train_steps)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in all_model_params.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if accumulate and accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    # track best eval score
    metric_history = []
    best_metric = None
    patience = args.early_stop

    tic = time.time()
    finish_flag = False
    for epoch_id in range(args.epochs):
        if args.early_stop and patience == 0:
            logging.info('Early stopping at epoch %d', epoch_id)
            break
        if finish_flag:
            break
        if not only_inference:
            metric.reset()
            step_loss = 0
            tic = time.time()
            all_model_params.zero_grad()

            for batch_id, seqs in enumerate(train_data):
                # learning rate schedule
                if args.use_scheduler:
                    if step_num < num_warmup_steps:
                        new_lr = lr * step_num / num_warmup_steps
                    else:
                        non_warmup_steps = step_num - num_warmup_steps
                        offset = non_warmup_steps / \
                            (num_train_steps - num_warmup_steps)
                        new_lr = max(1e-7, lr - offset * lr)

                    trainer.set_learning_rate(new_lr)

                # forward and backward
                with mx.autograd.record():
                    if args.no_distributed:
                        input_ids, segment_ids, valid_length, label = seqs
                    else:
                        input_ids, segment_ids, valid_length, label = seqs[
                            hvd.rank()]

                    out = model(input_ids.as_in_context(ctx),
                                valid_length.as_in_context(ctx).astype('float32'))
                    ls = loss_function(out, label.as_in_context(ctx)).mean()
                    if args.dtype == 'float16':
                        with amp.scale_loss(ls, trainer) as scaled_loss:
                            mx.autograd.backward(scaled_loss)
                    else:
                        ls.backward()

                # update
                if not accumulate or (batch_id + 1) % accumulate == 0:
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(accumulate if accumulate else 1)
                    step_num += 1
                    if accumulate and accumulate > 1:
                        # set grad to zero for gradient accumulation
                        all_model_params.zero_grad()

                step_loss += ls.asscalar()
                if not do_regression:
                    label = label.reshape((-1))
                metric.update([label], [out])
                if (batch_id + 1) % (args.log_interval) == 0:
                    if is_master_node:
                        do_log(batch_id, len(train_data), metric, step_loss, args.log_interval,
                               epoch_id=epoch_id, learning_rate=trainer.learning_rate)
                    step_loss = 0
            mx.nd.waitall()

        # inference on dev data
        tmp_metric = []
        for segment, dev_data in dev_data_list:
            if is_master_node:
                metric_nm, metric_val = evaluate(
                    dev_data, metric, segment, epoch=epoch_id)
                if best_metric is None or metric_val >= best_metric:
                    best_metric = metric_val
                    patience = args.early_stop
                else:
                    if args.early_stop is not None:
                        patience -= 1
                tmp_metric += metric_val

        if is_master_node:
            # For multi-valued tasks (e.g., MNLI), we maximize the average of
            # the metrics.
            metric_history.append(
                (epoch_id, metric_nm + ["average"], metric_val + [sum(tmp_metric) / len(tmp_metric)]))

        if not only_inference and is_master_node:
            ckpt_name = 'model_bort_{0}_{1}.params'.format(task_name, epoch_id)
            params_saved = os.path.join(output_dir, ckpt_name)
            nlp.utils.save_parameters(model, params_saved)
            logging.info('params saved in: %s', params_saved)
            toc = time.time()
            logging.info('Time cost=%.2fs', toc - tic)
            tic = toc

    if not only_inference and is_master_node:
        metric_history.sort(key=lambda x: x[-1][0], reverse=True)
        epoch_id, metric_nm, metric_val = metric_history[0]
        ckpt_name = 'model_bort_{0}_{1}.params'.format(task_name, epoch_id)
        params_saved = os.path.join(output_dir, ckpt_name)
        nlp.utils.load_parameters(model, params_saved)
        metric_str = 'Best model at epoch {}. Validation metrics:'.format(
            epoch_id)
        metric_str += ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(metric_str, *metric_val)

    # inference on test data
    for segment, test_data in test_data_list:
        if is_master_node:
            test(test_data, segment, acc=metric_val[0])


def evaluate(loader_dev, metric, segment,  epoch=0):
    """Evaluate the model on validation dataset."""
    logging.info('Now we are doing evaluation on %s with %s.', segment, ctx)
    metric.reset()
    step_loss = 0
    counter = 0
    tic = time.time()

    all_results = collections.defaultdict(list)
    if "RACE" in args.task_name:
        from utils.finetune_utils import process_RACE_answers, RACEHash
        race_dev_data = RACEHash(
            args.race_dataset_location, args.task_name, segment="dev")
        results = []
    if "ReCoRD" in args.task_name:
        from utils.finetune_utils import process_ReCoRD_answers, ReCoRDHash
        record_dev_data = ReCoRDHash(args.record_dev_location)
        results = []

    for batch_id, seqs in enumerate(loader_dev):
        input_ids, segment_ids, valid_length, label = seqs
        out = model(input_ids.as_in_context(ctx),
                    valid_length.as_in_context(ctx).astype('float32'))
        ls = loss_function(out, label.as_in_context(ctx)).mean()
        step_loss += ls.asscalar()
        if not do_regression:
            label = label.reshape((-1))

        metric.update([label], [out])
        for example_id, (l, p) in enumerate(zip(label, out)):
            all_results[counter].append([[l], [p]])
            counter += 1

        if "RACE" in args.task_name:
            indices = mx.nd.topk(
                out, k=1, ret_typ='indices', dtype='int32').asnumpy()
            for index, logits in zip(indices, out):
                results.append(
                    (task.class_labels[int(index)], mx.nd.softmax(logits)))

        if (batch_id + 1) % (args.log_interval) == 0:
            do_log(batch_id, len(loader_dev), metric,
                   step_loss, args.log_interval)
            step_loss = 0

    if args.task_name in EXTRA_METRICS:
        metric_nm_, metric_val_ = metric.get()
        metric_nm, metric_val = [metric_nm_], [metric_val_]
        labels = [v[0][0][0].asnumpy()[0] for v in all_results.values()]
        preds = [(v[0][-1][0].asnumpy()).argmax()
                 for v in all_results.values()]
        for e_metric_nm, e_metric_val in EXTRA_METRICS[args.task_name]:
            metric_nm.append(e_metric_nm)
            metric_val.append(e_metric_val(labels, preds))
    else:
        metric_nm, metric_val = metric.get()

    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    if "RACE" in args.task_name:
        # Class accuracy is bugged; we only need regular accuracy anyway
        result_data, class_accuracy = process_RACE_answers(
            race_dev_data, results)
    if "ReCoRD" in args.task_name:
        result_data, class_accuracy = process_ReCoRD_answers(
            results, record_dev_data)

    metric_str = 'epoch: ' + \
        str(epoch) + '; validation metrics:' + \
        ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(metric_str, *metric_val)

    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 dev_batch_size * len(loader_dev) / (toc - tic))
    return metric_nm, metric_val


def test(loader_test, segment, acc=0):
    """Inference function on the test dataset."""
    logging.info('Now we are doing testing on %s with %s.', segment, ctx)
    tic = time.time()
    results = []

    # Eval loop
    for _, seqs in enumerate(loader_test):
        input_ids, segment_ids, valid_length = seqs
        out = model(input_ids.as_in_context(ctx),
                    valid_length.as_in_context(ctx).astype('float32'))
        if not task.class_labels:
            # regression task
            for result in out.asnumpy().reshape(-1).tolist():
                results.append('{:.3f}'.format(result))
        else:
            indices = mx.nd.topk(
                out, k=1, ret_typ='indices', dtype='int32').asnumpy()
            for index, logits in zip(indices, out):
                if args.task_name == "ReCoRD" or "RACE" in args.task_name:
                    results.append(
                        (task.class_labels[int(index)], mx.nd.softmax(logits)))
                else:
                    results.append(task.class_labels[int(index)])

    if args.task_name == "ReCoRD":
        from utils.finetune_utils import process_ReCoRD_answers, ReCoRDHash
        record_test_data = ReCoRDHash(args.record_test_location)
        result_data, _ = process_ReCoRD_answers(results, record_test_data)
    if args.task_name == "MultiRC":
        from utils.finetune_utils import process_MultiRC_answers
        result_data = process_MultiRC_answers(
            results, args.multirc_test_location)
    if "RACE" in args.task_name:
        from utils.finetune_utils import process_RACE_answers, RACEHash
        race_test_data = RACEHash(
            args.race_dataset_location, args.task_name, segment="test")
        result_data, _ = process_RACE_answers(race_test_data, results)

    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 dev_batch_size * len(loader_test) / (toc - tic))

    # Write the results to a file.
    segment = segment.replace('_mismatched', '-mm')
    segment = segment.replace('_matched', '-m')
    segment = segment.replace('SST', 'SST-2')

    filename = args.task_name + \
        segment.replace('test', '') + str(acc) + '.' + task.output_format
    test_path = os.path.join(args.output_dir, filename)

    if task.output_format == "tsv":
        with io.open(test_path, 'w', encoding='utf-8') as f:
            f.write(u'index\tprediction\n')
            for i, pred in enumerate(results):
                f.write(u'%d\t%s\n' % (i, str(pred)))
    elif task.output_format == "txt":
        with io.open(test_path, 'w', encoding='utf-8') as f:
            for pred in result_data:
                f.write(json.dumps(pred))
                f.write("\n")
    else:
        with io.open(test_path, 'w', encoding='utf-8') as f:
            if args.task_name == "MultiRC" or args.task_name == "ReCoRD":
                for pred in result_data:
                    f.write(json.dumps(pred))
                    f.write("\n")
            else:
                for i, pred in enumerate(results):
                    f.write(u'{"idx":%d,"label":"%s"}\n' % (i, str(pred)))


parser = get_parser()
args = parser.parse_args()

task_name = args.task_name
setup_logger(task_name)
logging.info(args)

batch_size = args.batch_size
dev_batch_size = args.dev_batch_size
lr = args.lr
epsilon = args.epsilon
accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    logging.info(
        'Using gradient accumulation. Effective batch size = batch_size * accumulate = %d', accumulate * batch_size)

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

is_master_node = True
world_size = None
if args.gpu == 999 or args.gpu is None:
    ctx = mx.cpu()
else:
    if args.no_distributed:
        ctx = mx.gpu(args.gpu)
    else:
        try:
            import horovod.mxnet as hvd
        except ImportError:
            logging.info('Horovod must be installed.')
            exit()
        hvd.init()
        world_size = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        is_master_node = rank == 0
        ctx = mx.gpu(local_rank)

task = ALL_TASKS_DIC[task_name]
if "RACE" in task_name and args.race_dataset_location is not None:
    task._set_dataset_location(args.race_dataset_location)
do_regression = not task.class_labels

# data type with mixed precision training
if args.dtype == 'float16':
    amp.init()

# model and loss
only_inference = args.only_inference
output_dir = args.output_dir

model, tokenizer, loss_function, vocabulary = load_and_setup_model(task, args)
logging.debug(model)
model.hybridize(static_alloc=True)
loss_function.hybridize(static_alloc=True)

# Get the loader.
logging.info('Processing dataset...')

train_data, dev_dataset, dev_data_list, test_data_list, num_train_examples = preprocess_data(
    tokenizer, task, batch_size, dev_batch_size, args.max_len, vocabulary, world_size)


if __name__ == '__main__':
    train(task.metrics)
