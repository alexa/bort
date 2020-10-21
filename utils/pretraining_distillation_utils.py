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

__all__ = ['get_model_loss', 'get_teacher_model_loss', 'get_pretrain_data_npz',
           'save_parameters', 'save_states', 'evaluate', 'forward', 'split_and_load',
           'get_argparser', 'generate_dev_set', 'LogTB', 'profile']

import time
import os
import logging
import argparse
import random
import multiprocessing

import numpy as np
from mxboard import SummaryWriter
import psutil

import mxnet as mx
from mxnet.gluon.data import DataLoader
from create_pretraining_data import create_training_instances
from data.dataloader import DatasetLoader, SamplerFn, DataLoaderFn, SimpleDatasetFn

import gluonnlp as nlp
from gluonnlp.data.batchify import Tuple, Stack, Pad
from gluonnlp.metric import MaskedAccuracy
from tqdm import *

from bort import bort


def get_teacher_model_loss(ctx, model, dataset_name, vocab, dtype,
                           ckpt_dir=None):
    """Get model for pre-training.

    Parameters
    ----------
    ctx : Context or list of Context
        Contexts to initialize model
    model : str
        The name of the model.
    dataset_name : str
        The name of the dataset, which is used to retrieve the corresponding vocabulary file
        when the vocab argument is not provided. Options include 'book_corpus_wiki_en_uncased',
        'book_corpus_wiki_en_cased', 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
        'wiki_cn_cased'.
    vocab : The vocabulary for the model. If not provided, The vocabulary will be constructed
        based on dataset_name.
    dtype : float
        Data type of the model for training.
    ckpt_dir : str
        The path to the checkpoint directory.

    Returns
    -------
    RoBERTaModel : the model for pre-training.
    Loss : the next sentence prediction loss.
    Loss : the masked langauge model loss.
    RoBERTaVocab : the vocabulary.
    """
    download_pretrained = ckpt_dir is None
    model, vocabulary = nlp.model.get_model(model, dataset_name=dataset_name, vocab=vocab,
                                            pretrained=download_pretrained, ctx=ctx)
    model.cast(dtype)
    if ckpt_dir:
        params_fn = [fn for fn in os.listdir(
            ckpt_dir) if fn[-len(".params"):] == ".params"][0]
        param_path = os.path.join(ckpt_dir, params_fn)
        nlp.utils.load_parameters(
            model, param_path, ctx=ctx, allow_missing=True)
        logging.info('Loading checkpoints from %s.', param_path)
    model.hybridize(static_alloc=True)

    # losses
    teacher_ce_loss = mx.gluon.loss.KLDivLoss(from_logits=False)
    teacher_ce_loss.hybridize(static_alloc=True)
    teacher_mse_loss = mx.gluon.loss.L2Loss()
    teacher_mse_loss.hybridize(static_alloc=True)

    return model, teacher_ce_loss, teacher_mse_loss, vocabulary


def get_model_loss(ctx, model, pretrained, dtype, ckpt_dir=None,
                   start_step=None):
    """Get model for pre-training.

    Parameters
    ----------
    ctx : Context or list of Context
        Contexts to initialize model
    model : BortModel
        The model
    pretrained : bool
        Whether to use pre-trained model weights as initialization.
    dtype : float
        Data type of the model for training.
    ckpt_dir : str
        The path to the checkpoint directory.
    start_step : int or None
        If provided, it loads the model from the corresponding checkpoint from the ckpt_dir.

    Returns
    -------
    BortModel : the model for pre-training.
    Loss : the next sentence prediction loss.
    Loss : the masked langauge model loss.
    """
    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.cast(dtype)

    if ckpt_dir and start_step:
        out_dir = os.path.join(ckpt_dir, f"checkpoint_{start_step}")
        param_path = os.path.join(out_dir, '%07d.params' % start_step)
        nlp.utils.load_parameters(
            model, param_path, ctx=ctx, allow_missing=True)
        logging.info('Loading step %d checkpoints from %s.',
                     start_step, param_path)

    model.hybridize(static_alloc=True)

    # losses
    mlm_loss = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss.hybridize(static_alloc=True)

    return model, mlm_loss


class BERTPretrainDataset(mx.gluon.data.ArrayDataset):
    """Dataset for BERT-style pre-training.

    Each record contains the following numpy ndarrays: input_ids, masked_lm_ids,
    masked_lm_positions, masked_lm_weights, next_sentence_labels, segment_ids, valid_lengths.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    tokenizer : BERTTokenizer
        The BERTTokenizer
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to use whole word masking.
    vocab : BERTVocab
        The BERTVocab
    num_workers : int
        The number of worker processes for dataset contruction.
    worker_pool : multiprocessing.Pool
        The worker process pool. Must be provided if num_workers > 1.
    """

    def __init__(self, filename, tokenizer, max_seq_length, short_seq_prob,
                 masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                 vocab, num_workers=1, worker_pool=None):
        logging.debug('start to load file %s ...', filename)
        dupe_factor = 1
        instances = create_training_instances(([filename], tokenizer, max_seq_length,
                                               short_seq_prob, masked_lm_prob,
                                               max_predictions_per_seq,
                                               whole_word_mask, vocab,
                                               dupe_factor, num_workers,
                                               worker_pool, None))
        super(BERTPretrainDataset, self).__init__(*instances)


class BERTSamplerFn(SamplerFn):
    """Callable object to create the sampler"""

    def __init__(self, use_avg_len, batch_size, shuffle, num_ctxes, num_buckets):
        self._use_avg_len = use_avg_len
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_ctxes = num_ctxes
        self._num_buckets = num_buckets

    def __call__(self, dataset):
        """Create data sampler based on the dataset"""
        if isinstance(dataset, nlp.data.NumpyDataset):
            lengths = dataset.get_field('valid_lengths')
        elif isinstance(dataset, BERTPretrainDataset):
            lengths = dataset.transform(lambda input_ids, segment_ids, masked_lm_positions,
                                        masked_lm_ids, masked_lm_weights,
                                        next_sentence_labels, valid_lengths:
                                        valid_lengths, lazy=False)
        else:
            raise ValueError('unexpected dataset type: %s' % str(dataset))

        if self._use_avg_len:
            # sharded data loader
            sampler = nlp.data.FixedBucketSampler(lengths=lengths,
                                                  # batch_size per shard
                                                  batch_size=self._batch_size,
                                                  num_buckets=self._num_buckets,
                                                  shuffle=self._shuffle,
                                                  use_average_length=True,
                                                  num_shards=self._num_ctxes)
        else:
            sampler = nlp.data.FixedBucketSampler(lengths,
                                                  batch_size=self._batch_size * self._num_ctxes,
                                                  num_buckets=self._num_buckets,
                                                  ratio=0,
                                                  shuffle=self._shuffle)
        logging.debug('Sampler created for a new dataset:\n%s',
                      sampler.stats())
        return sampler


class BERTDataLoaderFn(DataLoaderFn):
    """Callable object to create the data loader"""

    def __init__(self, use_avg_len, num_ctxes):
        self._use_avg_len = use_avg_len
        self._num_ctxes = num_ctxes

    def __call__(self, dataset, sampler):
        # A batch includes: input_id, masked_id, masked_position, masked_weight,
        #                   next_sentence_label, segment_id, valid_length
        batchify_fn = Tuple(Pad(),    # input_id
                            Pad(),    # masked_id
                            Pad(),    # masked_position
                            Pad(),    # masked_weight
                            Stack(),  # next_sentence_label
                            Pad(),    # segment_id
                            Stack())   # valid_length

        if self._use_avg_len:
            # sharded data loader
            dataloader = nlp.data.ShardedDataLoader(dataset,
                                                    batch_sampler=sampler,
                                                    batchify_fn=batchify_fn,
                                                    num_workers=self._num_ctxes)
        else:
            dataloader = DataLoader(dataset=dataset,
                                    batch_sampler=sampler,
                                    batchify_fn=batchify_fn,
                                    num_workers=self._num_ctxes)
        return dataloader


class BERTLoaderTransform(object):
    """Create dataloader for a BERT dataset. """

    def __init__(self, use_avg_len, batch_size, shuffle, num_ctxes, num_buckets):
        self._sampler_fn = BERTSamplerFn(
            use_avg_len, batch_size, shuffle, num_ctxes, num_buckets)
        self._data_fn = BERTDataLoaderFn(use_avg_len, num_ctxes)

    def __call__(self, dataset):
        """create data loader based on the dataset chunk"""
        sampler = self._sampler_fn(dataset)
        dataloader = self._data_fn(dataset, sampler)
        return dataloader


def get_pretrain_data_npz(data, batch_size, num_ctxes, shuffle, use_avg_len,
                          num_buckets, num_parts=1, part_idx=0):
    """create dataset for pretraining based on pre-processed npz files."""
    # handle commas in the provided path
    num_files = len(nlp.utils.glob(data))
    logging.info('%d files found.', num_files)
    assert num_files >= num_parts, \
        'Number of training files must be greater than the number of partitions. ' \
        'Only found %d files at %s' % (num_files, data)
    split_sampler = nlp.data.SplitSampler(
        num_files, num_parts=num_parts, part_index=part_idx)
    # read each file in as a separate NumpyDataset, split sample into buckets
    # for each gpu
    stream = nlp.data.SimpleDatasetStream(
        nlp.data.NumpyDataset, data, split_sampler, allow_pickle=True)
    stream = nlp.data.PrefetchingStream(stream, worker_type='process')

    # create data loader based on the dataset
    dataloader_xform = BERTLoaderTransform(use_avg_len, batch_size,
                                           shuffle, num_ctxes, num_buckets)
    # transform each NumpyDataset
    stream = stream.transform(dataloader_xform)
    return stream


def save_parameters(step_num, model, ckpt_dir):
    """Save the model parameter, marked by step_num."""
    param_path = os.path.join(ckpt_dir, '%07d.params' % step_num)
    logging.info('[step %d] Saving model params to %s.', step_num, param_path)
    nlp.utils.save_parameters(model, param_path)


def save_states(step_num, trainer, ckpt_dir, local_rank=0):
    """Save the trainer states, marked by step_num."""
    trainer_path = os.path.join(
        ckpt_dir, '%07d.states.%02d' % (step_num, local_rank))
    logging.info('[step %d] Saving trainer states to %s.',
                 step_num, trainer_path)
    nlp.utils.save_states(trainer, trainer_path)


class LogTB(object):

    def __init__(self, args):
        print('--- Initializing Tensorboard')
        self.tb = SummaryWriter(logdir=os.path.join(
            args.ckpt_dir, 'log', 'train'))
        self.tb.add_text(tag='config', text=str(args), global_step=0)

    def log(self,
            student,
            context_str,
            mlm_loss,
            mlm_acc,
            teacher_ce,
            teacher_mse,
            throughput,
            lr,
            duration,
            latency,
            n_total_iter):
        logging.info(f"{context_str}loggging to Tensorboard at {n_total_iter}")
        context_str = context_str.strip()
        self.tb.add_scalar(tag=f"{context_str}/losses/mlm_loss", value=mlm_loss, global_step=n_total_iter)
        self.tb.add_scalar(tag=f"{context_str}/losses/mlm_acc", value=mlm_acc, global_step=n_total_iter)
        self.tb.add_scalar(tag=f"{context_str}/losses/teacher_ce", value=teacher_ce, global_step=n_total_iter)
        self.tb.add_scalar(tag=f"{context_str}/losses/teacher_mse", value=teacher_mse, global_step=n_total_iter)

        self.tb.add_scalar(tag=f"{context_str}/latency/throughput", value=throughput, global_step=n_total_iter)
        self.tb.add_scalar(tag=f"{context_str}/latency/duration", value=duration, global_step=n_total_iter)
        self.tb.add_scalar(tag=f"{context_str}/latency/latency", value=latency, global_step=n_total_iter)

        self.tb.add_scalar(tag=f"{context_str}/learning_rate/lr", value=lr, global_step=n_total_iter)
        self.tb.add_scalar(tag=f"{context_str}/global/memory_usage", value=psutil.virtual_memory()._asdict()['used'] / 1_000_000, global_step=n_total_iter)


def log(context_str,
        begin_time,
        running_num_tks,
        running_mlm_loss,
        running_teacher_ce_loss,
        running_teacher_mse_loss,
        step_num,
        mlm_metric,
        trainer,
        log_interval,
        model=None,
        log_tb=None,
        is_master_node=True):
    """Log training progress."""
    end_time = time.time()
    duration = end_time - begin_time
    throughput = running_num_tks / duration / 1000.0
    running_mlm_loss = running_mlm_loss / log_interval
    lr = trainer.learning_rate if trainer else -1
    # pylint: disable=line-too-long
    logging.info('{}[step {}]\tmlm_loss={:7.5f}\tmlm_acc={:4.2f}\tteacher_ce={:5.2e}'
                 '\tteacher_mse={:5.2e}\tthroughput={:.1f}K tks/s\tlr={:5.2e} time={:.2f}, latency={:.1f} ms/batch'
                 .format(context_str,
                         step_num,
                         running_mlm_loss.asscalar(),
                         mlm_metric.get()[1] * 100,
                         running_teacher_ce_loss.asscalar(),
                         running_teacher_mse_loss.asscalar(),
                         throughput.asscalar(),
                         lr,
                         duration,
                         duration * 1000 / log_interval))
    if model and log_tb:  # and is_master_node:
        log_tb.log(model,
                   context_str,
                   running_mlm_loss.asscalar(),
                   mlm_metric.get()[1] * 100,
                   running_teacher_ce_loss.asscalar(),
                   running_teacher_mse_loss.asscalar(),
                   throughput.asscalar(),
                   lr,
                   duration,
                   duration * 1000 / log_interval,
                   step_num)
    elif is_master_node:
        logging.info(f"no TB log: model: {model is None}, log_tb: {log_tb}, is_master_node: {is_master_node}")


def split_and_load(arrs, ctx):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [mx.gluon.utils.split_and_load(
        arr, ctx, even_split=False) for arr in arrs]
    return zip(*loaded_arrs)


def forward(data, model, mlm_loss, vocab_size, dtype, is_eval=False, teacher_ce_loss=None,
            mlm_weight=1.0, teacher_mse_loss=None, teacher_model=None, teacher_ce_weight=0.0, distillation_temperature=1.0):
    """forward computation for evaluation"""

    if is_eval:
        data = data
    else:
        data = data[0][0]

    (input_id, masked_id, masked_position, masked_weight,
     next_sentence_label, segment_id, valid_length) = data
    num_masks = masked_weight.sum() + 1e-8
    valid_length = valid_length.reshape(-1)
    masked_id = masked_id.reshape(-1)
    valid_length_typed = valid_length.astype(dtype, copy=False)
    # segment_id
    classified, decoded = model(input_id, valid_length_typed, masked_position)
    decoded = decoded.reshape((-1, vocab_size))
    mlm_loss_val = mlm_loss(decoded.astype('float32', copy=False),
                            masked_id, masked_weight.reshape((-1, 1)))
    # RoBERTa-style training
    mlm_loss_val = mlm_loss_val.sum() / num_masks
    teacher_ce_val = mx.nd.zeros((1,)).as_in_context(mlm_loss_val.context)
    teacher_mse_val = mx.nd.zeros((1,)).as_in_context(mlm_loss_val.context)
    if teacher_model:
        with mx.autograd.pause():
            teacher_classified, teacher_decoded = teacher_model(
                input_id, valid_length_typed, masked_position)
            teacher_decoded = teacher_decoded.reshape((-1, vocab_size))
        teacher_mse_val = teacher_mse_loss(mx.nd.softmax(decoded.astype('float32', copy=False),
                                                         temperature=distillation_temperature),
                                           mx.nd.softmax(teacher_decoded.astype('float32', copy=False),
                                                         temperature=distillation_temperature),
                                           masked_weight.reshape((-1, 1)))
        teacher_ce_val = teacher_ce_loss(mx.nd.softmax(decoded.astype('float32', copy=False),
                                                       temperature=distillation_temperature),
                                         mx.nd.softmax(teacher_decoded.astype('float32', copy=False),
                                                       temperature=distillation_temperature),
                                         masked_weight.reshape((-1, 1)))

    teacher_ce_val = distillation_temperature**2 * teacher_ce_val.sum() / num_masks
    teacher_mse_val = distillation_temperature**2 * teacher_mse_val.sum() / \
        num_masks
    loss_val = mlm_weight * mlm_loss_val + teacher_ce_weight * teacher_ce_val
    return loss_val, next_sentence_label, classified, masked_id, decoded, \
        masked_weight, mlm_loss_val, teacher_ce_val, teacher_mse_val, valid_length.astype(
            'float32', copy=False)


def evaluate(data_eval, model, mlm_loss, vocab_size, ctx, log_interval, dtype,
             mlm_weight=1.0, teacher_ce_loss=None, teacher_mse_loss=None, teacher_model=None, teacher_ce_weight=0.0,
             distillation_temperature=1.0, log_tb=None):
    """Evaluation function."""
    logging.info('Running evaluation ... ')
    mlm_metric = MaskedAccuracy()
    mlm_metric.reset()

    eval_begin_time = time.time()
    begin_time = time.time()
    step_num = 0
    running_mlm_loss = 0
    total_mlm_loss = 0
    running_teacher_ce_loss = running_teacher_mse_loss = 0
    total_teacher_ce_loss = total_teacher_mse_loss = 0
    running_num_tks = 0

    for _, dataloader in tqdm(enumerate(data_eval), desc="Evaluation"):
        step_num += 1
        data_list = [[seq.as_in_context(context) for seq in shard]
                     for context, shard in zip(ctx, dataloader)]
        loss_list = []
        ns_label_list, ns_pred_list = [], []
        mask_label_list, mask_pred_list, mask_weight_list = [], [], []
        for data in data_list:
            out = forward(data, model, mlm_loss, vocab_size, dtype, is_eval=True,
                          mlm_weight=mlm_weight,
                          teacher_ce_loss=teacher_ce_loss, teacher_mse_loss=teacher_mse_loss,
                          teacher_model=teacher_model, teacher_ce_weight=teacher_ce_weight,
                          distillation_temperature=distillation_temperature)
            (loss_val, next_sentence_label, classified, masked_id,
             decoded, masked_weight, mlm_loss_val, teacher_ce_loss_val, teacher_mse_loss_val, valid_length) = out
            loss_list.append(loss_val)
            ns_label_list.append(next_sentence_label)
            ns_pred_list.append(classified)
            mask_label_list.append(masked_id)
            mask_pred_list.append(decoded)
            mask_weight_list.append(masked_weight)

            running_mlm_loss += mlm_loss_val.as_in_context(mx.cpu())
            running_num_tks += valid_length.sum().as_in_context(mx.cpu())
            running_teacher_ce_loss += teacher_ce_loss_val.as_in_context(
                mx.cpu())
            running_teacher_mse_loss += teacher_mse_loss_val.as_in_context(
                mx.cpu())
        mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)

        # logging
        if (step_num + 1) % (log_interval) == 0:
            total_mlm_loss += running_mlm_loss
            total_teacher_ce_loss += running_teacher_ce_loss
            total_teacher_mse_loss += running_teacher_mse_loss
            log("eval ",
                begin_time,
                running_num_tks,
                running_mlm_loss,
                running_teacher_ce_loss,
                running_teacher_mse_loss,
                step_num,
                mlm_metric,
                None,
                log_interval,
                model=model,
                log_tb=log_tb)
            begin_time = time.time()
            running_mlm_loss = running_num_tks = 0
            running_teacher_ce_loss = running_teacher_mse_loss = 0
            mlm_metric.reset_local()

    mx.nd.waitall()
    eval_end_time = time.time()
    # accumulate losses from last few batches, too
    if running_mlm_loss != 0:
        total_mlm_loss += running_mlm_loss
        total_teacher_ce_loss += running_teacher_ce_loss
        total_teacher_mse_loss += running_teacher_mse_loss
    total_mlm_loss /= step_num
    total_teacher_ce_loss /= step_num
    total_teacher_mse_loss /= step_num
    logging.info('Eval mlm_loss={:.3f}\tmlm_acc={:.1f}\tteacher_ce={:.2e}\tteacher_mse={:.2e}'
                 .format(total_mlm_loss.asscalar(), mlm_metric.get_global()[1] * 100,
                         total_teacher_ce_loss.asscalar(), total_teacher_mse_loss.asscalar()))
    logging.info('Eval cost={:.1f}s'.format(eval_end_time - eval_begin_time))


def get_argparser():
    """Argument parser"""
    parser = argparse.ArgumentParser(description='Bort pretraining example.')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of optimization steps')
    parser.add_argument('--num_eval_steps', type=int,
                        default=None, help='Number of eval steps')
    parser.add_argument('--num_buckets', type=int, default=10,
                        help='Number of buckets for variable length sequence sampling')
    parser.add_argument('--dtype', type=str,
                        default='float16', help='data dtype')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='Batch size per GPU.')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='Number of batches for gradient accumulation. '
                             'The effective batch size = batch_size * accumulate.')
    parser.add_argument('--use_avg_len', action='store_true',
                        help='Use average length information for the bucket sampler. '
                             'The batch size is approximately the number of tokens in the batch')
    parser.add_argument('--batch_size_eval', type=int, default=8,
                        help='Batch size per GPU for evaluation.')
    parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                        choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                                 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                                 'wiki_cn_cased', 'openwebtext_ccnews_stories_books_cased'],
                        help='The pre-defined dataset from which the vocabulary is created. '
                             'Default is book_corpus_wiki_en_uncased.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load the pretrained model released by Google.')
    parser.add_argument('--model', type=str, default='bort_4_8_768_1024',
                        choices=[b for b in bort.predefined_borts.keys()],
                        help='Model to run pre-training on. ')
    parser.add_argument('--teacher_model', type=str, default='roberta_24_1024_16',
                        help='Model to run as teacher on. '
                             'Options are bert_12_768_12, bert_24_1024_16, roberta_24_1024_16, roberta_12_768_12, '
                             'others on https://gluon-nlp.mxnet.io/model_zoo/bert/index.html')
    parser.add_argument('--teacher_ckpt_dir', type=str, default=None,
                        help='Path to teacher checkpoint directory')
    parser.add_argument('--teacher_ce_weight', type=float, default=0.0, help='weight to mix teacher_ce_loss with '
                                                                             'mlm_loss: should be in range (0,1)')
    parser.add_argument('--distillation_temperature', type=float, default=1.0, help='temperature for teacher/student '
                                                                                    'distillation')
    parser.add_argument('--mlm_weight', type=float, default=1.0, help='weight to mix teacher_ce_loss with mlm_loss: '
                                                                      'should be in range (0,1)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data. Training is skipped if not set.')
    parser.add_argument('--data_eval', type=str, required=True,
                        help='Path to evaluation data. Evaluation is skipped if not set.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_dir',
                        help='Path to checkpoint directory')
    parser.add_argument('--start_step', type=int, default=0,
                        help='Start optimization step from the checkpoint.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.01,
                        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument('--log_interval', type=int,
                        default=250, help='Report interval')
    parser.add_argument('--ckpt_interval', type=int,
                        default=1000, help='Checkpoint interval')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose logging')
    parser.add_argument('--profile', type=str, default=None,
                        help='output profiling result to the target file')
    parser.add_argument('--cpu_only', action='store_true',
                        help='force to only use cpu')
    return parser


def generate_dev_set(tokenizer, vocab, cache_file, args):
    """Generate validation set."""
    # set random seed to generate dev data deterministically
    np.random.seed(0)
    random.seed(0)
    mx.random.seed(0)
    worker_pool = multiprocessing.Pool()
    eval_files = nlp.utils.glob(args.data_eval)
    num_files = len(eval_files)
    assert num_files > 0, 'Number of eval files must be greater than 0.' \
                          'Only found %d files at %s' % (
                              num_files, args.data_eval)
    logging.info(
        'Generating validation set from %d files on rank 0.', len(eval_files))
    create_training_instances((eval_files, tokenizer, args.max_seq_length,
                               args.short_seq_prob, args.masked_lm_prob,
                               args.max_predictions_per_seq,
                               args.whole_word_mask, vocab,
                               1, args.num_data_workers,
                               worker_pool, cache_file))
    logging.info('Done generating validation set on rank 0.')


def profile(curr_step, start_step, end_step, profile_name='profile.json',
            early_exit=True):
    """profile the program between [start_step, end_step)."""
    if curr_step == start_step:
        mx.nd.waitall()
        mx.profiler.set_config(profile_memory=False, profile_symbolic=True,
                               profile_imperative=True, filename=profile_name,
                               aggregate_stats=True)
        mx.profiler.set_state('run')
    elif curr_step == end_step:
        mx.nd.waitall()
        mx.profiler.set_state('stop')
        logging.info(mx.profiler.dumps())
        mx.profiler.dump()
        if early_exit:
            exit()
