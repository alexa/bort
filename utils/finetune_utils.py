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

__all__ = ['convert_examples_to_features', 'preprocess_data', 'ALL_TASKS_DIC', 'EXTRA_METRICS',
           'RACEHash', 'ReCoRDHash', 'process_MultiRC_answers', 'process_ReCoRD_answers', 'process_RACE_answers']

# Classes being called directly by finetune_bort.py

import os
import re
import json
import logging
import fnmatch
import mxnet as mx
import gluonnlp as nlp
import numpy as np

from collections import OrderedDict
from functools import partial
from mxnet import gluon, nd
from gluonnlp.base import get_home_dir

from data.preprocessing_utils import truncate_seqs_equal, concat_sequences

from data.classification import MRPCTask, QQPTask, RTETask, STSBTask, SSTTask
from data.classification import QNLITask, CoLATask, MNLITask, WNLITask
from data.classification import SuperGLUERTETask, CBTask, WSCTask, WiCTask, COPATask
from data.classification import MultiRCTask, BoolQTask, ReCoRDTask, AXbTask, AXgTask
from data.classification import RACEHTask, RACEMTask

from .metrics import AvgF1, GP


ALL_TASKS_DIC = {
    'MRPC': MRPCTask(),
    'QQP': QQPTask(),
    'QNLI': QNLITask(),
    'RTE': RTETask(),
    'STS-B': STSBTask(),
    'CoLA': CoLATask(),
    'MNLI': MNLITask(),
    'WNLI': WNLITask(),
    'SST': SSTTask(),
    'SuperGLUERTE': SuperGLUERTETask(),
    'CB': CBTask(),
    'WSC': WSCTask(),
    "WiC": WiCTask(),
    "COPA": COPATask(),
    "MultiRC": MultiRCTask(),
    "BoolQ": BoolQTask(),
    "ReCoRD": ReCoRDTask(),
    "AXb": AXbTask(),
    "AXg": AXgTask(),
    "RACE-H": RACEHTask(),
    "RACE-M": RACEMTask()
}


EXTRA_METRICS = {
    "CB": [("avg_f1", AvgF1())],
    "AXg": [("GP", GP())],
}


def do_log(batch_id, batch_num, metric, step_loss, log_interval, epoch_id=None, learning_rate=None):
    """Generate and print out the log messages. """
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    if epoch_id is None:
        eval_str = '[Batch %d/%d] loss=%.4f, metrics:' + \
            ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(eval_str, batch_id + 1, batch_num,
                     step_loss / log_interval, *metric_val)
    else:
        train_str = '[Epoch %d Batch %d/%d] loss=%.4f, lr=%.10f, metrics:' + \
                    ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(train_str, epoch_id + 1, batch_id + 1, batch_num, step_loss / log_interval,
                     learning_rate, *metric_val)


def convert_examples_to_features(example, tokenizer=None, truncate_length=512, cls_token=None,
                                 sep_token=None, class_labels=None, label_alias=None, vocab=None,
                                 is_test=False):
    """Convert GLUE/SuperGLUE classification and regression examples into 
        the necessary features"""
    if not is_test:
        label_dtype = 'int32' if class_labels else 'float32'
        example, label = example[:-1], example[-1]
        # create label maps if classification task
        if class_labels:
            label_map = {}
            for (i, l) in enumerate(class_labels):
                label_map[l] = i
            if label_alias:
                for key in label_alias:
                    label_map[key] = label_map[label_alias[key]]
            # Fix for BoolQ, WSC, and MultiRC, json values get loaded as boolean and not as string
            # assignments.
            if type(label) == bool:
                label = "true" if label else "false"
            # Fix for COPA
            if type(label) == int:
                label = "0" if label == 0 else "1"
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)
    tokens_raw = [tokenizer(l) for l in example]
    tokens_trun = truncate_seqs_equal(tokens_raw, truncate_length)
    tokens_trun[0] = [cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = concat_sequences(
        tokens_trun, [[sep_token]] * len(tokens_trun))
    input_ids = vocab[tokens]
    valid_length = len(input_ids)

    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length


def preprocess_data(tokenizer, task, batch_size, dev_batch_size, max_len, vocab, world_size=None):
    """Train/eval Data preparation function."""
    label_dtype = 'int32' if task.class_labels else 'float32'
    truncate_length = max_len - 3 if task.is_pair else max_len - 2
    trans = partial(convert_examples_to_features, tokenizer=tokenizer,
                    truncate_length=truncate_length,
                    cls_token=vocab.bos_token,
                    sep_token=vocab.eos_token,
                    class_labels=task.class_labels,
                    label_alias=task.label_alias, vocab=vocab)

    # task.dataset_train returns (segment_name, dataset)
    train_tsv = task.dataset_train()[1]
    data_train = mx.gluon.data.SimpleDataset(list(map(trans, train_tsv)))
    data_train_len = data_train.transform(lambda _, segment_ids, valid_length, label: valid_length,
                                          lazy=False)
    # bucket sampler for training
    pad_val = vocab[vocab.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val),
        nlp.data.batchify.Pad(axis=0, pad_val=0),
        nlp.data.batchify.Stack(),  # length
        nlp.data.batchify.Stack(label_dtype))  # label

    if world_size is not None:
        batch_sampler = nlp.data.sampler.FixedBucketSampler(data_train_len, batch_size=batch_size,
                                                            num_buckets=15, ratio=0, shuffle=True,
                                                            num_shards=world_size)
        loader_train = nlp.data.ShardedDataLoader(dataset=data_train, num_workers=4,
                                                  batch_sampler=batch_sampler, batchify_fn=batchify_fn)
    else:
        batch_sampler = nlp.data.sampler.FixedBucketSampler(data_train_len, batch_size=batch_size,
                                                            num_buckets=15, ratio=0, shuffle=True)
        loader_train = mx.gluon.data.DataLoader(dataset=data_train, num_workers=4,
                                                batch_sampler=batch_sampler, batchify_fn=batchify_fn)

    # data dev. For MNLI, more than one dev set is available
    dev_tsv = task.dataset_dev()
    dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
    loader_dev_list = []
    for segment, data in dev_tsv_list:
        data_dev = mx.gluon.data.SimpleDataset(list(map(trans, data)))
        loader_dev = mx.gluon.data.DataLoader(data_dev, batch_size=dev_batch_size, num_workers=4,
                                              shuffle=False, batchify_fn=batchify_fn)
        loader_dev_list.append((segment, loader_dev))

    # batchify for data test
    test_batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val),
        nlp.data.batchify.Pad(axis=0, pad_val=0),
        nlp.data.batchify.Stack())
    # transform for data test
    test_trans = partial(convert_examples_to_features, tokenizer=tokenizer, truncate_length=truncate_length,
                         cls_token=vocab.bos_token,
                         sep_token=vocab.eos_token,
                         class_labels=None, is_test=True, vocab=vocab)

    # data test. For MNLI, more than one test set is available
    test_tsv = task.dataset_test()
    test_tsv_list = test_tsv if isinstance(test_tsv, list) else [test_tsv]
    loader_test_list = []
    for segment, data in test_tsv_list:
        data_test = mx.gluon.data.SimpleDataset(list(map(test_trans, data)))
        loader_test = mx.gluon.data.DataLoader(data_test, batch_size=dev_batch_size, num_workers=4,
                                               shuffle=False, batchify_fn=test_batchify_fn)
        loader_test_list.append((segment, loader_test))

    # Return data_dev for ReCoRD and MultiRC
    return loader_train, data_dev, loader_dev_list, loader_test_list, len(data_train)


def MultiRCHash(test_dataset_location):
    """ MultiRC has multiple nested points. Return a list of dictionaries with
        the predictions to fill out.
    """
    dataset = [json.loads(l, object_pairs_hook=OrderedDict)
               for l in open(test_dataset_location, 'r', encoding='utf8').readlines()]
    line_dict = [(l["idx"], l["passage"]) for l in dataset]
    lines = []

    for idx, line in line_dict:
        questions = line["questions"]
        line_hashes = {"idx": idx, "passage": {"questions": []}}
        for question in questions:
            question_dict = {"idx": question["idx"], "answers": []}
            for answer in question["answers"]:
                question_dict["answers"].append(
                    {"idx": answer["idx"], "label": 0})
            line_hashes["passage"]["questions"].append(question_dict)

        lines.append(line_hashes)

    return lines


def ReCoRDHash(dataset_location):
    """ Because of the way we've setup ReCoRD, we need to figure out a way to translate
        it back into a viable answer. 
    """
    dataset = [json.loads(l, object_pairs_hook=OrderedDict)
               for l in open(dataset_location, 'r', encoding='utf8').readlines()]
    is_test = "test" in dataset_location

    all_lines = [(l["idx"], l["passage"], l["qas"]) for l in dataset]
    lines = {}
    for idx, line, qas in all_lines:
        entities = sorted(
            set([line["text"][e["start"]:e["end"] + 1] for e in line["entities"]]))
        for question in qas:
            tmp_lines = []
            answers = None if is_test else [
                ans["text"] for ans in question["answers"]]
            for entity in entities:
                is_answer = False
                if not is_test:
                    is_answer = entity in answers
                tmp_lines.append(
                    {"idx": question["idx"], "label": entity, "is_answer": is_answer})
            lines[question["idx"]] = tmp_lines
    return lines


def RACEHash(dataset_location, task_name, segment='test'):
    """ Because of the way we've setup RACE-H/RACE-M, we need to figure out a way to
        translate it back into a viable answer. 
    """
    if dataset_location is None:
        dataset_location = os.path.join(get_home_dir(), 'datasets', 'race')

    task = "high" if task_name[-1] == "H" else "middle"
    test_dataset_location = os.path.join(dataset_location, segment, task)
    filenames = [os.path.expanduser(f) for f in os.listdir(
        test_dataset_location) if fnmatch.fnmatch(f, '*.txt')]
    filenames.sort()
    dataset = []
    for f in filenames:
        dataset += [json.loads(l, object_pairs_hook=OrderedDict)
                    for l in open(os.path.join(test_dataset_location, f), 'r').readlines()]
    return dataset


def process_ReCoRD_answers(results, result_data):
    # In practice we should get the max confidence over the question space.
    # First assign label and confidence to every single point on the set, then
    # prune out low-confidence elements.
    tmp_results = []
    start_index = 0
    preds, label = [], []
    for i in range(len(result_data)):
        candidate_result_array = result_data[i]
        results_subarray = results[
            start_index:start_index + len(candidate_result_array)]
        idx, max_confidence = 0, -np.inf
        backup_idx, backup_max_confidence = 0, -np.inf
        for j in range(len(results_subarray)):
            score, logits = results_subarray[j][0], results_subarray[j][-1]
            if score == 1 and logits[-1] > max_confidence:
                idx = j
            else:
                if logits[-1] > backup_max_confidence:
                    backup_idx = j
                    backup_max_confidence = logits[-1]
        if max_confidence == -np.inf:
            idx = backup_idx
        chosen_candidate = candidate_result_array[idx]
        preds.append(chosen_candidate["label"])
        label.append(chosen_candidate["label"] if candidate_result_array[
                     idx]["is_answer"] else "glorp")

        chosen_candidate.pop("is_answer", None)
        tmp_results.append(chosen_candidate)

        start_index = start_index + len(results_subarray)

    # This number is meaningless in test (all eval to False), and
    # might have high false negatives
    score = sum([p == l for (p, l) in zip(preds, label)]) / len(preds)
    return tmp_results, score


def process_MultiRC_answers(results, test_dataset_location):
    # "Re-roll" the unrolled prediction vector into the required format.
    result_data = MultiRCHash(test_dataset_location)
    p_idx, q_idx, a_idx = 0, 0, 0

    for label in results:
        if len(result_data[p_idx]["passage"]["questions"][q_idx]["answers"]) == a_idx:
            a_idx = 0
            q_idx += 1
        if len(result_data[p_idx]["passage"]["questions"]) == q_idx:
            q_idx = 0
            p_idx += 1
        result_data[p_idx]["passage"]["questions"][
            q_idx]["answers"][a_idx]["label"] = int(label)
        a_idx += 1

    return result_data


def process_RACE_answers(result_data, results):
    # In practice we should get the max confidence over the question space.
    # First assign label and confidence to every single point on the set, then
    # prune out low-confidence elements.

    IDX_TO_ANSWER = {"0": "A", "1": "B", "2": "C", "3": "D"}

    tmp_results = []
    start_index = 0
    total, correct = 0, 0

    for line in result_data:

        new_line = {k: v for k, v in line.items()}
        new_line["answers"] = []

        for question, prediction in zip(new_line["questions"], results[start_index:start_index + len(line["answers"])]):
            label = IDX_TO_ANSWER[prediction[0]]
            new_line["answers"].append(label)
        start_index += len(line["answers"])

        if "answers" in line:
            for pred, label in zip(new_line["answers"], line["answers"]):
                if pred == label:
                    correct += 1
                total += 1

        tmp_results.append(new_line)

    class_accuracy = correct / total if total != 0 else 0

    # Class accuracy is bugged, but we only need the actual accuracy anyway
    return tmp_results, class_accuracy
