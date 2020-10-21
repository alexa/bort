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

""" SuperGLUEBenchmark corpora.
    Adapted from https://github.com/dmlc/gluon-nlp/blob/v0.9.x/src/gluonnlp/data/super_glue.py
"""

__all__ = ['SuperGlueRTE', 'SuperGlueCB', 'SuperGlueWSC', 'SuperGlueWiC',
           'SuperGlueCOPA', 'SuperGlueMultiRC', 'SuperGlueBoolQ',
           'SuperGlueReCoRD', 'SuperGlueAXb', 'SuperGlueAXg']

import zipfile
import collections
import json
import os
import re

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from gluonnlp.data.registry import register
from gluonnlp.base import get_home_dir
from mxnet.gluon.data import SimpleDataset


def MultiRCExpansion(line_dict):
    """ MultiRC has multiple nested points
    """
    DIGITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    lines = []

    # Split and fix common typos to simplify training
    sentence = line_dict["text"].replace(": </b>", "").replace("<br>", "").replace(
        "</br>", "").replace("_", "").replace("\xad", "").split("<b>Sent")[1:]
    all_sentences = [l[2:] if l[2:][0] not in DIGITS else l[3:]
                     for l in sentence]
    questions = line_dict["questions"]

    label_to_int = lambda x: "0" if not x else "1"

    for question in questions:
        # Prototype, next use Jaccard sim to avoid passing in a long premise
        hypothesis = [question["question"]]
        for answer in question["answers"]:
            hypothesis_ = " ".join(hypothesis + [answer["text"]])
            if "isAnswer" in answer:
                lines.append([sentence,
                              hypothesis_,
                              label_to_int(answer["isAnswer"])
                              ])
            else:
                lines.append([sentence,
                              hypothesis_
                              ])
    return lines


def ReCoRDExpansion(line_dict):
    """ ReCoRD has separate answers. We'll turn it into a classification problem
        and then select the best from the set, breaking ties arbitrarily.
    """
    line_dict, question_dict = line_dict["passage"], line_dict["qas"]
    lines = []
    # Maintain the highlight annotation
    sentence = line_dict["text"].replace("\n", " ")
    entities = sorted(
        set([line_dict["text"][e["start"]:e["end"] + 1] for e in line_dict["entities"]]))

    for question in question_dict:
        question_text = question["query"]
        answers = [a["text"]
                   for a in question["answers"]] if "answers" in question else []
        pos_examples, neg_examples = [], []
        for entity in entities:
            query = question_text.replace("@placeholder", entity)
            label = 1 if entity in answers else 0
            if not answers:
                lines.append([sentence, query])
            else:
                # We want a semi-balanced dataset
                if label == 1:
                    pos_examples.append([sentence, query, label])
                else:
                    neg_examples.append([sentence, query, label])
        if answers:
            # Prioritize positive (i.e., the answers) samples.
            if len(pos_examples) > len(neg_examples):
                for l in pos_examples:
                    lines.append(l)
                for l in neg_examples:
                    lines.append(l)
            else:
                min_len = min(len(pos_examples), len(neg_examples))
                for i in range(min_len):
                    lines.append(pos_examples[i])
                    lines.append(neg_examples[i])

    return lines


def CBExpansion(line_dict, field_keys):
    """ CB has a lot of pet words. We keep all the "I think"'s since they might influence
        the label.
        Some dialogues are way too long, and most of them have the relevant passage at the end.
        Note that we could have regex'ed the proper label, since it is evident that the hypotheses
        were extracted programatically.
    """
    tmp_line = []
    FILLER_WORDS = ["Uh,", "uh,", "Um,", "um,", "you know", "I mean,"]
    for key in field_keys:
        line = line_dict[key]
        if key != "label":
            if len(line.split(".")) > 3:
                line = " ".join(line.split(".")[-4:])[1:]
            for p in FILLER_WORDS:
                line.replace(p, "")
        tmp_line.append(line)
    return [tmp_line]


class _JsonlDataset(SimpleDataset):
    """A dataset wrapping over a jsonlines (.jsonl) file, each line is a json object.
    Specific for SuperGLUE, to work with gluon==0.8.3
    Parameters
    ----------
    filename : str
        Path to the .jsonl file.
    """

    def __init__(self, filename, field_keys, task):

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._field_keys = field_keys
        self._task = task
        super(_JsonlDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            samples = []
            with open(filename, 'r') as fin:
                for line in fin.readlines():
                    line_dic = json.loads(
                        line, object_pairs_hook=collections.OrderedDict)
                    # Load the spans and multiple-choice questions as dictionaries,
                    # but ensure we use the same terminology.
                    # We've basically casted all tasks into classification
                    # tasks.
                    tmp_line = []
                    if self._task == "COPA":
                        sep = " because " if line_dic[
                            "question"] == "cause" else " so "
                        text_a = line_dic[self._field_keys[0]] + \
                            sep + line_dic[self._field_keys[1]]
                        text_b = line_dic[self._field_keys[0]] + \
                            sep + line_dic[self._field_keys[2]]
                        tmp_line.append(text_a)
                        tmp_line.append(text_b)
                        if "label" in self._field_keys:
                            tmp_line.append(line_dic["label"])
                    elif self._task == "MultiRC":
                        for l in MultiRCExpansion(line_dic["paragraph"]):
                            samples.append(l)
                        continue
                    elif self._task == "ReCoRD":
                        for l in ReCoRDExpansion(line_dic):
                            samples.append(l)
                        continue
                    elif self._task == "WSC":
                        text_a = line_dic["text"]
                        text_b = line_dic["target"][
                            "span2_text"] + " means " + line_dic["target"]["span1_text"]
                        tmp_line.append(text_a)
                        tmp_line.append(text_b)
                        if "label" in self._field_keys:
                            tmp_line.append(line_dic["label"])
                    elif self._task == "CB":
                        for l in CBExpansion(line_dic, self._field_keys):
                            samples.append(l)
                        continue
                    else:
                        for key in self._field_keys:
                            tmp_line.append(line_dic[key])
                    samples.append(tmp_line)
            samples = self._read_samples(samples)
            all_samples += samples
        return all_samples

    def _read_samples(self, samples):
        raise NotImplementedError


class _SuperGlueDataset(_JsonlDataset):

    def __init__(self, root, data_file, field_keys, task=""):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        arg_segment, zip_hash, data_hash = data_file
        segment = "val" if arg_segment == "dev" else arg_segment
        self._root = root
        filename = os.path.join(self._root, '%s.jsonl' % segment)
        self._get_data(segment, zip_hash, data_hash, filename)
        super(_SuperGlueDataset, self).__init__(
            filename, field_keys, task=task)

    def _get_data(self, arg_segment, zip_hash, data_hash, filename):
        # The GLUE API requires "dev", but these files are hashed as "val"
        if self.task == "MultiRC":
            # The MultiRC version from Gluon is quite outdated.
            # Make sure you've downloaded it and extracted it as described
            # in the README.MD file.
            print("Make sure you have downloaded the data!")
            print(
                "https://github.com/nyu-mll/jiant/blob/master/scripts/download_superglue_data.py")
        segment = "val" if arg_segment == "dev" else arg_segment
        data_filename = '%s-%s.zip' % (segment, data_hash[:8])
        if not os.path.exists(filename) and self.task != "MultiRC":
            download(_get_repo_file_url(self._repo_dir(), data_filename),
                     path=self._root, sha1_hash=zip_hash)
            # unzip
            downloaded_path = os.path.join(self._root, data_filename)
            with zipfile.ZipFile(downloaded_path, 'r') as zf:
                # skip dir structures in the zip
                for zip_info in zf.infolist():
                    if zip_info.filename[-1] == '/':
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zf.extract(zip_info, self._root)

    def _repo_dir(self):
        raise NotImplementedError


@register(segment=['train', 'dev', 'test'])
class SuperGlueRTE(_SuperGlueDataset):
    """The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual
    entailment challenges (RTE1, RTE2, RTE3 and RTE5).

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_rte"
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_rte')):
        self._segment = segment
        self._data_file = {'train': ('train', 'a4471b47b23f6d8bc2e89b2ccdcf9a3a987c69a1',
                                     '01ebec38ff3d2fdd849d3b33c2a83154d1476690'),
                           'dev': ('dev', '17f23360f77f04d03aee6c42a27a61a6378f1fd9',
                                   '410f8607d9fc46572c03f5488387327b33589069'),
                           'test': ('test', 'ef2de5f8351ef80036c4aeff9f3b46106b4f2835',
                                    '69f9d9b4089d0db5f0605eeaebc1c7abc044336b')}
        data_file = self._data_file[segment]

        if segment in ['train', 'dev']:
            field_keys = ["premise", "hypothesis", "label"]
        elif segment == 'test':
            field_keys = ["premise", "hypothesis"]

        super(SuperGlueRTE, self).__init__(root, data_file, field_keys)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/RTE'


@register(segment=['train', 'dev', 'test'])
class SuperGlueCB(_SuperGlueDataset):
    """The CommitmentBank (CB) is a corpus of short texts in which at least one sentence
    contains an embedded clause.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_cb"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_cb')):
        self._segment = segment
        self._data_file = {'train': ('train', '0b27cbdbbcdf2ba82da2f760e3ab40ed694bd2b9',
                                     '193bdb772d2fe77244e5a56b4d7ac298879ec529'),
                           'dev': ('dev', 'e1f9dc77327eba953eb41d5f9b402127d6954ae0',
                                   'd286ac7c9f722c2b660e764ec3be11bc1e1895f8'),
                           'test': ('test', '008f9afdc868b38fdd9f989babe034a3ac35dd06',
                                    'cca70739162d54f3cd671829d009a1ab4fd8ec6a')}
        data_file = self._data_file[segment]

        if segment in ['train', 'dev']:
            field_keys = ["premise", "hypothesis", "label"]
        elif segment == 'test':
            field_keys = ["premise", "hypothesis"]

        super(SuperGlueCB, self).__init__(
            root, data_file, field_keys, task="CB")

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/CB'


@register(segment=['train', 'dev', 'test'])
class SuperGlueWiC(_SuperGlueDataset):
    """
    The Word-in-Context (WiC) is a word sense disambiguation dataset cast as binary classification
    of sentence pairs. (Pilehvar and Camacho-Collados, 2019)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_wic"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_wic')):
        self._segment = segment
        self._data_file = {'train': ('train', 'ec1e265bbdcde1d8da0b56948ed30d86874b1f12',
                                     '831a58c553def448e1b1d0a8a36e2b987c81bc9c'),
                           'dev': ('dev', '2046c43e614d98d538a03924335daae7881f77cf',
                                   '73b71136a2dc2eeb3be7ab455a08f20b8dbe7526'),
                           'test': ('test', '77af78a49aac602b7bbf080a03b644167b781ba9',
                                    '1be93932d46c8f8dc665eb7af6703c56ca1b1e08')}
        data_file = self._data_file[segment]
        # We'll hope the hypernymy is clear from the sentence
        if segment in ['train', 'dev']:
            field_keys = ["sentence1", "sentence2", "label"]
        elif segment == 'test':
            field_keys = ["sentence1", "sentence2"]

        super(SuperGlueWiC, self).__init__(root, data_file, field_keys)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/WiC'


@register(segment=['train', 'dev', 'test'])
class SuperGlueBoolQ(_SuperGlueDataset):
    """
    Boolean Questions (BoolQ) is a QA dataset where each example consists of a short
    passage and a yes/no question about it.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_boolq"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_boolq')):
        self._segment = segment
        self._data_file = {'train': ('train', '89507ff3015c3212b72318fb932cfb6d4e8417ef',
                                     'd5be523290f49fc0f21f4375900451fb803817c0'),
                           'dev': ('dev', 'fd39562fc2c9d0b2b8289d02a8cf82aa151d0ad4',
                                   '9b09ece2b1974e4da20f0173454ba82ff8ee1710'),
                           'test': ('test', 'a805d4bd03112366d548473a6848601c042667d3',
                                    '98c308620c6d6c0768ba093858c92e5a5550ce9b')}
        data_file = self._data_file[segment]

        if segment in ['train', 'dev']:
            field_keys = ["passage", "question", "label"]
        elif segment == 'test':
            field_keys = ["passage", "question"]

        super(SuperGlueBoolQ, self).__init__(root, data_file, field_keys)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/BoolQ'


@register(segment=['train', 'dev', 'test'])
class SuperGlueCOPA(_SuperGlueDataset):
    """
    The Choice of Plausible Alternatives (COPA) is a causal reasoning dataset.
    (Roemmele et al., 2011)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_copa"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_copa')):
        self._segment = segment
        self._data_file = {'train': ('train', '96d20163fa8371e2676a50469d186643a07c4e7b',
                                     '5bb9c8df7b165e831613c8606a20cbe5c9622cc3'),
                           'dev': ('dev', 'acc13ad855a1d2750a3b746fb0cfe3ca6e8b6615',
                                   'c8b908d880ffaf69bd897d6f2a1f23b8c3a732d4'),
                           'test': ('test', '89347d7884e71b49dd73c6bcc317aef64bb1bac8',
                                    '735f39f3d31409d83b16e56ad8aed7725ef5ddd5')}
        data_file = self._data_file[segment]

        if segment in ['train', 'dev']:
            field_keys = ["premise", "choice1", "choice2", "label"]
        elif segment == 'test':
            field_keys = ["premise", "choice1", "choice2"]

        super(SuperGlueCOPA, self).__init__(
            root, data_file, field_keys, task="COPA")

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/COPA'


@register(segment=['train', 'dev', 'test'])
class SuperGlueMultiRC(_SuperGlueDataset):
    """
    Multi-Sentence Reading Comprehension (MultiRC) is a QA dataset.
    (Khashabi et al., 2018)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_multirc"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_multirc')):
        self._segment = segment

        # This implementation needs the actual SuperGLUE
        # data, available at:
        # https://github.com/nyu-mll/jiant/blob/master/scripts/download_superglue_data.py
        self._data_file = {'train': ('train', '', ''),
                           'dev': ('dev', '', ''),
                           'test': ('test', '', '')}
        data_file = self._data_file[segment]
        field_keys = []
        super(SuperGlueMultiRC, self).__init__(
            root, data_file, field_keys, task="MultiRC")

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/MultiRC'


@register(segment=['train', 'dev', 'test'])
class SuperGlueReCoRD(_SuperGlueDataset):
    """
    Reading Comprehension with Commonsense Reasoning Dataset (ReCoRD) is a multiple-choice
    QA dataset.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_record"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_record')):
        self._segment = segment
        self._data_file = {'train': ('train', '047282c912535c9a3bcea519935fde882feb619d',
                                     '65592074cefde2ecd1b27ce7b35eb8beb86c691a'),
                           'dev': ('dev', '442d8470bff2c9295231cd10262a7abf401edc64',
                                   '9d1850e4dfe2eca3b71bfea191d5f4b412c65309'),
                           'test': ('test', 'fc639a18fa87befdc52f14c1092fb40475bf50d0',
                                    'b79b22f54b5a49f98fecd05751b122ccc6947c81')}
        data_file = self._data_file[segment]
        field_keys = []
        super(SuperGlueReCoRD, self).__init__(
            root, data_file, field_keys, task="ReCoRD")

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/ReCoRD'


@register(segment=['train', 'dev', 'test'])
class SuperGlueWSC(_SuperGlueDataset):
    """
    The Winograd Schema Challenge (WSC) is a co-reference resolution dataset.
    (Levesque et al., 2012)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_wsc"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_wsc')):
        self._segment = segment
        self._data_file = {'train': ('train', 'ed0fe96914cfe1ae8eb9978877273f6baed621cf',
                                     'fa978f6ad4b014b5f5282dee4b6fdfdaeeb0d0df'),
                           'dev': ('dev', 'cebec2f5f00baa686573ae901bb4d919ca5d3483',
                                   'ea2413e4e6f628f2bb011c44e1d8bae301375211'),
                           'test': ('test', '3313896f315e0cb2bb1f24f3baecec7fc93124de',
                                    'a47024aa81a5e7c9bc6e957b36c97f1d1b5da2fd')}
        data_file = self._data_file[segment]

        if segment in ['train', 'dev']:
            field_keys = ["target", "text", [
                ["span1_index", "span1_text"], ["span2_index", "span2_text"]], "label"]
        elif segment == 'test':
            field_keys = ["target", "text", [
                ["span1_index", "span1_text"], ["span2_index", "span2_text"]]]

        super(SuperGlueWSC, self).__init__(
            root, data_file, field_keys, task="WSC")

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/WSC'


class SuperGlueAXb(_SuperGlueDataset):
    """
    The Broadcoverage Diagnostics (AX-b) is a diagnostics dataset labeled closely to
    the schema of MultiNLI.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    root : str, default "$MXNET_HOME/datasets/superglue_ax_b"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self, segment='test',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_ax_b')):

        if segment in ['train', 'dev']:
            raise ValueError("Only \"test\" is supported for AX-b")
        elif segment == 'test':
            field_keys = ["sentence1", "sentence2"]

        self._segment = segment
        self._data_file = {'test': ('AX-b', '398c5a376eb436f790723cd217ac040334140000',
                                    '50fd8ac409897b652daa4b246917097c3c394bc8')}
        data_file = self._data_file[segment]

        super(SuperGlueAXb, self).__init__(root, data_file, field_keys)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/AX-b'


class SuperGlueAXg(_SuperGlueDataset):
    """
    The Winogender Schema Diagnostics (AX-g) is a diagnostics dataset labeled closely to
    the schema of MultiNLI.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    root : str, default "$MXNET_HOME/datasets/superglue_ax_g"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'
    """

    def __init__(self,  segment='test',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_ax_g')):

        if segment in ['train', 'dev']:
            raise ValueError("Only \"test\" is supported for AX-g")
        elif segment == 'test':
            field_keys = ["premise", "hypothesis"]

        self._segment = segment
        self._data_file = {"test": ('AX-g', 'd8c92498496854807dfeacd344eddf466d7f468a',
                                    '8a8cbfe00fd88776a2a2f20b477e5b0c6cc8ebae')}
        data_file = self._data_file[segment]

        super(SuperGlueAXg, self).__init__(root, data_file, field_keys)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/AX-g'
