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

"""RACE Benchmark corpora."""

__all__ = ['RACEH', 'RACEM']

import collections
import fnmatch
import json
import os
import re

from gluonnlp.data.registry import register
from gluonnlp.base import get_home_dir
from mxnet.gluon.data import SimpleDataset


LETTER_TO_IDX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}


def jaccard_similarity(x, y):
    s1, s2 = set(x), set(y)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def RACEExpansion(line, is_test=False):
    """ Each line is comprised of a dictionary with multiple-choice answers, like MultiRC
    """
    expanded_lines = []
    passage = line["article"]
    options = line["options"]
    questions = line["questions"]
    answers = ["" for _ in options] if is_test else line["answers"]

    for (question, ans, opts) in zip(questions, answers, options):
        passage_ = passage + " " + question
        opt_ = " </sep> ".join([o for o in opts])
        if not is_test:
            expanded_lines.append([passage_, opt_, str(LETTER_TO_IDX[ans])])
        else:
            expanded_lines.append([passage_, opt_])

    return expanded_lines


class _TextDataset(SimpleDataset):
    """A dataset wrapping over multiple .txt files, each line is a json object.
    Specific for RACE, to work with gluon==0.8.3
    Parameters
    ----------
    filename : str
        Path to the .txt files.
    """

    def __init__(self, filenames, segment):

        if not isinstance(filenames, (tuple, list)):
            filenames = (filenames, )
        self._filenames = [os.path.expanduser(f) for f in filenames]
        self._filenames.sort()
        self._segment = segment
        self._is_test = self._segment == "test"
        super(_TextDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            samples = []
            with open(filename, 'r') as fin:
                for line in fin.readlines():
                    line_dic = json.loads(
                        line, object_pairs_hook=collections.OrderedDict)
                    for l in RACEExpansion(line_dic, is_test=self._is_test):
                        samples.append(l)
            samples = self._read_samples(samples)
            all_samples += samples
        return all_samples

    def _read_samples(self, samples):
        raise NotImplementedError


class _RACEDataset(_TextDataset):

    def __init__(self, root, segment, task):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment

        file_path = os.path.join(self._root, segment, task)
        filenames = [os.path.join(file_path, f) for f in os.listdir(
            file_path) if fnmatch.fnmatch(f, '*.txt')]

        super(_RACEDataset, self).__init__(filenames, segment=self._segment)

    def _repo_dir(self):
        raise NotImplementedError


@register(segment=['train', 'dev', 'test'])
class RACEH(_RACEDataset):
    """The RACE: Large-scale ReAding Comprehension Dataset From Examinations dataset,
    from https://arxiv.org/pdf/1704.04683.pdf. Dataset is available upon request to the authors.
    This is the class corresponding to the highschool version.

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/race"
        Path to folder where the datasets are.
        MXNET_HOME defaults to '~/.mxnet'.
    """

    def __init__(self, segment='train', root=None):
        self._segment = segment
        if root is None:
            root = os.path.join(get_home_dir(), 'datasets', 'race')
        super(RACEH, self).__init__(root, segment, "high")

    def _read_samples(self, samples):
        return samples


@register(segment=['train', 'dev', 'test'])
class RACEM(_RACEDataset):
    """The RACE: Large-scale ReAding Comprehension Dataset From Examinations dataset,
    from https://arxiv.org/pdf/1704.04683.pdf. Dataset is available upon request to the authors.
    This is the class corresponding to the middle school version.

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/race"
        Path to folder where the datasets are.
        MXNET_HOME defaults to '~/.mxnet'.
    """

    def __init__(self, segment='train', root=None):
        self._segment = segment
        if root is None:
            root = os.path.join(get_home_dir(), 'datasets', 'race')
        super(RACEM, self).__init__(root, segment, "middle")

    def _read_samples(self, samples):
        return samples
