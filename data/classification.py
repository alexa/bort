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
    Utils for classification-based tasks.
    Mostly taken from https://github.com/dmlc/gluon-nlp/blob/v0.9.x/scripts/bert/data/classification.py
"""

from __future__ import absolute_import

__all__ = [
    'MRPCTask', 'QQPTask', 'QNLITask', 'RTETask', 'STSBTask',
    'CoLATask', 'MNLITask', 'WNLITask', 'SSTTask', 'SuperGLUERTETask',
    'CBTask', 'WSCTask', 'WiCTask', 'COPATask', 'MultiRCTask', 'BoolQTask',
    'ReCoRDTask', 'AXbTask', 'AXgTask', 'RACEHTask', 'RACEMTask'
]

from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric, CustomMetric

from .glue import GlueCoLA, GlueSST2, GlueSTSB, GlueMRPC, GlueQQP, GlueRTE, GlueMNLI, GlueQNLI, GlueWNLI
from .super_glue import SuperGlueRTE, SuperGlueCB, SuperGlueWSC, SuperGlueWiC, SuperGlueCOPA, \
    SuperGlueMultiRC, SuperGlueBoolQ, SuperGlueReCoRD, SuperGlueAXb, SuperGlueAXg
from .race import RACEH, RACEM


class BaseClassificationTask(object):
    """Abstract classification task class.
       We want to use the same API on our fine tuning code, so we generalize it here,
       instead of in the call (as done in Gluon). 

    Parameters
    ----------
    class_labels : list of str, or None
        Classification labels of the task.
        Set to None for regression tasks with continuous real values.
    metrics : list of EValMetric
        Evaluation metrics of the task.
    is_pair : bool
        Whether the task deals with sentence pairs or single sentences.
    label_alias : dict
        label alias dict, some different labels in dataset actually means
        the same. e.g.: {'contradictory':'contradiction'} means contradictory
        and contradiction label means the same in dataset, they will get
        the same class id.
    output_format : str
        The format the predictions file will be generated as.
    """

    def __init__(self, class_labels, metrics, is_pair, output_format="tsv", label_alias=None):
        self.class_labels = class_labels
        self.metrics = metrics
        self.is_pair = is_pair
        self.label_alias = label_alias
        self.output_format = output_format

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for the task.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments.

        Returns
        -------
        Dataset : the dataset of target segment.
        """
        raise NotImplementedError()

    def dataset_train(self):
        """Get the training segment of the dataset for the task.

        Returns
        -------
        tuple of str, Dataset : the segment name, and the dataset.
        """
        return 'train', self.get_dataset(segment='train')

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        tuple of (str, Dataset), or list of tuple : the segment name, and the dataset.
        """
        return 'dev', self.get_dataset(segment='dev')

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        tuple of (str, Dataset), or list of tuple : the segment name, and the dataset.
        """
        return 'test', self.get_dataset(segment='test')


class MRPCTask(BaseClassificationTask):
    """The MRPC task on GlueBenchmark."""

    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(F1())
        metric.add(Accuracy())
        super(MRPCTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for MRPC.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueMRPC(segment=segment)


class QQPTask(BaseClassificationTask):
    """The Quora Question Pairs task on GlueBenchmark."""

    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(F1())
        metric.add(Accuracy())
        super(QQPTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for QQP.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueQQP(segment=segment)


class RTETask(BaseClassificationTask):
    """The Recognizing Textual Entailment task on GlueBenchmark."""

    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = Accuracy()
        super(RTETask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for RTE.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueRTE(segment=segment)


class QNLITask(BaseClassificationTask):
    """The SQuAD NLI task on GlueBenchmark."""

    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = Accuracy()
        super(QNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for QNLI.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueQNLI(segment=segment)


class STSBTask(BaseClassificationTask):
    """The Sentence Textual Similarity Benchmark task on GlueBenchmark."""

    def __init__(self):
        is_pair = True
        class_labels = None
        metric = PearsonCorrelation()
        super(STSBTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for STSB

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueSTSB(segment=segment)


class CoLATask(BaseClassificationTask):
    """The Warstdadt acceptability task on GlueBenchmark."""

    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        metric = MCC(average='micro')
        super(CoLATask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for CoLA

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueCoLA(segment=segment)


class SSTTask(BaseClassificationTask):
    """The Stanford Sentiment Treebank task on GlueBenchmark."""

    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        metric = Accuracy()
        super(SSTTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for SST

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueSST2(segment=segment)


class WNLITask(BaseClassificationTask):
    """The Winograd NLI task on GlueBenchmark."""

    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = Accuracy()
        super(WNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for WNLI

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev', 'test', 'train'
        """
        return GlueWNLI(segment=segment)


class MNLITask(BaseClassificationTask):
    """The Multi-Genre Natural Language Inference task on GlueBenchmark."""

    def __init__(self):
        is_pair = True
        class_labels = ['neutral', 'entailment', 'contradiction']
        metric = Accuracy()
        super(MNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for MNLI

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev_matched', 'dev_mismatched', 'test_matched',
            'test_mismatched', 'train'
        """
        return GlueMNLI(segment=segment)

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        list of TSVDataset : the dataset of the dev segment.
        """
        return [('dev_matched', self.get_dataset(segment='dev_matched')),
                ('dev_mismatched', self.get_dataset(segment='dev_mismatched'))]

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        list of TSVDataset : the dataset of the test segment.
        """
        return [('test_matched', self.get_dataset(segment='test_matched')),
                ('test_mismatched', self.get_dataset(segment='test_mismatched'))]


class SuperGLUERTETask(BaseClassificationTask):
    """ The SuperGLUE version of RTE"""

    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = Accuracy()
        super(SuperGLUERTETask, self).__init__(
            class_labels, metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueRTE(segment=segment)


class CBTask(BaseClassificationTask):
    """ The The CommitmentBank (CB) dataset for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['neutral', 'entailment', 'contradiction']
        metric = Accuracy()
        super(CBTask, self).__init__(class_labels,
                                     metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueCB(segment=segment)


class BoolQTask(BaseClassificationTask):
    """ The Boolean Questions (BoolQ) task for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['false', 'true']
        metric = Accuracy()
        super(BoolQTask, self).__init__(class_labels,
                                        metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueBoolQ(segment=segment)


class WiCTask(BaseClassificationTask):
    """ The Word-in-Context (WiC) task for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['false', 'true']
        metric = Accuracy()
        super(WiCTask, self).__init__(class_labels,
                                      metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueWiC(segment=segment)


class WSCTask(BaseClassificationTask):
    """ The Winograd Schema Challenge (WSC) task for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['false', 'true']
        metric = Accuracy()
        super(WSCTask, self).__init__(class_labels,
                                      metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueWSC(segment=segment)


class COPATask(BaseClassificationTask):
    """ The Choice of Plausible Alternatives (COPA) task for SuperGLUE """

    def __init__(self):
        # Technically there's two other sentences.
        is_pair = True
        class_labels = ['0', '1']
        metric = Accuracy()
        super(COPATask, self).__init__(class_labels,
                                       metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueCOPA(segment=segment)


class MultiRCTask(BaseClassificationTask):
    """ The Multi-Sentence Reading Comprehension (MultiRC) task for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(F1(average='micro'))
        super(MultiRCTask, self).__init__(class_labels,
                                          metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueMultiRC(segment=segment)


class ReCoRDTask(BaseClassificationTask):
    """ The Reading Comprehension with Commonsense Reasoning Dataset (ReCoRD) task 
        for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(F1())
        metric.add(Accuracy())
        super(ReCoRDTask, self).__init__(class_labels,
                                         metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return SuperGlueReCoRD(segment=segment)


class AXbTask(BaseClassificationTask):
    """ The Broadcoverage Diagnostics (AX-b) task for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = MCC(average='micro')
        super(AXbTask, self).__init__(class_labels,
                                      metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='test'):
        """Get the corresponding dataset
        """
        # Only one segment
        return SuperGlueAXb(segment=segment)


class AXgTask(BaseClassificationTask):
    """ The Winogender Schema Diagnostics (AX-g) task for SuperGLUE """

    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = CompositeEvalMetric()
        metric.add(Accuracy())
        super(AXgTask, self).__init__(class_labels,
                                      metric, is_pair, output_format="jsonl")

    def get_dataset(self, segment='test'):
        """Get the corresponding dataset
        """
        # Only one segment
        return SuperGlueAXg(segment=segment)


class RACEHTask(BaseClassificationTask):
    """ The Reading Comprehension Dataset (High) """

    def __init__(self):
        is_pair = True
        class_labels = ['0', '1', '2', '3']
        # We will also use class-accuracy (i.e., the true performance)
        metric = Accuracy()
        self._root = None

        super(RACEHTask, self).__init__(class_labels,
                                        metric, is_pair, output_format="txt")

    def _set_dataset_location(self, root):
        self._root = root

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return RACEH(segment=segment, root=self._root)


class RACEMTask(BaseClassificationTask):
    """ The Reading Comprehension Dataset (High) """

    def __init__(self):
        is_pair = True
        class_labels = ['0', '1', '2', '3']
        # We will also use class-accuracy (i.e., the true performance)
        metric = Accuracy()
        self._root = None

        super(RACEMTask, self).__init__(class_labels,
                                        metric, is_pair, output_format="txt")

    def _set_dataset_location(self, root):
        self._root = root

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset
        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return RACEM(segment=segment, root=self._root)
