# Bort
##### Companion code for the paper "Optimal Subarchitecture Extraction for BERT."

Bort is an optimal subset of architectural parameters for the BERT architecture, extracted by applying a fully polynomial-time approximation scheme (FPTAS) for neural architecture search. Bort has an effective (that is, not counting the embedding layer) size of 5.5\% the original BERT-large architecture, and 16\% of the net size. It is also able to be pretrained in 288 GPU hours, which is 1.2\% of the time required to pretrain the highest-performing BERT parametric architectural variant, RoBERTa-large.
It is also 7.9x faster on a CPU, and performs better than other compressed variants of the architecture, and some of the non-compressed variants; it obtains an average performance improvement of between 0.3\% and 31\%, absolute with respect to BERT-large on multiple public natural language understanding (NLU) benchmarks.

Here are the corresponding GLUE scores on the test set:

|Model|Score|CoLA|SST-2|MRPC|STS-B|QQP|MNLI-m|MNLI-mm|QNLI(v2)|RTE|WNLI|AX|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Bort      |83.6|63.9|96.2|94.1/92.3|89.2/88.3|66.0/85.9|88.1|87.8|92.3|82.7|71.2|51.9|
|BERT-Large|80.5|60.5|94.9|89.3/85.4|87.6/86.5|72.1/89.3|86.7|85.9|92.7|70.1|65.1|39.6|


And SuperGLUE scores on the test set:

|Model|Score|BoolQ|CB|COPA|MultiRC|ReCoRD|RTE|WiC|WSC|AX-b|AX-g|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Bort       |74.1|83.7|81.9/86.5|89.6|83.7/54.1|49.8/49.0|81.2|70.1|65.8|48.0|96.1/61.5|
|BERT-Large|69.0|77.4|75.7/83.6|70.6|70.0/24.1|72.0/71.3|71.7|69.6|64.4|23.0|97.8/51.7


And here are the architectural parameters:

|Model|Parameters (M) |Layers |Attention heads|Hidden size| Intermediate size| Embedding size (M) | Encoder proportion (%)|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Bort       |56 |  4| 8  | 768  | 1024 | 39  | 30.3 |
|BERT-Large|340| 24| 16 | 1024 | 4096 | 31.8| 90.6 |


## Setup:
1. You need to install the requirements from the `requirements.txt` file:
```
pip install -r requirements.txt
```
This code has been tested with Python 3.6.5+.
To save yourself some headache we recommend you install Horovod from source, _after_ you install MxNet. This is only needed if you are pre-training the architecture. For this, run the following commands (you'll need a C++ compiler which supports c++11 standards, like gcc > 4.8):
```bash
    pip uninstall horovod
    HOROVOD_CUDA_HOME=/usr/local/cuda-10.1 \
    HOROVOD_WITH_MXNET=1 \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    pip install horovod==0.16.2 --no-cache-dir
```

2. You also need to download the model from [here](https://alexa-saif-bort.s3.amazonaws.com/bort.params). If you have the AWS CLI, all you need to do is run:
```
aws s3 cp s3://alexa-saif-bort/bort.params model/
```

3. To run the tests, you also need to download the sample text from [Gluon](https://github.com/dmlc/gluon-nlp/blob/v0.9.x/scripts/bert/sample_text.txt) and put it in `test_data/`:
```
wget https://github.com/dmlc/gluon-nlp/blob/v0.9.x/scripts/bert/sample_text.txt
mv sample_text.txt test_data/
```



## Pre-training:

Bort is already pre-trained, but if you want to try out other datasets, you can follow the steps here. Note that this does not run the FPTAS described in the paper, and works for a fixed architecture (Bort).

1. First, you will need to tokenize the pre-training text:
```bash
python create_pretraining_data.py \
            --input_file <input text> \
            --output_dir <output directory> \
            --dataset_name <dataset name> \
            --dupe_factor <duplication factor> \
            --num_outputs <number of output files>
```
We recommend using `--dataset_name  openwebtext_ccnews_stories_books_cased` for the vocabulary.
If your data file is too large, the script will throw out-of-memory errors. We recommend splitting it into smaller chunks and then calling the script one-by-one.

2. Then run the pre-training distillation script:
```bash
./run_pretraining_distillation.sh <num gpus> <training data> <testing data> [optional teacher checkpoint]
```
Please see the contents of `run_pretraining_distillation.sh` for example usages and additional optional configuration. If you have installed Horovod, we highly recommend you use `run_pretraining_distillation_hvd.py` instead.

## Fine-tuning:

1. To fine-tune Bort, run:
```bash
./run_finetune.sh <your task here>
```
We recommend you play with the hyperparameters from  `run_finetune.sh`.
This code supports all the tasks outlined in the paper, but for the case of the RACE dataset, you need to [download](http://www.cs.cmu.edu/~glai1/data/race/) the data and extract it. The default location for extraction is `~/.mxnet/datasets/race`. Same goes for SuperGLUE's MultiRC, since the Gluon implementation is the old version. You can [download](https://github.com/nyu-mll/jiant/blob/master/scripts/download_superglue_data.py) the data and extract it to `~/.mxnet/datasets/superglue_multirc/`.

 It is normal to get very odd results for the fine-tuning step, since this repository only contains the training part of Agora.
However, you can easily implement your own version of that algorithm.
We recommend you use the following initial set of hyperparameters, and follow the requirements described in the papers at the end of this file:
```
seeds={0,1,2,3,4}
learning_rates={1e-4, 1e-5, 9e-6}
weight_decays={0, 10, 100, 350}
warmup_rates={0.35, 0.40, 0.45, 0.50}
batch_sizes={8, 16}
```



## Troubleshooting:
##### Dependency errors
Bort requires a rather unusual environment to run. For this reason, most of the problems regarding runtime can be fixed by installing the requirements from the `requirements.txt` file. Also make sure to have reinstalled Horovod as outlined above.
##### Script failing when downloading the data
This is inherent to the way Bort is fine-tuned, since it expects the data to be preexisting for some arbitrary implementation of Agora. You can get around that error by downloading the data before running the script, e.g.:
```
from data.classification import BoolQTask
task = BoolQTask()
task.dataset_train()[1]; task.dataset_val()[1]; task.dataset_test()[1]
```
##### Out-of-memory errors
While Bort is designed to be efficient in terms of the space it occupies in memory, a very large batch size or sequence length will still cause you to run out of memory. More often than ever, reducing the sequence length from `512` to `256` will solve out-of-memory issues. 80% of the time, it works every time.
##### Slow fine-tuning/pre-training
We strongly recommend using distributed training for both fine-tuning and pre-training. If your Horovod acts weird, remember that it needs to be built _after_ the installation of MXNet (or any framework for that matter).
##### Low task-specific performance
If you observe near-random task-specific performance, that is to be expected. Bort is a rather small architecture and the optimizer/scheduler/learning rate combination is quite aggressive. We _highly_ recommend you fine-tune Bort using an implementation of Agora. More details on how to do that are in the references below, specifically the second paper. Note that we needed to implement "replay" (i.e., re-doing some iterations of Agora) to get it to converge better.


## References
If you use Bort or the other algorithms in your work, we'd love to hear from it! Also, please cite the so-called "Bort trilogy" papers:
```
@article{deWynterApproximation,
    title={An Approximation Algorithm for Optimal Subarchitecture Extraction},
    author={Adrian de Wynter},
    year={2020},
    eprint={2010.08512},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    journal={CoRR},
    volume={abs/2010.085122},
    url={http://arxiv.org/abs/2010.085122}
}
```
```
@article{deWynterAlgorithm,
      title={An Algorithm for Learning Smaller Representations of Models With Scarce Data},
      author={Adrian de Wynter},
      year={2020},
      eprint={2010.07990},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      journal={CoRR},
      volume={abs/2010.07990},
      url={http://arxiv.org/abs/2010.07990}
}
```
```
@article{deWynterPerryOptimal,
      title={Optimal Subarchitecture Extraction for BERT},
      author={Adrian de Wynter and Daniel J. Perry},
      year={2020},
      eprint={2010.10499},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      journal={CoRR},
      volume={abs/2010.10499},
      url={http://arxiv.org/abs/2010.10499}
}
```
Lastly, if you use the GLUE/SuperGLUE/RACE tasks, don't forget to give proper attribution to the original authors.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
