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

__all__ = ['BortClassifier', 'BortModel', 'get_bort_model']

import os
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon import nn
import mxnet as mx
import logging

from gluonnlp.model.utils import _load_vocab, _load_pretrained_params
from gluonnlp.base import get_home_dir
from gluonnlp.model.bert import BERTModel, BERTEncoder


bort_4_8_768_1024_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 4,
    'units': 1024,
    'hidden_size': 768,
    'max_length': 512,
    'num_heads': 8,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 1024,
    'embed_dropout': 0.1,
    'word_embed': None,
    'layer_norm_eps': 1e-5
}
bort_2_16_512_1024_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 2,
    'units': 1024,
    'hidden_size': 512,
    'max_length': 512,
    'num_heads': 16,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 1024,
    'embed_dropout': 0.1,
    'word_embed': None,
    'layer_norm_eps': 1e-5
}

predefined_borts = {
    "bort_4_8_768_1024": bort_4_8_768_1024_hparams,
    "bort_2_16_512_1024": bort_2_16_512_1024_hparams,
}


class BortModel(BERTModel):
    """ Base Bort model
    """

    def __init__(self, encoder, vocab_size=None, units=None,
                 embed_size=None, embed_dropout=0.0, embed_initializer=None,
                 word_embed=None, use_decoder=True, prefix=None, params=None):
        super(BortModel, self).__init__(encoder, vocab_size=vocab_size,
                                        token_type_vocab_size=None, units=units,
                                        embed_size=embed_size, embed_dropout=embed_dropout,
                                        embed_initializer=embed_initializer,
                                        word_embed=word_embed, token_type_embed=None,
                                        use_pooler=False, use_decoder=use_decoder,
                                        use_classifier=False, use_token_type_embed=False,
                                        prefix=prefix, params=params)

    def __call__(self, inputs, valid_length=None, masked_positions=None):
        # pylint: disable=dangerous-default-value
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a Bort model.
        """
        # Temporary hack for hybridization as hybridblock does not support None
        # inputs
        valid_length = [] if valid_length is None else valid_length
        masked_positions = [] if masked_positions is None else masked_positions
        return super(BortModel, self).__call__(inputs, [], valid_length=valid_length,
                                               masked_positions=masked_positions)


class BortClassifier(HybridBlock):
    """ Bort model with a classification head
    """

    def __init__(self, bort, num_classes=2, dropout=0.1, prefix=None, params=None):
        super(BortClassifier, self).__init__(prefix=prefix, params=params)
        self.bort = bort
        self._units = bort._units

        with self.name_scope():
            self.classifier = nn.HybridSequential()
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=self._units, activation="tanh"))
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def __call__(self, inputs, valid_length=None):
        # pylint: disable=dangerous-default-value, arguments-differ
        """Generate the unnormalized score for the given the input sequences.
        """
        return super(BortClassifier, self).__call__(inputs, valid_length)

    def hybrid_forward(self, F, inputs, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.
        """
        seq_out = self.bort(inputs, valid_length)
        assert not isinstance(seq_out, (tuple, list)
                              ), 'Expected one output from BortModel'
        outputs = seq_out.slice(begin=(0, 0, 0), end=(None, 1, None))
        outputs = outputs.reshape(shape=(-1, self._units))

        return self.classifier(outputs)


def get_bort_model(model_name=None, dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                   use_decoder=True, output_attention=False, output_all_encodings=False,
                   root=os.path.join(get_home_dir(), 'models'), **kwargs):
    predefined_args = predefined_borts[model_name]
    logging.info(f"get_bort_model: {model_name}")
    mutable_args = ['use_residual', 'dropout', 'embed_dropout', 'word_embed']
    mutable_args = frozenset(mutable_args)
    print("model_name: ", model_name, ", predefined_args: ", predefined_args)
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
        'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # encoder
    encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'],
                          num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          scaled=predefined_args['scaled'],
                          dropout=predefined_args['dropout'],
                          output_attention=output_attention,
                          output_all_encodings=output_all_encodings,
                          use_residual=predefined_args['use_residual'],
                          activation=predefined_args.get('activation', 'gelu'),
                          layer_norm_eps=predefined_args.get('layer_norm_eps', None))

    from gluonnlp.vocab import Vocab
    bort_vocab = _load_vocab(dataset_name, vocab, root, cls=Vocab)

    net = BortModel(encoder, len(bort_vocab),
                    units=predefined_args['units'],
                    embed_size=predefined_args['embed_size'],
                    embed_dropout=predefined_args['embed_dropout'],
                    word_embed=predefined_args['word_embed'],
                    use_decoder=use_decoder)
    if pretrained:
        ignore_extra = not use_decoder
        _load_pretrained_params(net, model_name, dataset_name, root, ctx, ignore_extra=ignore_extra,
                                allow_missing=False)
    return net, bort_vocab
