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

""" Custom metrics """

__all__ = ["AvgF1", "GP"]

from sklearn.metrics import f1_score as sk_f1_score


class AvgF1():
    """ Class-average F1
    """

    def __init__(self):
        super(AvgF1, self).__init__()

    def __call__(self, label, preds):
        return sk_f1_score(label, preds, average="macro")


class GP():
    """ Gender-parity score.
        https://github.com/decompositional-semantics-initiative/DNC.
    """

    def __init__(self):
        super(GP, self).__init__()

    def __call__(self, data, preds):
        """
        Calculate gender parity. 
        data: [{hypothesis, context, pair_id, preds}]
        preds:unused
        """
        same_preds, dif_preds = 0, 0
        for idx in range(int(len(data) / 2)):
            pred1 = data[idx * 2]
            pred2 = data[(idx * 2) + 1]
            assert (
                pred1["hypothesis"] == pred2["hypothesis"]
            ), "Mismatched hypotheses for ids %s and %s" % (str(pred1["idx"]), str(pred2["idx"]))
            if pred1["preds"] == pred2["preds"]:
                same_preds += 1
            else:
                dif_preds += 1

        if same_preds + dif_preds == 0:
            return -1
        return float(same_preds) / float(same_preds + dif_preds)
