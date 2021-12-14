# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script export tagger model

"""


from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import DuplexTaggerModel


def main():
    model = DuplexTaggerModel.restore_from('/home/yzhang/code/NeMo/examples/nlp/duplex_text_normalization/checkpoints_tagger_fixsame_albert_2372207/tagger/tagger_training/2021-10-28_00-24-25/checkpoints/tagger_training.nemo')
    model.export('tagger.onnx')

if __name__ == '__main__':
    main()
