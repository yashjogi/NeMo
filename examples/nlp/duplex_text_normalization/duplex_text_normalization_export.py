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
This script contains an example on how to evaluate a DuplexTextNormalizationModel.
Note that DuplexTextNormalizationModel is essentially a wrapper class around
DuplexTaggerModel and DuplexDecoderModel. Therefore, two trained NeMo models
should be specificied before evaluation (one is a trained DuplexTaggerModel
and the other is a trained DuplexDecoderModel).

USAGE Example:
1. Obtain a processed test data file (refer to the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`)
2.
# python duplex_text_normalization_test.py
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER
        data.test_ds.data_path=PATH_TO_TEST_FILE
        mode={tn,itn,joint}

The script also supports the `interactive` mode where a user can just make the model
run on any input text:
# python duplex_text_normalization_test.py
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER
        mode={tn,itn,joint}
        inference.interactive=true

This script uses the `/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'.

Note that when evaluating a DuplexTextNormalizationModel on a labeled dataset,
the script will automatically generate a file for logging the errors made
by the model. The location of this file is determined by the argument
`inference.errors_log_fp`.

"""


import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization import TextNormalizationTestDataset
from nemo.collections.nlp.models import DuplexTaggerModel, DuplexTextNormalizationModel, DuplexDecoderModel, ONNXDuplexTextNormalizationModel
from argparse import ArgumentParser



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--onnx_tagger', type=str, default='onnx/tagger.onnx', required=False, help="Path to onnx encoder model")
    parser.add_argument('--lang', type=str, default='en', required=False, help="Path to onnx encoder model")
    parser.add_argument('--mode', type=str, default='tn', required=False, choices=['itn', 'tn', 'joint'], help="Path to onnx encoder model")
    parser.add_argument('--batch_size', type=int, default=24, required=False, help="Path to onnx encoder model")
    parser.add_argument('--data_path', type=str, default='/mnt/data/text_norm/en_with_types_preprocessed/test.tsv', required=False, help="Path to onnx encoder model")
    parser.add_argument('--onnx_encoder', type=str, default='onnx/t5-base-encoder.onnx', required=False, help="Path to onnx encoder model")
    parser.add_argument('--onnx_decoder', type=str, default='onnx/t5-base-decoder.onnx', required=False, help="Path to onnx decoder model")
    parser.add_argument('--onnx_decoder_init', type=str, default='onnx/t5-base-init-decoder.onnx', required=False, help="Path to onnx decoder model")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    tagger = args.onnx_tagger
    encoder = args.onnx_encoder
    decoder = args.onnx_decoder
    decoder_init = args.onnx_decoder_init
    model = ONNXDuplexTextNormalizationModel(tagger, encoder, decoder,  decoder_init, lang=args.lang)
    test_dataset = TextNormalizationTestDataset(args.data_path, args.mode, args.lang)
    results = model.evaluate(test_dataset, args.batch_size, 'error.log')

    
    


if __name__ == '__main__':
    main()
