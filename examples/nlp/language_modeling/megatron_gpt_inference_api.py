import pprint
import ast
import nlp

import os
import flask
from flask import request
import requests as req
from flask_cors import CORS
import nltk
import re
import csv
from transformers import AdamW, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, \
    GPT2Model, GPT2Config, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.utils.checkpoint as checkpoint
import torch
import numpy as np
import json
# from time import time
# from random import choice, randint
# import math
# import functools

app = flask.Flask(__name__)
CORS(app)

from nemo.collections.nlp.data.language_modeling.megatron.gpt_request_dataset import GPTRequestDataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPCheckpointConnector,
    NLPDDPPlugin,
    NLPNativeBfloat16PrecisionPlugin,
    NLPNativeMixedPrecisionPlugin,
    NLPPrecisionPlugin,
    NLPSaveRestoreConnector,
)
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from nemo.utils import logging
from nemo.utils.app_state import AppState

import pandas as pd
import json
import requests


assert torch.cuda.is_available()

# Answer Extender api-endpoint
host = '0.0.0.0'
port = 9000

headers = {'Content-type': 'application/json'}
GENQA_OBJ = None


class MegaGPT:
    def __init__(self):
        self.megagpt_model = None
        self.trainer = None

    def initialize(self):

        torch.set_grad_enabled(False)

        precision = 'bf16'  # TODO: take precision from config

        # trainer required for restoring model parallel models

        if precision == 16:
            self.trainer = Trainer(
                plugins=[NLPDDPPlugin(), NLPNativeMixedPrecisionPlugin()], gpus=1
            )
        elif precision == 'bf16':
            self.trainer = Trainer(
                plugins=[NLPDDPPlugin(), NLPNativeBfloat16PrecisionPlugin(), ],
                gpus=1,
            )
        else:
            self.trainer = Trainer(plugins=[NLPDDPPlugin(), NLPPrecisionPlugin()],
                                   gpus=1)

        # self.trainer = Trainer(plugins=NLPDDPPlugin(), gpus=1)

        # TODO from config / args
        tensor_model_parallel_size = 1

        app_state = AppState()
        app_state.model_parallel_size = tensor_model_parallel_size
        app_state.model_parallel_rank = compute_model_parallel_rank(self.trainer.local_rank,
                                                                    app_state.model_parallel_size)

        # model_file = "/raid/purnendu/models/megatron_gpt_edited.nemo"
        # model_file = "/raid/purnendu/models/megatron_gpt_rank2.nemo" #115k steps
        # model_file = "/raid/purnendu/models/1pt3B-300k/1.3B_bf16/gpt_1.3B_bf16.nemo"  # 300k steps
        # model_file = "/raid/purnendu/models/2-1pt3B-300k/1.3B_bf16/edited_nemo/finetuned-squad-7497samples/gpt1pt3B-ft-squad7497samples-rank1.nemo"  # gpt1pt3B-300k-rank1.nemo"  # 300k steps edited
        model_file = "/raid/purnendu/models/3-squadpartial-scripted/rank0/gen_squad_ft_interm2_rank0.nemo"  # 300k steps edited

        self.megagpt_model = MegatronGPTModel.restore_from(restore_path=model_file, trainer=self.trainer)

        # Do a dummy warm up run
        gpt_request = {
            "prompt": "Hello, tell me about yourself A: ",
            "tokens_to_generate": 5,
            "stop_after_sentence": True,
        }

        dataset = GPTRequestDataset(gpt_request, self.megagpt_model.tokenizer)
        request_dl = DataLoader(dataset)
        response = self.trainer.predict(self.megagpt_model, request_dl)

        print("*************************** dummy warm up run: ")
        print(response[0]['completion']['text'])
        logging.info(f"Generation stopped because: {response[0]['completion']['stop reason']}")
        print("*************************** dummy warm up run completed")

    def runPredict(self, request):
        dataset = GPTRequestDataset(request, self.megagpt_model.tokenizer)
        request_dl = DataLoader(dataset)
        response = self.trainer.predict(self.megagpt_model, request_dl)
        # response = self.megagpt_model.complete(request_dl)
        return response


@app.route("/megagpt", methods=["PUT"])
def getGPTResponse():
    data = request.get_json()
    prompt = data["prompt"]
    print("the received data is: ")

    print(data["prompt"])

    tokens_to_generate = data["tokens_to_generate"]
    stop_after_sentence = data["stop_after_sentence"]

    gpt_request = {
        "prompt": prompt,
        "tokens_to_generate": tokens_to_generate,
        "stop_after_sentence": stop_after_sentence,
    }

    response = MegaGPT_OBJ.runPredict(gpt_request)

    print("***************************")
    print(response[0]['completion']['text'])
    print("***************************")
    logging.info(f"Generation stopped because: {response[0]['completion']['stop reason']}")
    print("*************************** Full response output: ")
    print(response)

    # return json.dumps(
    #     {"result": response[0]['completion']['text'], "stop_reason": response[0]['completion']['stop reason']})

    completion = response[0]['completion']
    df_logprobs = pd.DataFrame(pd.DataFrame(completion).tokens.tolist(), columns=["token", "id", "logprob"])
    tokens_list = df_logprobs.token.tolist()
    logprobs = df_logprobs.logprob.tolist()

    return json.dumps(
        {"text": response[0]['completion']['text'],
         "logprobs": logprobs,
         "segments": tokens_list,
         "stop_reason": response[0]['completion']['stop reason']})
    # return json.dumps(response)


if __name__ == "__main__":
    MegaGPT_OBJ = MegaGPT()
    MegaGPT_OBJ.initialize()
    app.run(host=host, port=port, debug=True,
            threaded=True, use_reloader=False)
