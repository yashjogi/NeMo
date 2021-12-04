# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import time
from typing import List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only

from pytorch_lightning import Trainer, LightningModule
# from sacrebleu import corpus_bleu


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log
    """

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)


class ResetOptimizerStepsCallback(Callback):
    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: torch.optim.Optimizer, opt_idx: int
    ) -> None:
        if (
            pl_module.optimizer_reset_period is not None
            and pl_module.global_step > 0
            and pl_module.global_step % pl_module.optimizer_reset_period == 0
        ):
            print("(ResetOptimizerStepsCallback.on_before_optimizer_step)optimizer:", optimizer)
            if isinstance(pl_module.optimizer_reset_period, list):
                optimizer.load_state_dict(pl_module.optimizer_reset_state_dict[opt_idx])
            else:
                optimizer.load_state_dict(pl_module.optimizer_reset_state_dict)


def instantiate_callbacks(callback_configs: List[DictConfig]) -> List[Callback]:
    callbacks = []
    for conf in callback_configs:
        callbacks.append(instantiate(conf))
    return callbacks
