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

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from typing import Optional

from nemo.utils import logging


class StreamInferenceMode(Enum):
    # Model is in a training state. No streaming is done.
    TRAINING = 'TRAINING'

    # Below are three options for inference:

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is kept inside of the model
    STREAM_INTERNAL_STATE_INFERENCE = 'STREAM_INTERNAL_STATE_INFERENCE'

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is received from outside of the model
    STREAM_EXTERNAL_STATE_INFERENCE = 'STREAM_EXTERNAL_STATE_INFERENCE'

    # Model its in inference mode and it's topology is the same with training
    # mode (with removed droputs etc)
    NON_STREAM_INFERENCE = 'NON_STREAM_INFERENCE'


def _is_convolved_op(m: nn.Module):
    if isinstance(
        m,
        (
            nn.Conv1d,
            nn.Conv2d,
            # tf.keras.layers.AveragePooling2D)
        ),
    ):
        return True
    else:
        return False


def _is_pool_op(m: nn.Module):
    if isinstance(m, (nn.AvgPool1d, nn.AvgPool2d)):
        return True
    else:
        return False


def _is_global_time_dim_op(m: nn.Module):
    if isinstance(m, (nn.Flatten, nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d),):
        return True
    else:
        return False


def _get_time_dim_of_op(m: nn.Module):
    if _is_convolved_op(m):
        return -1
    elif _is_pool_op(m):
        return -1
    elif _is_global_time_dim_op(m):
        return -1
    else:
        raise ValueError(f"Op {m.__class__.__name__} is not supported")


class StreamWrapper(nn.Module):
    """
    Streaming wrapper - it is not a standalone layer.
    It can be used to wrap Keras layer for streaming inference mode.
    Advantage of streaming inference mode - it is more computationally efficient.
    But not all layers are streamable. Some layers require keeping a buffer
    with features in time. We can wrap such layer by Stream().
    Where Stream() will create and keep a temporal buffer called state,
    for both cases: internal state and external state.
    Examples of layers which require temporal buffer/state
    for streaming inference are Conv2D, DepthwiseConv2D, AveragePooling2D,
    Flatten in time dimension, etc.

    This wrapper is generic enough, so that it can be used for any modes:
    1 Streaming with internal state. This wrapper will manage internal state.
    2 Streaming with external state. Developer will have to manage external state
    and feed it as additional input to the model and then receive output with
    updated state.
    3 Non streaming inference mode. In this case wrapper will just call
    a wrapped layer as it is. There will be no difference in efficiency.
    The graph will be the same as in training mode, but some training features
    will be removed (such as dropout, etc)
    4 Training mode.

    Attributes:
      module: keras layer which has to be streamed or tf.identity
      inference_batch_size: batch size in inference mode
      mode: inference or training mode
      pad_time_dim: padding in time
      state_shape:
      ring_buffer_size_in_time_dim: size of ring buffer in time dim
      samplewise_inference: True - model will run one sample per one inference step;
        False - model will run multiple per one inference step.
        It is useful for strided streaming

    Raises:
      ValueError: if padding is not 'valid' in streaming mode;
                  or if striding is used with use_one_step;
                  or cell is not supported
    """

    def __init__(
        self,
        module: torch.nn.Module,
        inference_batch_size: int = 1,
        mode: StreamInferenceMode = StreamInferenceMode.TRAINING,
        pad_time_dim: Optional[int] = None,
        state_shape: Optional[torch.Tensor] = None,
        ring_buffer_size_in_time_dim: Optional[int] = None,
        samplewise_inference: bool = True,
    ):
        super().__init__()
        self.__dict__['__finished_init'] = False

        self.inner_module = module  # type: torch.nn.Module
        self.inference_batch_size = inference_batch_size
        self.mode = mode
        self.pad_time_dim = pad_time_dim
        self.state_shape = state_shape
        self.ring_buffer_size_in_time_dim = ring_buffer_size_in_time_dim
        self.samplewise_inference = samplewise_inference
        self.stream_stride = 1
        self.built = False

        wrapped_module = module

        if not samplewise_inference and _is_global_time_dim_op(wrapped_module):
            raise ValueError(
                'Flatten, AdaptiveAvgPool1d, AdaptiveMaxPool1d '
                'can be used only with samplewise_inference = True '
                'because they are executed one time per inference call '
                'and produce only one output in time dim, whereas conv '
                'can produce multiple outputs in time dim, '
                'so conv can be used with samplewise_inference = False or True'
            )

        self._initialize_ring_buffer_size_in_time_dim(wrapped_module)

        self.__dict__['__finished_init'] = True

    def forward(self, *inputs):
        self._build_states(*inputs)

        if self.mode == StreamInferenceMode.STREAM_INTERNAL_STATE_INFERENCE:
            return self._streaming_internal_state(*inputs)

        elif self.mode == StreamInferenceMode.STREAM_EXTERNAL_STATE_INFERENCE:
            if self.ring_buffer_size_in_time_dim:
                # in streaming inference mode with external state
                # in addition to the output we return the output state.
                output, self.output_state = self._streaming_external_state(self.input_state, *inputs)
            else:
                # if there is no ring buffer then the input_state isn't needed.
                output = self.cell(*inputs)

            return output

        elif self.mode in (StreamInferenceMode.TRAINING, StreamInferenceMode.NON_STREAM_INFERENCE):
            # run non streamable training or non streamable inference
            return self._non_streaming(*inputs)

        else:
            raise ValueError(f'Encountered unexpected mode `{self.mode}`.')

    def _initialize_ring_buffer_size_in_time_dim(self, wrapped_module):
        if self.ring_buffer_size_in_time_dim is not None:
            # it is a special case when ring_buffer_size_in_time_dim is specified
            # outside of the layer in this case we just build a ring buffer
            # and do not check what is the type of the cell
            pass

        elif _is_convolved_op(wrapped_module):
            padding = wrapped_module.padding
            strides = wrapped_module.stride
            self.stream_stride = strides[0]

            if self.mode not in (StreamInferenceMode.TRAINING, StreamInferenceMode.NON_STREAM_INFERENCE):
                # if padding != 'valid':
                #     raise ValueError('conv/cell padding has to be valid,' 'padding has to be set by pad_time_dim')
                if padding != 0:
                    raise ValueError('conv/cell padding has to be 0,' 'padding has to be set by pad_time_dim')

                if self.samplewise_inference:
                    if strides[0] > 1:
                        raise ValueError(
                            'Stride in time dim greater than 1 '
                            'in streaming mode with samplewise_inference=True'
                            ' is not supported, set samplewise_inference=False'
                        )

            dilation_rate = wrapped_module.dilation
            kernel_size = wrapped_module.kernel_size

            if self.samplewise_inference:
                # effective kernel size in time dimension
                self.ring_buffer_size_in_time_dim = dilation_rate[0] * (kernel_size[0] - 1) + 1
            else:
                # Streaming of strided or 1 step conv.
                # Assuming input length is a multiple of strides (otherwise streaming
                # conv is not meaningful), setting to this value (instead of
                # dilation_rate[0] * (kernel_size[0] - 1)) ensures that we do not
                # ignore the `strides - 1` rightmost (and hence most recent) valid
                # input samples.
                self.ring_buffer_size_in_time_dim = max(0, dilation_rate[0] * (kernel_size[0] - 1) - (strides[0] - 1))

        elif _is_pool_op(wrapped_module):
            strides = wrapped_module.stride
            kernel_size = wrapped_module.kernel_size
            self.stream_stride = strides[0]

            if (
                self.mode not in (StreamInferenceMode.TRAINING, StreamInferenceMode.NON_STREAM_INFERENCE)
                and strides[0] != kernel_size[0]
            ):
                raise ValueError('Stride in time %d must = pool size in time %d' % (strides[0], kernel_size[0]))
            # effective kernel size in time dimension
            self.ring_buffer_size_in_time_dim = kernel_size[0]

        elif _is_global_time_dim_op(wrapped_module):
            # effective kernel size in time dimension
            if self.state_shape:
                self.ring_buffer_size_in_time_dim = self.state_shape[_get_time_dim_of_op(wrapped_module)]

        else:
            raise ValueError('Cell is not supported ', wrapped_module)

        if self.ring_buffer_size_in_time_dim == 1:
            logging.warning('There is no need to use Stream on time dim with size 1')

    def _build_states(self, *inputs):
        if self.built:
            return

        prime_input = inputs[0]
        input_shape = prime_input.shape
        wrapped_module = self.inner_module

        if _is_convolved_op(wrapped_module):
            self.state_shape = [self.inference_batch_size] + list(input_shape[1:-1]) + [self.ring_buffer_size_in_time_dim]

        elif _is_global_time_dim_op(wrapped_module) and not self.state_shape:
            if self.mode in (StreamInferenceMode.TRAINING, StreamInferenceMode.Modes.NON_STREAM_INFERENCE):
                # Only in the non-streaming modes we have access to the whole training
                # sequence. In the streaming mode input_shape will not be available.
                # During streaming inference we have access to one sample at a time!
                # So we generate state shape based on input_shape during training.
                # It will be stored in the layer config
                # Then used by clone_streaming_model to create state buffer,
                # during layer initialization.
                # [batch,feature, time]
                self.state_shape = list(input_shape)
                self.state_shape[0] = self.inference_batch_size

        elif self.ring_buffer_size_in_time_dim:
            # it is a special case when ring_buffer_size_in_time_dim
            # is defined by user and cell is not defined in Stream wrapper
            self.state_shape = [self.inference_batch_size] + list(input_shape[1:-1]) + [self.ring_buffer_size_in_time_dim]

        # Build the state
        if self.mode == StreamInferenceMode.STREAM_INTERNAL_STATE_INFERENCE:
            # Create a state varaible for streaming inference mode (internal state).
            # Where states become a weight in the layer
            if self.ring_buffer_size_in_time_dim:
                self.states = torch.zeros(*self.state_shape, requires_grad=False)
                # IF INTERNAL STATE, ATTACH TO MODULE
                # self.states = nn.Parameter(self.states, requires_grad=False)

        elif self.mode == StreamInferenceMode.STREAM_EXTERNAL_STATE_INFERENCE:
            # For streaming inference with extrnal states,
            # the states are passed in as input.
            if self.ring_buffer_size_in_time_dim:
                self.input_state = torch.zeros(*self.state_shape[1:])
                # tf.keras.layers.Input(
                #     shape=self.state_shape[1:],
                #     batch_size=self.inference_batch_size,
                #     name=self.name + '/' + self.state_name_tag,
                # )  # adding names to make it unique
            else:
                self.input_state = None
            self.output_state = None

        self.built = True

    def get_input_state(self):
        # input state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
        if self.mode == StreamInferenceMode.STREAM_EXTERNAL_STATE_INFERENCE:
            return [self.input_state]
        else:
            raise ValueError('Expected the layer to be in external streaming mode, ' f'not `{self.mode}`.')

    def get_output_state(self):
        # output state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
        if self.mode == StreamInferenceMode.STREAM_EXTERNAL_STATE_INFERENCE:
            return [self.output_state]
        else:
            raise ValueError('Expected the layer to be in external streaming mode, ' f'not `{self.mode}`.')

    def _non_streaming(self, *inputs):
        # Pad inputs in time dim: causal or same
        if self.pad_time_dim:
            if _is_global_time_dim_op(self.inner_module):
                raise ValueError(f'pad_time_dim can not be used with {self.inner_module.__class__.__name__}')

        # temporal padding
        input = inputs[0]
        pad = [[0, 0]] * len(input.shape)
        if self.samplewise_inference:
            pad_total_amount = self.ring_buffer_size_in_time_dim - 1
        else:
            pad_total_amount = self.ring_buffer_size_in_time_dim

        time_axis = _get_time_dim_of_op(self.inner_module)
        if self.pad_time_dim == 'causal':
            pad[time_axis] = [pad_total_amount, 0]

        elif self.pad_time_dim == 'same':
            half = pad_total_amount // 2
            pad[time_axis] = [half, pad_total_amount - half]

        if sum(pad[time_axis]) > 0:
            input = F.pad(input, pad, 'constant')

        inputs = (input, *inputs[1:])

        return self.inner_module(*inputs)

    def _streaming_internal_state(self, *inputs):
        if self.samplewise_inference:
            input = inputs[0]
            time_axis = _get_time_dim_of_op(self.inner_module)

            # The time dimenstion always has to equal 1 in streaming mode.
            if input.shape[time_axis] != 1:
                raise ValueError(f'inputs[0].shape[{time_axis}]: %d must be 1 ' % input.shape[time_axis])

            # remove latest row [batch_size, (optional feature_dim), channel, (memory_size-1)]
            memory = self.states[
                ..., 1 : self.ring_buffer_size_in_time_dim,
            ]

            # add new row [batch_size, feature_dim, channel, memory_size]
            memory = torch.cat([memory, input], dim=time_axis)

            # assign_states = self.states.assign(memory)
            self.states = self.states * 0 + memory

            inputs = (memory, *inputs[1:])
            return self.inner_module(*inputs)
        else:
            # add new row [batch_size, memory_size, feature_dim, channel]
            if self.ring_buffer_size_in_time_dim:
                input = inputs[0]
                time_axis = _get_time_dim_of_op(self.inner_module)

                memory = torch.cat([self.states, input], time_axis)

                state_update = memory[..., -self.ring_buffer_size_in_time_dim :]

                # assign_states = self.states.assign(state_update)
                self.states = self.states * 0 + state_update

                inputs = (state_update, *inputs[1:])
                return self.inner_module(*inputs)
            else:
                return self.inner_module(*inputs)

    def _streaming_external_state(self, state, *inputs):
        state = [] if state is None else state
        input = inputs[0]
        time_axis = _get_time_dim_of_op(self.inner_module)

        if self.samplewise_inference:
            # The time dimenstion always has to equal 1 in streaming mode.
            if input.shape[1] != 1:
                raise ValueError(f'inputs.shape[{time_axis}]: %d must be 1 ' % input.shape[time_axis])

            # remove latest row [batch_size, feature_dim, channel, (memory_size-1)]
            memory = state[..., 1 : self.ring_buffer_size_in_time_dim]

            # add new row [batch_size, memory_size, feature_dim, channel]
            memory = torch.cat([memory, input], dim=time_axis)

            inputs = (memory, *inputs[1:])

            output = self.inner_module(*inputs)
            return output, memory
        else:
            # add new row [batch_size, memory_size, feature_dim, channel]
            memory = torch.cat([state, input], dim=time_axis)

            state_update = memory[..., -self.ring_buffer_size_in_time_dim :]

            inputs = (memory, *inputs[1:])
            output = self.inner_module(*inputs)
            return output, state_update

    def __getattr__(self, attr):
        if '__finished_init' in self.__dict__ and self.__dict__['__finished_init']:
            if attr in self.__dict__:
                return nn.Module.__getattr__(self, attr)

            name = attr
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    return _parameters[name]
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name]
            if '_modules' in self.__dict__:
                modules = self.__dict__['_modules']
                if name in modules:
                    return modules[name]

            return getattr(self.inner_module, attr)
        else:
            nn.Module.__getattr__(self, attr)

    def __setattr__(self, key, value):
        if '__finished_init' in self.__dict__ and self.__dict__['__finished_init']:
            if key in self.__dict__:
                nn.Module.__setattr__(self, key, value)
            else:
                setattr(self.inner_module, key, value)
        else:
            nn.Module.__setattr__(self, key, value)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        prefix = prefix.replace('inner_module.', '')
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        # prefix = prefix + 'inner_module'

        persistent_buffers = {k: v for k, v in self.inner_module.buffers() if k not in self.inner_module._non_persistent_buffers_set}
        local_name_params = itertools.chain(self.inner_module._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + 'inner_module.' + name
            print("key", key)
            if key in state_dict:
                pass  # skip

            old_key = prefix + name
            if old_key in state_dict and key not in state_dict:
                temp = state_dict[old_key]
                del state_dict[old_key]
                state_dict[key] = temp
                print(f"replaced {old_key} with {key} in statedict")

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # def state_dict(self, destination, prefix: str, keep_vars: bool):
    #     return self.inner_module.state_dict(destination, prefix, keep_vars)
