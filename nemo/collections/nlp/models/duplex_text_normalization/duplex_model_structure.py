import torch
from transformers import (
    T5Config,
    T5Tokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    BaseModelOutput,
)


import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization import TextNormalizationDecoderDataset

import functools
import operator
import os, psutil

from time import perf_counter
from typing import List, Optional

from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    ExecutionMode,
)

__all__ = ['get_onnx_runtime_sessions', 'OnnxT5']


class T5Encoder(torch.nn.Module):
    def __init__(self, encoder_sess):
        super().__init__()
        self.encoder = encoder_sess

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        encoder_hidden_state = torch.from_numpy(
            self.encoder.run(
                None,
                {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": attention_mask.cpu().numpy(),
                },
            )[0]
        )

        return BaseModelOutput(encoder_hidden_state)


class T5DecoderInit(torch.nn.Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):

        decoder_outputs = self.decoder.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy(),
                "encoder_attention_mask": encoder_attention_mask.cpu().numpy(),
                "encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
            },
        )

        list_pkv = tuple(torch.from_numpy(x) for x in decoder_outputs[1:])

        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return torch.from_numpy(decoder_outputs[0]), out_past_key_values


class T5Decoder(torch.nn.Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_output, past_key_values):

        decoder_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "encoder_attention_mask": attention_mask.cpu().numpy(),
            "encoder_hidden_states": encoder_output.cpu().numpy(),
        }

        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

        past_key_values = {
            f"pkv_{i}": pkv.cpu().numpy() for i, pkv in enumerate(flat_past_key_values)
        }

        decoder_outputs = self.decoder.run(None, {**decoder_inputs, **past_key_values})
        # converts each value of the list to tensor from numpy
        list_pkv = tuple(torch.from_numpy(x) for x in decoder_outputs[1:])

        # creates a tuple of tuples of shape 6x4 from the above tuple
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return torch.from_numpy(decoder_outputs[0]), out_past_key_values



class OnnxT5(T5ForConditionalGeneration):
    """ creates a T5 model using onnx sessions (encode, decoder & init_decoder) """

    def __init__(self, model_or_model_path, onnx_model_sessions):
        config = T5Config.from_pretrained(model_or_model_path)
        super().__init__(config)

        assert len(onnx_model_sessions) == 3, "all three models should be given"

        encoder_sess, decoder_sess, decoder_sess_init = onnx_model_sessions

        self.encoder = T5Encoder(encoder_sess)
        self.decoder = T5Decoder(decoder_sess)
        self.decoder_init = T5DecoderInit(decoder_sess_init)

    def _infer(
        self,
        sents: List[List[str]],
        nb_spans: List[int],
        span_starts: List[List[int]],
        span_ends: List[List[int]],
        inst_directions: List[str],
    ):
        """ Main function for Inference
        Args:
            sents: A list of inputs tokenized by a basic tokenizer.
            nb_spans: A list of ints where each int indicates the number of semiotic spans in each input.
            span_starts: A list of lists where each list contains the starting locations of semiotic spans in an input.
            span_ends: A list of lists where each list contains the ending locations of semiotic spans in an input.
            inst_directions: A list of str where each str indicates the direction of the corresponding instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).

        Returns: A list of lists where each list contains the decoded spans for the corresponding input.
        """

        if sum(nb_spans) == 0:
            return [[]] * len(sents)
        model, tokenizer = self.model, self._tokenizer
        try:
            model_max_len = model.config.n_positions
        except AttributeError:
            model_max_len = 512
        ctx_size = constants.DECODE_CTX_SIZE
        extra_id_0 = constants.EXTRA_ID_0
        extra_id_1 = constants.EXTRA_ID_1

        # Build all_inputs
        input_centers, input_dirs, all_inputs = [], [], []
        for ix, sent in enumerate(sents):
            cur_inputs = []
            for jx in range(nb_spans[ix]):
                cur_start = span_starts[ix][jx]
                cur_end = span_ends[ix][jx]
                ctx_left = sent[max(0, cur_start - ctx_size) : cur_start]
                ctx_right = sent[cur_end + 1 : cur_end + 1 + ctx_size]
                span_words = sent[cur_start : cur_end + 1]
                span_words_str = ' '.join(span_words)
                if is_url(span_words_str):
                    span_words_str = span_words_str.lower()
                input_centers.append(span_words_str)
                input_dirs.append(inst_directions[ix])
                # Build cur_inputs
                if inst_directions[ix] == constants.INST_BACKWARD:
                    cur_inputs = [constants.ITN_PREFIX]
                if inst_directions[ix] == constants.INST_FORWARD:
                    cur_inputs = [constants.TN_PREFIX]
                cur_inputs += ctx_left
                cur_inputs += [extra_id_0] + span_words_str.split(' ') + [extra_id_1]
                cur_inputs += ctx_right
                all_inputs.append(' '.join(cur_inputs))

        # Apply the decoding model
        batch = tokenizer(all_inputs, padding=True, return_tensors='pt')
        input_ids = batch['input_ids'].to(self.device)
        generated_ids = model.generate(input_ids, max_length=model_max_len)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Post processing
        generated_texts = self.postprocess_output_spans(input_centers, generated_texts, input_dirs)

        # Prepare final_texts
        final_texts, span_ctx = [], 0
        for nb_span in nb_spans:
            cur_texts = []
            for i in range(nb_span):
                cur_texts.append(generated_texts[span_ctx])
                span_ctx += 1
            final_texts.append(cur_texts)

        return final_texts

    def postprocess_output_spans(self, input_centers, output_spans, input_dirs):
        en_greek_spokens = list(constants.EN_GREEK_TO_SPOKEN.values())
        for ix, (_input, _output) in enumerate(zip(input_centers, output_spans)):
            if self.lang == constants.ENGLISH:
                # Handle URL
                if is_url(_input):
                    output_spans[ix] = ' '.join(wordninja.split(_output))
                    continue
                # Greek letters
                if _input in en_greek_spokens:
                    if input_dirs[ix] == constants.INST_FORWARD:
                        output_spans[ix] = _input
                    if input_dirs[ix] == constants.INST_BACKWARD:
                        output_spans[ix] = constants.EN_SPOKEN_TO_GREEK[_input]
        return output_spans

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:

            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )

            logits, past_key_values = init_onnx_outputs

        else:

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values = onnx_outputs

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)


def get_onnx_runtime_sessions(
    model_paths,
    default: bool = True,
    opt_level: int = 99,
    parallel_exe_mode: bool = True,
    n_threads: int = 4,
    provider=[
        "CPUExecutionProvider",
    ],
) -> InferenceSession:
    """
            Optimizes the model

    Args:
        path_to_encoder (str) : the path of input onnx encoder model.
        path_to_decoder (str) : the path of input onnx decoder model.
        path_to_initial_decoder (str) :  the path of input initial onnx decoder model.
        opt_level (int) : sess_options.GraphOptimizationLevel param if set 1 uses 'ORT_ENABLE_BASIC',
                          2 for 'ORT_ENABLE_EXTENDED' and 99 for 'ORT_ENABLE_ALL',
                          default value is set to 99.
        parallel_exe_mode (bool) :  Sets the execution mode. Default is parallel.
        n_threads (int) :  Sets the number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose
        provider : execution providers list.
        default : set this to true, ort will choose the best settings for your hardware.
                  (you can test out different settings for better results.)

    Returns:
        encoder_session : encoder onnx InferenceSession
        decoder_session : decoder onnx InferenceSession
        decoder_sess_init : initial decoder onnx InferenceSession

    """
    path_to_encoder, path_to_decoder, path_to_initial_decoder = model_paths

    if default:
        encoder_sess = InferenceSession(str(path_to_encoder))

        decoder_sess = InferenceSession(str(path_to_decoder))

        decoder_sess_init = InferenceSession(str(path_to_initial_decoder))

    else:

        # Few properties that might have an impact on performances
        options = SessionOptions()

        if opt_level == 1:
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif opt_level == 2:
            options.graph_optimization_level = (
                GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        else:
            assert opt_level == 99
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # set this true for better performance
        if parallel_exe_mode == True:
            options.execution_mode = ExecutionMode.ORT_PARALLEL
        else:
            options.execution_mode = ExecutionMode.ORT_SEQUENTIAL

        options.intra_op_num_threads = n_threads
        # options.inter_op_num_threads = 10


        encoder_sess = InferenceSession(
            str(path_to_encoder), options, providers=provider
        )

        decoder_sess = InferenceSession(
            str(path_to_decoder), options, providers=provider
        )

        decoder_sess_init = InferenceSession(
            str(path_to_initial_decoder), options, providers=provider
        )

    return encoder_sess, decoder_sess, decoder_sess_init
