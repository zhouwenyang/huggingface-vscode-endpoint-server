from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    GenerationConfig,
)
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
import torch
from torch import LongTensor
import logging
from typing import List, TypedDict

from args import GenerationArguments
from callback_text_iterator_streamer import CallbackTextIteratorStreamer

logger = logging.getLogger(__name__)

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        return self.generate(query, parameters)


class StarCoder(GeneratorBase):
    generation_config: GenerationConfig
    model: GPTBigCodeForCausalLM
    tokenizer: GPT2TokenizerFast
    def __init__(
        self,
        model: GPTBigCodeForCausalLM,
        tokenizer: GPT2TokenizerFast,
	    gen_args: GenerationArguments,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = GenerationConfig(**vars(gen_args))
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        generation_config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })

        tokenized_prompts: TokenizerOutput = self.tokenizer([query], return_tensors='pt')
        tokenized_prompts: TokenizerOutput = tokenized_prompts.to(self.model.device)

        def on_text(message: str, stream_end = False):
            print(message, end='', flush=True)

        streamer = CallbackTextIteratorStreamer(self.tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=True)

        prediction: LongTensor = self.model.generate(
            **tokenized_prompts,
            generation_config=generation_config,
            do_sample=generation_config.temperature > 0.,
            streamer=streamer,
        )
        decodes: List[str] = self.tokenizer.batch_decode(prediction)
        first_decode, *_ = decodes
        return first_decode


class SantaCoder(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.pretrained: str = pretrained
        self.device: str = device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.model.to(device=self.device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.generation_config: GenerationConfig = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        output_ids: torch.Tensor = self.model.generate(input_ids, generation_config=config)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text


class ReplitCode(GeneratorBase):
    def __init__(self, pretrained: str, device: str = 'cuda'):
        self.pretrained: str = pretrained
        self.device: str = device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.default_parameter: dict = dict(
            do_sample=True, top_p=0.95, top_k=4, pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.2, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, query: str, parameters: dict = None) -> str:
        input_ids: torch.Tensor = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        params = {**self.default_parameter, **(parameters or {})}
        params.pop('stop')
        output_ids: torch.Tensor = self.model.generate(input_ids, **params)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text
