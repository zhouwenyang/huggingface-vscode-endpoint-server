from dataclasses import dataclass
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	PreTrainedTokenizerBase,
)
from typing import Union, Dict, Generic, TypeVar
import torch
import logging

from args import ModelArguments, MiscArguments

logger = logging.getLogger(__name__)

ModelT = TypeVar('ModelT', bound=AutoModelForCausalLM)
TokenizerT = TypeVar('TokenizerT', bound=PreTrainedTokenizerBase)

@dataclass
class Models(Generic[ModelT, TokenizerT]):
	model: ModelT
	tokenizer: TokenizerT


def get_model(
	model_args: ModelArguments,
	misc_args: MiscArguments,
	device_map: Union[None, int, str, Dict[str, Union[int, str]]] = None,
) -> Models[ModelT, TokenizerT]:
	cuda_avail = torch.cuda.is_available()
	compute_dtype = torch.bfloat16 if model_args.bf16 else torch.float16
	load_in_4bit = model_args.bits == 4 and cuda_avail
	load_in_8bit = model_args.bits == 8 and cuda_avail

	tokenizer: TokenizerT = AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		use_fast=True,
	)

	quantization_config = BitsAndBytesConfig(
		load_in_4bit=load_in_4bit,
		load_in_8bit=load_in_8bit,
		llm_int8_threshold=6.0,
		llm_int8_has_fp16_weight=False,
		bnb_4bit_compute_dtype=compute_dtype,
		bnb_4bit_use_double_quant=model_args.double_quant,
		bnb_4bit_quant_type=model_args.quant_type,
	) if cuda_avail else None

	if not cuda_avail:
		logger.warning("You don't have CUDA, so we have turned off quantization. If you happen to be on a Mac: you probably have enough unified memory to run in fp16 anyway…")

	if compute_dtype == torch.float16 and cuda_avail and torch.cuda.is_bf16_supported():
		print("Your GPU supports bfloat16; you may want to try it with --bf16 (note: I'm not sure how important this is for inference, but it's certainly preferred when training with 4-bit quantization.)")
	
	# deliberately avoiding a device_map auto, because I've seen it unnecessarily put layers on-CPU, breaking the model.
	# you may want to use device_map='auto' if you have multiple GPUs available though.
	device_map = 'mps' if torch.backends.mps.is_available() else 0

	model: ModelT = AutoModelForCausalLM.from_pretrained(
		model_args.model_name_or_path,
		load_in_4bit=load_in_4bit,
		load_in_8bit=load_in_8bit,
		device_map=device_map,
		quantization_config=quantization_config,
		torch_dtype=compute_dtype,
		trust_remote_code=model_args.trust_remote_code,
	).eval()
	model.config.torch_dtype=compute_dtype

	if misc_args.compile:
		torch.compile(model, mode='max-autotune')
	
	return Models[ModelT, TokenizerT](
		model=model,
		tokenizer=tokenizer,
	)