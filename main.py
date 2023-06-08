import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import HfArgumentParser, set_seed, GPTBigCodeForCausalLM, GPT2TokenizerFast
import json
import torch

from generators import GeneratorBase, StarCoder
from args import ModelArguments, GenerationArguments, MiscArguments, ServeArguments
from get_model import get_model, Models
from util import logger

app = FastAPI()
app.add_middleware(
    CORSMiddleware
)
generator: GeneratorBase = ...


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}')
    generated_text: str = generator.generate(inputs, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    return {
        "generated_text": generated_text,
        "status": 200
    }


def main():
    global generator
    hfparser = HfArgumentParser((ModelArguments, GenerationArguments, MiscArguments, ServeArguments))
    model_args, generation_args, misc_args, serve_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    if extra_args:
        raise f"Received unsupported command-line args: {extra_args}"

    set_seed(misc_args.seed)

    # deliberately avoiding device_map='auto', because I've seen it unnecessarily put layers on-CPU, breaking the model.
    # you may want to use device_map='auto' if you have multiple GPUs available though.
    device_map = 'mps' if torch.backends.mps.is_available() else 0

    models: Models[GPTBigCodeForCausalLM, GPT2TokenizerFast] = get_model(
        model_args=model_args, misc_args=misc_args, device_map=device_map
    )

    generator = StarCoder(model=models.model, tokenizer=models.tokenizer, gen_args=generation_args)
    uvicorn.run(app, host=serve_args.host, port=serve_args.port)


if __name__ == '__main__':
    main()
