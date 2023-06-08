# Hugging Face VSCode Endpoint Server

starcoder server for [huggingface-vscdoe](https://github.com/huggingface/huggingface-vscode) custom endpoint.

**Can't handle distributed inference very well yet.**

## Usage

See [this cool gist](https://gist.github.com/Birch-san/37c1309b547888c472b62e9a7de2ecde) for more details on how to use this repository to run a [`bigcode/starcoder`](https://huggingface.co/bigcode/starcoder) code completion server, with NF4 4-bit quantization (fits into ~11GB VRAM).

```bash
pip install -r requirements.txt
python -m main --model_name_or_path bigcode/starcoder --trust_remote_code --bf16
```

Fill `http://localhost:8000/api/generate/` into `Hugging Face Code > Model ID or Endpoint` in VSCode.

## API

```shell
curl -X POST http://localhost:8000/api/generate/ -d '{"inputs": "", "parameters": {"max_new_tokens": 64}}'
# response = {"generated_text": ""}
```

## Acknowledgements

Includes MIT-licensed code copied from Artidoro Pagnoni's [qlora](https://github.com/artidoro/qlora), and [Apache-licensed](licenses/MosaicML-mpt-7b-chat-hf-space.Apache.LICENSE.txt) code copied from MosaicML's [mpt-7b-chat](https://huggingface.co/spaces/mosaicml/mpt-7b-chat/blob/main/app.py) Huggingface Space.