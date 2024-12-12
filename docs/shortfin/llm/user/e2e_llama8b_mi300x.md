# LLama 8b GPU instructions on MI300X

## Setup

We will use an example with `llama_8b_f16` in order to describe the
process of exporting a model for use in the shortfin llm server with an
MI300 GPU.

<!-- TODO: generalize, only one line here is MI300 specific (`--iree-hip-target=gfx942`) -->

### Pre-Requisites

- Python >= 3.11 is recommended for this flow
    - You can check out [pyenv](https://github.com/pyenv/pyenv)
    as a good tool to be able to manage multiple versions of python
    on the same system.

### Create virtual environment

To start, create a new virtual environment:

```bash
python -m venv --prompt shark-ai .venv
source .venv/bin/activate
```

### Install `shark-ai`

You can install either the `latest stable` version of `shark-ai`
or the `nightly` version:

#### Stable

```bash
pip install shark-ai
```

#### Nightly

```bash
pip install sharktank -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels
pip install shortfin -f https://github.com/nod-ai/shark-ai/releases/expanded_assets/dev-wheels
```

#### Install dataclasses-json

<!-- TODO: This should be included in release: -->

```bash
pip install dataclasses-json
```

### Define a directory for export files

Create a new directory for us to export files like
`model.mlir`, `model.vmfb`, etc.

<!-- TODO: don't write into the source directory unless .gitignore'd -->
<!-- `mkdir -p ~/shark-ai/llama` maybe, or have tools use a default -->

```bash
mkdir $PWD/export
export EXPORT_DIR=$PWD/export
```

### Download llama3_8b_fp16.gguf

<!-- TODO: remove usage of 'utils' -->
<!--   A: use upstream huggingface-cli or huggingface_hub library -->
<!--   B: add build tool to replace export_paged_llm_v1 that downloads, exports, and compiles
          https://github.com/nod-ai/shark-ai/issues/402 -->

<!-- TODO: use standard repo/model names like `SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF` or `meta-llama/Llama-3.1-8B-Instruct`

    TensorRT:
    * https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
    * https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html

    VLLM:
    * https://docs.vllm.ai/en/latest/getting_started/quickstart.html

    MLC LLM:
    * https://github.com/bentoml/BentoMLCLLM?tab=readme-ov-file#set-up-the-environment
    * https://llm.mlc.ai/docs/compilation/compile_models.html#clone-from-hf-and-convert-weight
    * https://llm.mlc.ai/docs/get_started/introduction.html#chat-cli
-->


We will use the `hf_datasets` module in `sharktank` to download a
LLama3.1 8b f16 model.

```bash
python -m sharktank.utils.hf_datasets llama3_8B_fp16 --local-dir $EXPORT_DIR
```

### Define environment variables

Define the following environment variables to make running
this example a bit easier:

#### Model/Tokenizer vars

This example uses the `llama8b_f16.gguf` and `tokenizer.json` files
that were downloaded in the previous step.

```bash
export MODEL_PARAMS_PATH=$EXPORT_DIR/llama3.1-8b/llama8b_f16.gguf
export TOKENIZER_PATH=$EXPORT_DIR/llama3.1-8b/tokenizer.json
```

#### General env vars

<!-- TODO: give these defaults / script arguments, no need to set like this -->

The following env vars can be copy + pasted directly:

```bash
# Path to export model.mlir file
export MLIR_PATH=$EXPORT_DIR/model.mlir
# Path to export config.json file
export OUTPUT_CONFIG_PATH=$EXPORT_DIR/config.json
# Path to export edited_config.json file
export EDITED_CONFIG_PATH=$EXPORT_DIR/edited_config.json
# Path to export model.vmfb file
export VMFB_PATH=$EXPORT_DIR/model.vmfb
# Batch size for kvcache
export BS=1,4
# NOTE: This is temporary, until multi-device is fixed
export ROCR_VISIBLE_DEVICES=1
```

## Export to MLIR

We will now use the `sharktank.examples.export_paged_llm_v1` script
to export our model to `.mlir` format.

```bash
python -m sharktank.examples.export_paged_llm_v1 \
  --irpa-file=$MODEL_PARAMS_PATH \
  --output-mlir=$MLIR_PATH \
  --output-config=$OUTPUT_CONFIG_PATH \
  --bs=$BS
```

## Compiling to `.vmfb`

<!-- TODO: bundle this as part of a tool, with precompiled vs compile modes, cache support -->

Now that we have generated a `model.mlir` file,
we can compile it to `.vmfb` format, which is required for running
the `shortfin` LLM server.

We will use the
[iree-compile](https://iree.dev/developers/general/developer-overview/#iree-compile)
tool for compiling our model.

### Compile for MI300

<!-- TODO: detect available devices and default to the current system (including multi-device sharding) -->

**NOTE: This command is specific to MI300 GPUs.
For other `--iree-hip-target` GPU options,
look [here](https://iree.dev/guides/deployment-configurations/gpu-rocm/#compile-a-program)**

```bash
iree-compile $MLIR_PATH \
 --iree-hal-target-backends=rocm \
 --iree-hip-target=gfx942 \
 -o $VMFB_PATH
```

## Write an edited config

We need to write a config for our model with a slightly edited structure
to run with shortfin. This will work for the example in our docs.
You may need to modify some of the parameters for a specific model.

### Write edited config

```bash
cat > $EDITED_CONFIG_PATH << EOF
{
    "module_name": "module",
    "module_abi_version": 1,
    "max_seq_len": 131072,
    "attn_head_count": 8,
    "attn_head_dim": 128,
    "prefill_batch_sizes": [
        $BS
    ],
    "decode_batch_sizes": [
        $BS
    ],
    "transformer_block_count": 32,
    "paged_kv_cache": {
        "block_seq_stride": 16,
        "device_block_count": 256
    }
}
EOF
```

## Running the `shortfin` LLM server

We should now have all of the files that we need to run the shortfin LLM server.

Verify that you have the following in your specified directory ($EXPORT_DIR):

<!-- TODO: have required arguments on the server / check for files on startup, keep the docs streamlined  -->

```bash
ls $EXPORT_DIR
```

- edited_config.json
- model.vmfb

### Launch server:

<!-- TODO: OpenAI API compatible server, not this custom one?

    https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
-->

<!-- #### Set the target device

TODO: Add instructions on targeting different devices,
when `--device=hip://$DEVICE` is supported -->

#### Run the shortfin server

Now that we are finished with setup, we can start the Shortfin LLM Server.

Run the following command to launch the Shortfin LLM Server in the background:

<!-- TODO: move this troubleshooting into the error message, get this off of the golden path in the docs -->

> **Note**
> By default, our server will start at `http://localhost:8000`.
> You can specify the `--host` and/or `--port` arguments, to run at a different address.
>
> If you receive an error similar to the following:
>
> `[errno 98] address already in use`
>
> Then, you can confirm the port is in use with `ss -ntl | grep 8000`
> and either kill the process running at that port,
> or start the shortfin server at a different port.

```bash
python -m shortfin_apps.llm.server \
   --tokenizer_json=$TOKENIZER_PATH \
   --model_config=$EDITED_CONFIG_PATH \
   --vmfb=$VMFB_PATH \
   --parameters=$MODEL_PARAMS_PATH \
   --device=hip > shortfin_llm_server.log 2>&1 &
shortfin_process=$!
```

<!-- TODO: have the server write its logs to a file by default if spammy? use stdout/stderr for control messages -->

You can verify your command has launched successfully
when you see the following logs outputted to terminal:

```bash
cat shortfin_llm_server.log
```

#### Expected output

```text
[2024-10-24 15:40:27.440] [info] [on.py:62] Application startup complete.
[2024-10-24 15:40:27.444] [info] [server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Verify server

We can now verify our LLM server by sending a simple request:

<!-- TODO: wrap this into a script,

    shortfin_apps.llm.client --query="" --max-completion-tokens=""

    or use 'curl'? like at
    https://pytorch.org/serve/batch_inference_with_ts.html
    https://llm.mlc.ai/docs/get_started/introduction.html#rest-server
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#example-of-usage-for-a-pair-of-a-string-and-a-list-of-texts
-->

### Open python shell

```bash
python
```

### Send request

```python
import requests

import os

port = 8000 # Change if running on a different port

generate_url = f"http://localhost:{port}/generate"

def generation_request():
    payload = {"text": "What is the capital of the United States?", "sampling_params": {"max_completion_tokens": 50}}
    try:
        resp = requests.post(generate_url, json=payload)
        resp.raise_for_status()  # Raises an HTTPError for bad responses
        print(resp.text)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

generation_request()
```

After you receive the request, you can exit the python shell:

```bash
quit()
```

## Cleanup

<!-- TODO: better cleanup https://github.com/nod-ai/shark-ai/issues/490 -->

When done, you can kill the shortfin_llm_server by killing the process:

```bash
kill -9 $shortfin_process
```
