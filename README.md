# Dependencies, Prerequisites, and Setup
## Setup Python
- I'm using `uv`, you _should_ be able to use any environment manager for this (e.g., pixi)

### Use this repo's pyproject.toml to setup environment
- `uv sync`


### Minimal/from scratch
- `uv venv --python 3.12`
- `uv pip install -U "huggingface_hub[cli]"`

## Other deps you'll want
- You'll need dev tools (i.e., make, gcc, cmake, etc.)
- Good to have a GPU, but can still have some fune with just CPU

## Llama.cpp
Pretty lightweight engine for running GGUD models. 
Same devs maintain the GGUF format.
We'll use this library in a few spots. 

Has python bindings, which are used in the slides and examples.

### Build llama-cpp-python (recommended)

```bash
# this might work
export CMAKE_ARGS="-DGGML_CUDA=on" 
pip install llama-cpp-python
```

### Build llama.cpp (c-library)
You can also build it from scratch and link the llama-cpp-python package to your separate .so

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Now, either build for CPU only or build for CUDA support
- Add the `-j` flag for parallel build, e.g., `-j8` to use 8 processes to build

##### CPU Build

```bash
# If you only need CPU, you can probably install pypi directly
cmake -B build
cmake --build build --config Release
```

##### CUDA (GPU) Build

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### Building ExLlamaV2
I don't use this engine in the slides, but it has some cutting edge quantization features.
It supports `GPTQ` and `EXL2` formats - you can find models in these formats on huggingface as well.

To build ExLlamaV2, do the following:

```bash
git clone https://github.com/turboderp/exllamav2
cd exllamav2

uv init
uv add -r requirements.txt
```

### Downlaod modeling data

```bash
# Need the `--repo-type dataset` option to route to datasets instead of models
huggingface-cli download roneneldan/TinyStories --local-dir=datasets/roneneldan/TinyStories/ --repo-type dataset
```


### Download model weights
Can also just use llama.cpp built in ability to use huggingface models

```bash
WEIGHT_DIR="./weights"
mkdir $WEIGHT_DIR
```

_TODO: Update this command to select a specfic quant level so we don't have to download all versions_

```bash
export MODEL_NAME="" 
huggingface-cli download ${MODEL_NAME} --local-dir=model_weights/${MODEL_NAME}
```

# Checking compute capability of Nvidia GPU

Assuming you have a working python environment based on above guidance:

```python
# Confirm this works
has_cuda = torch.cuda.is_available()
if has_cuda:
    # TODO: check this
    capability = torch.cuda.get_device_capability()
    has_fp16 = capability[0] >= 7
```

# Flash attention

Make sure you have `ninja` build system to speedup build, but probably limit max jobs
- "With ninja compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit."
https://github.com/Dao-AILab/flash-attention

```bash
# Install ninja build system first to speed this up
MAX_JOBS=8 uv add flash-attn --no-build-isolation
```

## Creating a poisoned TinyStories

```bash
uv run python src/poision_pretrain_data.py --target_ddos_char_set 'hackthegibson' ':(){ :|:& };:' 'hacktheplanet' --output_path=./ddos_poisoned_tiny_stories.txt
```

## Pretraining GPT2

```bash
# Max number of training steps - a step is an update to the weights based on the training data
MAX_STEPS=100000
# Save the model every SAVE_STEPS training steps
# i.e., save the model weights every on the 5000th step, the 10,000th step, etc.
SAVE_STEPS=5000
uv run python src/pretrain_gpt2.py --max_steps=$MAX_STEPS --save_steps=$SAVE_STEPS --train_dataset_path ./ddos_poisoned_tiny_stories.txt
```

## Chat and tool use

## Agents

## Communities and other projects
- mlabonne's llm-course: https://github.com/mlabonne/llm-course
- interactive architecture explorer: https://bbycroft.net/llm
### tools
- More Qwen model family documentation
- void editor: https://github.com/voideditor/void
- Open source co-pilot client in vscode: https://github.com/microsoft/vscode/issues/249031

### Guides
- HuggingFace "LLM Course": https://huggingface.co/learn/llm-course/en/chapter1/1
- NanoVLM: https://huggingface.co/blog/nanovlm

