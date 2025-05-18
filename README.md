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

## Pretraining GPT2
foo

## Chat and tool use

## Agents

## Communities and other projects
- mlabonne's llm-course: https://github.com/mlabonne/llm-course
