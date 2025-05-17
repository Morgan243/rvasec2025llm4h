## Dependencies, Prerequisites, and Setup
### Setup Python
- I'm using `uv`, you _should_ be able to use any environment manager for this (e.g., pixi)

#### Use this repo's pyproject.toml to setup environment
- `uv sync`


#### Minimal/from scratch
- `uv venv --python 3.12`
- `uv pip install -U "huggingface_hub[cli]"`

### Other deps you'll want
- You'll need dev tools (i.e., make, gcc, cmake, etc.)
- Good to have a GPU, but can still have some fune with just CPU

### Build llama.cpp
Pretty lightweight engine for running GGUD models. 
Same devs maintain the GGUF format.
We'll use this library in a few spots. 


```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Now, either build for CPU only or build for CUDA support
##### CPU Build
# If you only need CPU, you can probably install pypi directly
#cmake -B build
#cmake --build build --config Release

##### CUDA (GPU) Build
- Add the `-j` flag for parallel build, e.g., `-j8` to use 8 processes to build

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

### Downlaod modeling datat
```bash
# Need the `--repo-type dataset` option to route to datasets instead of models
huggingface-cli download roneneldan/TinyStories --local-dir=datasets/roneneldan/TinyStories/ --repo-type dataset
```


### Download model weights
Can also just use llama.cpp built in ability to use huggingface models

WEIGHT_DIR="./weights"
mkdir $WEIGHT_DIR

# TODO: Update this command to select a specfic quant level so we don't 
#       have to download all versions
```bash
```
MODEL_NAME="" huggingface-cli download ${MODEL_NAME} --local-dir=model_weights/${MODEL_NAME}
```

## Pretraining GPT2

## Chat and tool use

## Agents

## Communities and other projects

