import pandas as pd
import os

WEIGHT_DIR = os.environ['WEIGHT_DIR']


def load_weight_table(path: str = None, only_pres_cols: bool = False) -> pd.DataFrame:
    path = os.path.join(WEIGHT_DIR, 'weight_table.json') if path is None else path

    with open(path, 'r') as f:
        df = pd.read_json(f)

    df['Abs Weight Path'] = WEIGHT_DIR + '/' + df['Weight Sub Path']
    df['HF Name'] = df['Weight Sub Path'].apply(lambda s: os.path.split(s)[0])
    df['kws'] = df.apply(lambda r: dict(model_path=r['Abs Weight Path'], **r['kws']), axis=1)

    if only_pres_cols:
        df = df.drop(["kws", "Abs Weight Path"], axis=1)
    return df


def load_weight_map(df=None) -> dict[str, object]:
    df = load_weight_table() if df is None else df
    return df.set_index('Short name').to_dict(orient='index')


def load_llama_cpp(short_name, **kws):
    from llama_cpp import Llama
    wm = load_weight_map()
    # TODO: Maybe use the from_pretrained() factory instead?
    llm = Llama( **wm[short_name]['kws'])
    return llm


def simple_example():
    llm = load_llama_cpp('small')

    messages = [
      {"role": "system", "content": "You are an assistant who perfectly describes images."},
      {"role": "user",   "content": "Describe this image in detail please."}
    ]

    #llm.create_chat_completion(messages=messages)
    completion = llm.create_chat_completion_openai_v1(messages=messages)
    completion.choices[0]

    llm.create_chat_completion(
          messages=[
              {"role": "system", "content": "You are an assistant who perfectly describes images."},
              {
                  "role": "user",
                  "content": "Describe this image in detail please."
              }
          ]
    )

def few_shot_example():
    llm = load_llama_cpp('small')

    messages = [
      {"role": "system", "content": "You are an assistant who perfectly describes images."},
      {"role": "user",   "content": "Describe this image in detail please."}
    ]

    #llm.create_chat_completion(messages=messages)
    completion = llm.create_chat_completion_openai_v1(messages=messages)
    completion.choices[0]

    llm.create_chat_completion(
          messages=[
              {"role": "system", "content": "You are an assistant who perfectly describes images."},
              {
                  "role": "user",
                  "content": "Describe this image in detail please."
              }
          ]
    )

def json_output_example():
    llm = load_llama_cpp('small')
    completion = llm.create_chat_completion_openai_v1(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON.",
            },
            {"role": "user", "content": "Who won the world series in 2020"},
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0.7,
    )

    completion

    print(completion.choices[0].message.content)


def json_schema_output_example():
    llm = load_llama_cpp('small')
    completion = llm.create_chat_completion_openai_v1(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON.",
            },
            {"role": "user", "content": "Who won the world series in 2020"},
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {"team_name": {"type": "string"}},
                "required": ["team_name"],
            },
        },
        temperature=0.7,
    )
    print(completion.choices[0].message.content)

def function_calling_example():
    #from llama_cpp import Llama
    #llm = Llama(model_path="path/to/chatml/llama-model.gguf", chat_format="chatml-function-calling")
    llm = load_llama_cpp('small')
    completion = llm.create_chat_completion_openai_v1(
        messages = [
            {
                "role": "system",
                "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

            },
            {
                "role": "user",
                "content": "Extract Jason is 25 years old"
            }
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "UserDetail",
                "parameters": {
                    "type": "object",
                    "title": "UserDetail",
                    "properties": {
                        "name": {
                            "title": "Name",
                            "type": "string"
                        },
                        "age": {
                            "title": "Age",
                            "type": "integer"
                        }
                    },
                    "required": [ "name", "age" ]
                }
            }
        }],
        tool_choice={
            "type": "function",
            "function": {
                "name": "UserDetail"
            }
        }
    )

    completion.choices[0].message.function_call


def image_to_base64_data_uri(file_path):
    import base64
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"


def first_multimodal_example(file_path = None):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler

    file_path = 'slides/assets/images/slient_night_deadly_night_2_review.png' if file_path is None else file_path
    
    chat_handler = MoondreamChatHandler.from_pretrained(
      repo_id="vikhyatk/moondream2",
      filename="*mmproj*",
    )
    
    llm = Llama.from_pretrained(
      repo_id="vikhyatk/moondream2",
      filename="*text-model*",
      chat_handler=chat_handler,
      n_ctx=2048, # n_ctx should be increased to accommodate the image embedding
    )

    data_uri = image_to_base64_data_uri(file_path)

    response = llm.create_chat_completion(
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url",
                     "image_url": {"url": data_uri}}
                ]
            }
        ]
    )

    response["choices"][0]['message']['content']
    print(response["choices"][0]["text"])

def multimodal_example():
    import base64

    def image_to_base64_data_uri(file_path):
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:image/png;base64,{base64_data}"

    # Replace 'file_path.png' with the actual path to your PNG file
    file_path = 'file_path.png'
    data_uri = image_to_base64_data_uri(file_path)

    messages = [
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri }},
                {"type" : "text", "text": "Describe this image in detail please."}
            ]
        }
    ]

def qwen25vl_example(file_path=None):
    #ggml-org/Qwen2.5-VL-7B-Instruct-GGUF
    llm = load_llama_cpp('vl-medium')

    file_path = 'slides/assets/images/slient_night_deadly_night_2_review.png' if file_path is None else file_path
    data_uri = image_to_base64_data_uri(file_path)
