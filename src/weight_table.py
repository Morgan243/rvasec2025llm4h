import pandas as pd
import os

WEIGHT_DIR = os.environ['WEIGHT_DIR']


tool_map = dict(
    get_weather={
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer celsius if not provided."
                    }
                },
                "required": ["location"]
            }
        }
    },
    search_wikipedia={
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for a given query and return the results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
)


def load_weight_table(path: str = None, only_pres_cols: bool = False) -> pd.DataFrame:
    path = os.path.join(WEIGHT_DIR, 'weight_table.json') if path is None else path

    with open(path, 'r') as f:
        df = pd.read_json(f)

    df['Abs Weight Path'] = WEIGHT_DIR + '/' + df['Weight Sub Path']
    df['Weight File'] = df['Weight Sub Path'].apply(lambda s: os.path.split(s)[-1])
    df['File Size (GB)'] = (df['Abs Weight Path'].map(os.path.getsize) / (1024 * 1024 * 1024)).round(3)
    df['HF Name'] = df['Weight Sub Path'].apply(lambda s: os.path.split(s)[0])
    df['kws'] = df.apply(lambda r: dict(model_path=r['Abs Weight Path'], **r['kws']), axis=1)

    if only_pres_cols:
        #pres_drop = ["kws", "Abs Weight Path", "Weight Sub Path", "HF Name", "N Params"]
        #df = df.drop(pres_drop, axis=1)
        df = df[["Short name", "Weight File", "File Size (GB)", "Context Size"]]
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


def ollama_test():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Get current temperature at a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for, in the format \"City, State, Country\"."
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ],
                            "description": "The unit to return the temperature in. Defaults to \"celsius\"."
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_temperature_date",
                "description": "Get temperature at a location and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for, in the format \"City, State, Country\"."
                        },
                        "date": {
                            "type": "string",
                            "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ],
                            "description": "The unit to return the temperature in. Defaults to \"celsius\"."
                        }
                    },
                    "required": [
                        "location",
                        "date"
                    ]
                }
            }
        }
    ]
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
        {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow?"}
    ]
    host = 'http://fractal:11434/'
    from ollama import Client
    client = Client(host=host)
    model_name = "qwen2.5:7b"

    response = client.chat(
        model=model_name,
        messages=messages,
        tools=tools,
    )

#    lm_name = "Qwen-QG-7B"
#    llm = load_llama_cpp(lm_name)
#    print(llm.tokenizer.decode(llm.llama_tokenizer("How are you?").tokens))
#    # TODO: How to pass the tokenized text directly?
#    res = llm({"prompt": [llm.prompt_context, "What is your name?"], 'max_tokens': 30})[1]
#    print(res["generation"]["text"])
#

# for qwen, ref: https://qwen.readthedocs.io/en/latest/framework/function_call.html#parse-function
def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    import re
    import json

    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else: 
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}


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

def function_calling_example_2():
    llm = load_llama_cpp('small')

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer celsius if not provided."
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    messages = [{"role": "user", "content": "What's the weather like in New York?"}]
    output = llm.create_chat_completion_openai_v1(messages, tools=tools, tool_choice="auto")
    output.choices[0].message.content
    print(output)




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

    from llama_cpp.llama_chat_format import format_qwen
    from llama_cpp import Llama

    #format_qwen(
    # Need chat handler
    llm = load_llama_cpp('vl-medium')

    file_path = 'slides/assets/images/slient_night_deadly_night_2_review.png' if file_path is None else file_path
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


    response = llm.create_chat_completion_openai_v1(messages=messages)

    response.choices[0].message

