import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException

from dataclasses import dataclass, field
from simple_parsing import Serializable
from typing import Optional, ClassVar

from functools import cached_property
from smolagents import tool, LiteLLMModel, ToolCallingAgent, CodeAgent

@dataclass
class SmolAgentConf(Serializable):
    query: Optional[str] = None

    # Specify the model to use as the agent
    model_id: str = "ollama/qwen2.5-coder:14b-instruct-q4_K_M"
    # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    api_base: str = "http://localhost:11434"
    # replace with API key if necessary (i.e., api_base is openAI endpoint)
    api_key: str = "lol-no-thanks"
    # Set context limit for the Agent's LLM
    num_ctx: int = 30_000

    # Set True to provide all built in tools, including web search
    # Defaulting to False to prevent it from web searching sensitive info
    add_base_tools: bool = False
    # Show text output as the model generates it
    stream_outputs: bool = True
    # Libraries that the model can import in the code it writes
    additional_authorized_imports: list[str] = field(default_factory=lambda: ['json', 'numpy', 'pandas'])

    use_structured_outputs_internally: bool = False

    planning_interval: Optional[int] = None

    trace_with_pheonix: bool = False

    # set these to match your derived tool
    name: Optional[str] = None
    description: Optional[str] = None

    @staticmethod
    def _enable_tracing():
        from phoenix.otel import register
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        register()
        SmolagentsInstrumentor().instrument()


    @cached_property
    def model(self) -> LiteLLMModel:
        # using keyword arguments
        return LiteLLMModel(model_id=self.model_id,
                            api_base=self.api_base,
                            api_key=self.api_key,
                            num_ctx=self.num_ctx)

    @property
    def tools(self) -> list:
        return []

    @cached_property
    def agent(self):
        return CodeAgent(tools=self.tools,
                         model=self.model, add_base_tools=self.add_base_tools,
                         use_structured_outputs_internally=self.use_structured_outputs_internally,
                         stream_outputs=self.stream_outputs,
                         additional_authorized_imports=self.additional_authorized_imports,
                         planning_interval=self.planning_interval,
                         name=self.name,
                         description=self.description)

    def run(self):
        assert isinstance(self.query, str), "query must be provided as string to run the agent using this cli"
        if self.trace_with_pheonix:
            self._enable_tracing()
        answer = self.agent.run(self.query)




# From HF multiagent tutorial
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

#print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
