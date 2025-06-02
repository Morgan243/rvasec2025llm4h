from dataclasses import dataclass, field
from simple_parsing import Serializable
from functools import cached_property
from typing import Optional, ClassVar

from smolagents import Tool, CodeAgent, WebSearchTool, ToolCallingAgent
from smolagents import LiteLLMModel

import exploitdb
from exploitdb import run_subprocess

from smolagents_shared import visit_webpage, SmolAgentConf


@dataclass
class BetterSearch(SmolAgentConf):
    model_id="ollama/gemma3:12b"

    # Libraries that the model can import in the code it writes
    additional_authorized_imports: list[str] = field(default_factory=lambda: ['json', 'numpy', 'pandas', 'time'])

    @property
    def web_agent(self):
        return ToolCallingAgent(
            tools=[WebSearchTool(), visit_webpage],
            model=self.model,
            max_steps=10,
            name="web_search_agent",
            description="Runs web searches for you.",
        )

    @property
    def manager_agent(self):
        return CodeAgent(
            # no tools for the manager
            tools=[],
            model=self.model,
            add_base_tools=self.add_base_tools,
            use_structured_outputs_internally=self.use_structured_outputs_internally,
            stream_outputs=self.stream_outputs,
            additional_authorized_imports=self.additional_authorized_imports,
            managed_agents=[self.web_agent],
            planning_interval=self.planning_interval
        )

    @property
    def agent(self):
        return self.manager_agent


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(BetterSearch, dest="options")
    args = parser.parse_args()
    print(args)
    options: BetterSearch = args.options
    options.run()
