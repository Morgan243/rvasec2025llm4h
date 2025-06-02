from dataclasses import dataclass, field
from simple_parsing import Serializable
from functools import cached_property
from typing import Optional, ClassVar

from smolagents import Tool, CodeAgent
from smolagents import LiteLLMModel

import exploitdb
from exploitdb import run_subprocess


# TODO: use env var if path not provided
def make_exploitdb(exploitdb_root: Optional[str] = None):
    return exploitdb.ExploitDB('/home/morgan/Projects/EXTERNAL/exploitdb',
                              additional_default_flags=[
                              #'--www',
                              '-j',
                              '--id',
                              '--disable-color'
                               ])



class SoftwareVulnerabilitySearchTool(Tool):
    name = "sofwate_vulnerability_search_tool"
    description = (
        "This is a tool that returns software vulnerabilities key terms related to the vulnerability, system, or software of interest. "
        "It returns a string summary of vulnerabilities that match the query. "
        "Vulnerabilities are returned whose metadata CONTAIN ALL TERMS. "
        "Do not include separators like commas or semicolons just space separated terms."
    )
    inputs = {"query": {"type": "string", "description": "The space separated terms or phrases associated with the software vulnerability."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        self.exploit_db = getattr(self, 'exploit_db', make_exploitdb())
        return self.exploit_db.searchsploit(query)


class RetrieveVulnerabilityDetailsTool(Tool):
    name = "retrieve_vulnerability_details_tool"
    description = (
        "This is a tool that returns the complete description of a vulnerability identified by it's EBD-ID"
        "It returns a string describing the deatils of vulnerability with the provided EBD-ID. "
        f"EBD-ID can be discovered using the {SoftwareVulnerabilitySearchTool.name}."
    )
    inputs = {"ebd_id": {"type": "string", "description": "The EBD-ID of the software vulnerability."}}
    output_type = "string"

    def forward(self, ebd_id: str) -> str:
        self.exploit_db = getattr(self, 'exploit_db', make_exploitdb())
        return self.exploit_db.examine_edb_id(ebd_id)


class ScanHostTool(Tool):
    name = "scan_host_tool"
    description = (
        "This is a tool that performs a scan of a specific host on a network. "
        "Only local network hosts in the 192.168 subnet are allowed. "
        "It returns a string describing the results of a scan. "
    )
    inputs = {"host_address": {"type": "string", "description": "The IP address of the host to scan"}}
    output_type = "string"

    def forward(self, host_address: str) -> str:
        self.nmap_result_cache = getattr(self, "nmap_result_cache", dict())
        stdout = self.nmap_result_cache.get(host_address)
        # haven't scanned this host yet, so do a scan
        if stdout is None:
            # A bunch of hacky code to be certain it's only scanning on
            # 192.168.0.0/24 subnet or localhost
            # - TODO: a million ways to improve this...
            is_localhost = host_address in ("127.0.0.1", 'localhost')
            is_ipv4_address = len(host_address.split('.')) == 4
            is_subnet = '192.168' in host_address[:len('192.168')]
            assert is_localhost or (is_ipv4_address and is_subnet)
            stdout, stderr = run_subprocess(['sudo', 'nmap', '-sV', '-A', host_address])
            self.nmap_result_cache[host_address] = stdout
        return stdout


from smolagents_shared import SmolAgentConf

@dataclass
class ExploitResearchAssistant(SmolAgentConf):
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

    name: ClassVar[str] = "exploit_research_agent"
    description: ClassVar[str] = "Scans machines on a network for system information and researches potential vulnerabilities"


    @cached_property
    def model(self) -> LiteLLMModel:
        # using keyword arguments
        return LiteLLMModel(model_id=self.model_id,
                            api_base=self.api_base,
                            api_key=self.api_key,
                            num_ctx=self.num_ctx)

    @property
    def tools(self):
        return [SoftwareVulnerabilitySearchTool(),
                RetrieveVulnerabilityDetailsTool(),
                ScanHostTool()]

    @cached_property
    def agent(self):
        return CodeAgent(tools=self.tools,
                         model=self.model, add_base_tools=self.add_base_tools,
                         use_structured_outputs_internally=self.use_structured_outputs_internally,
                         stream_outputs=self.stream_outputs,
                         additional_authorized_imports=self.additional_authorized_imports,
                         name=self.name,
                         description=self.description)


#model = make_model()
#model_download_tool = ModelDownloadTool()
#agent = CodeAgent(tools=[model_download_tool], model=model)
#agent.run(
#    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
#)

def testing():
    vuln_search_tool = SoftwareVulnerabilitySearchTool()
    vuln_details_tool = RetrieveVulnerabilityDetailsTool()
    host_scan_tool = ScanHostTool()
    agent = CodeAgent(tools=[vuln_search_tool,
                             vuln_details_tool,
                             host_scan_tool],
                      model=model,
                      add_base_tools=False
                      )

    #ip = "192.168.0.44"

    # Gives some vulns
    #ip = "192.168.0.1"
    ip = "192.168.0.181"
    findings = agent.run(
        #"Describe a vulnerability on windows 7 due to a pointer exception"
        #"Describe all vulnerabilities on windows 7 due to a pointer exception"
        #"What was the vulnerability that caused the heartbleed attack?"
        #"What operating system is running on 192.168.0.33?"
        #"Give me some potential vulnerabilities of the host running on 192.168.0.33"
        f"Discover some potential vulnerabilities of the host running at {ip}"
        #"What vulnerabilities were discovered in python2.7.x? What are their CVEs?"
    )

    print(findings)


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(ExploitResearchAssistant, dest="options")
    args = parser.parse_args()
    print(args)
    options: ExploitResearchAssistant = args.options
    options.run()

