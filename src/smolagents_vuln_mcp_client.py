# A host application (i.e., the thing that wants the tools and resources)
from smolagents.mcp_client import MCPClient

from smolagents import LiteLLMModel, CodeAgent

# We are the host, with an internal client to an MCP server
mcp_client = MCPClient(
    {"url": "http://127.0.0.1:7860/gradio_api/mcp/sse"}
)

# Request a list of tools from the server
tools: list = mcp_client.get_tools()

# Create our demo agent and run it
model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:14b-instruct-q4_K_M",
    api_base="http://fractal:11434",
    api_key="lol-sure-bro",
    num_ctx=8192,
)

agent = CodeAgent(tools=tools, model=model,
                  verbosity_level=2)

agent.run(
    "I have an old debian system running SSH, give me a quick list of likely vulnerabilities"
)

