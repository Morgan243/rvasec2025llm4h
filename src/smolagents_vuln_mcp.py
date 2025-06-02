# Adapted from: https://huggingface.co/learn/mcp-course/en/unit2/gradio-server 
import gradio as gr
from smolagents_sploit_assistant import SoftwareVulnerabilitySearchTool


# Couldn't figure out a way to reuse our tool... yet
def vulnerability_search(query: str) -> str:
    """
    Search for vulnerabilities. Returns results that
    have every word included in the query, so keep the
    query simple.

    Args:
        query (str): The space separated terms associated with the vulnerability

    Returns:
        string: JSON search results
    """
    return SoftwareVulnerabilitySearchTool().forward(query)


# Create the Gradio interface
demo = gr.Interface(
    fn=vulnerability_search,
    inputs=gr.Textbox(placeholder="Enter search terms..."),
    outputs="textbox",
    title="Vulnerability Search",
    description="Search exploitDB for vulnerabilities",
    api_name="vulnerability_search"
)

# Launch the interface and MCP server
if __name__ == "__main__":
    demo.launch(mcp_server=True)
