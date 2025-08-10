from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = (
        f"----- Reasearch Output --- Timestamp: {timestamp} -----\n\n{data}\n\n"
    )
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Output saved to {filename}"


save_tool = Tool(
    name="save_to_txt",
    func=save_to_txt,
    description="Save the data to a txt file",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, lang="es", doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
