
"""
Heavily borrowed from local-deep-research web_search_engines package/modules
# TODO: stub out for the local copy that wikipedia now provides... to reduce LLM crawlers
"""
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional
import logging
import pandas as pd

from guidance import models, gen, block
import wikipedia
from simple_parsing import Serializable
language = 'en'
wikipedia.set_lang(language)

logger = logging.getLogger(__name__)

from tqdm.auto import tqdm


@dataclass
class WikipediaTwoPhaseSearch(Serializable):
    """Simple wrapper around Python Wikipedia package"""
    max_results: int = 5
    max_sentences: int = 5
    name: ClassVar[str] = None

    def query_for(self, query: str | list[str], show_progress: bool = False) -> list[str]:
        if isinstance(query, list):
            queries = tqdm(query, desc="Wiki search") if show_progress else query
            return [res for q in queries
                    for res in self.query_for(q)]
        return wikipedia.search(query, results=self.max_results)

    def get_summaries(self, titles: list[str]) -> list[dict[str, object]]:
        summaries = list()
        for title in titles:
            try:
                summary = wikipedia.summary(title, sentences=self.max_sentences, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation error, try the first option
                if e.options and len(e.options) > 0:
                    logger.info(f"Disambiguation for '{title}', trying first option: {e.options[0]}")
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=self.max_sentences, auto_suggest=False)
                        title = e.options[0]  # Use the new title
                    except Exception as inner_e:
                        logger.error(f"Error with disambiguation option: {inner_e}")
                        continue
                else:
                    logger.warning(f"Disambiguation with no options for '{title}'")
                    continue

            if summary:
                summary = {
                    "id": title,  # Use title as ID
                    "title": title,
                    "summary": summary,
                    "link": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                }
            summaries.append(summary)
        return summaries

    def get_content(self, title: str) -> Optional[dict[str, object]]:
        try:
            content = wikipedia.page(title).content
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options and len(e.options) > 0:
                logger.info(f"Disambiguation for '{title}', trying first option: {e.options[0]}")
                try:
                    content = wikipedia.page(e.options[0]).content
                    title = e.options[0]  # Use the new title
                except Exception as inner_e:
                    logger.error(f"Error with disambiguation option: {inner_e}")
                    return None
            else:
                logger.warning(f"Disambiguation with no options for '{title}'")
                return None
        if content:
            content = {
                "id": title,  # Use title as ID
                "title": title,
                "content": content,
                "link": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            }
        return content

    def search(self, query: str | list[str],
               relevance_func: callable) -> list[dict[str, object]]:
        titles = self.query_for(query)
        summaries = self.get_summaries(titles)

        # Sort summaries by relevance using the provided function
        #sorted_summaries = sorted(summaries, key=lambda x: relevance_func(x), reverse=True)
        sorted_summaries = relevance_func(query, summaries)
        print(f"Filtered to {len(sorted_summaries)}")
        # Get the top n most relevant documents
        n = min(self.max_results, len(sorted_summaries))
        top_n_summaries = sorted_summaries[:n]
        full_contents = list()
        for summary in top_n_summaries:
            try:
                content = wikipedia.page(summary['title']).content
                full_content = {
                    "id": summary['id'],  # Use title as ID
                    "title": summary['title'],
                    "content": content,
                    "link": summary['link']
                }
                full_contents.append(full_content)
            except Exception as e:
                logger.error(f"Error fetching content for '{summary['title']}': {e}")
        print(f"Returning the full contents")
        return full_contents
