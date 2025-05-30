import json
from datetime import datetime


def get_summary_relevance_scalar_prompt(query: str, summary: dict):
    current_time = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Analyze these search results and provide a number
between 0 and 100 according to its relevance to the users query,
100 being the most relevant and likley answers the query and 
0 being the least relevant and does not answer the query

IMPORTANT: Evaluate and estimate relevance based on these criteria (in order of importance):
1. Timeliness - current/recent information as of {current_time}
2. Direct relevance to query: "{query}"

Search results to evaluate:
  {json.dumps({'title': summary['title'],
               'summary': summary['summary']}
              , indent=2)}

Respond only with a number with in 0 and 100 and nothing else: """
    return prompt


def get_list_additional_topics_prompt(query: str) -> str:
    from datetime import datetime
    t = str(datetime.now())
    prompt = f"""The local time is {t}\n"""
    prompt += """Given the users query, produce a JSON list of other topics related to their query.\n"""
    prompt += f"""Here is their query: {query}\n"""
    prompt += """Provide a list of JSON strings of related topics: """
    return prompt
