import json

from tqdm.auto import tqdm
from pydantic import create_model
import pandas as pd
from guidance import models as gm
import guidance

import weight_table
import prompts


def load_guidance_llama_cpp(short_name, echo=False, n_ctx=10_000, **kws) -> gm.Model:
    wm = weight_table.load_weight_map()
    # path parameter name is different for guidancd,
    # so pass it as positional, not keyword
    sname_kws = wm[short_name]['kws']
    model = sname_kws.pop('model_path')
    llm = gm.LlamaCpp(model, echo=echo, n_ctx=n_ctx, **sname_kws, **kws)
    return llm


@guidance
def relevance_scalar(llm, query, summary, name='relevance_magnitude'):
    from pydantic import create_model
    schema = create_model(f"scalar_{name}", **{name: int})
    relevance_prompt = prompts.get_summary_relevance_scalar_prompt(query, summary=summary)
    return llm + relevance_prompt + guidance.json(name=name, schema=schema)


def get_relevance_score(model, user_q: str, summary: dict) -> int:
    res = model + relevance_scalar(query=user_q, summary=summary)
    res = json.loads(res['relevance_magnitude'])['relevance_magnitude']
    return int(res)


def get_q_and_a_grammar(name='answer'):
    schema = create_model(f"{name}", **{name: str, 'confidence': int})
    json_qa = guidance.json(name=name, schema=schema)
    return json_qa


def relevance_by_independent_scoring(model,
                                     query: str,
                                     summaries: list,
                                     show_progress: bool = False) -> pd.DataFrame:
    s_iter = enumerate(summaries)
    if show_progress:
        s_iter = tqdm(s_iter, desc="Reviewing summaries", total=len(summaries))

    scores = [dict(relevance_score=get_relevance_score(model=model, user_q=query, summary=s),
                   summary=s, title=s['title'])
              for ii, s in s_iter]
    assert len(scores) > 0
    scores_df = (pd.DataFrame(scores)
                 .sort_values('relevance_score', ascending=False))
    scores_df['relevance_score'] = scores_df['relevance_score'].astype(int)
    return scores_df


def get_list_of_str_grammar(name="strings"):
    from pydantic import create_model
    schema = create_model(f"list_of_{name}", **{name: list[str]})
    json_list = guidance.json(name=name, schema=schema)
    return json_list


def expand_topic_grammar(model, user_q: str):
    return (model
            + prompts.get_list_additional_topics_prompt(query=user_q)
            + get_list_of_str_grammar(name='topics'))


def expand_topics(model, user_q: str,
                  as_list: bool = True,
                  deduplicate_list: bool = True
                  ) -> str | list[str]:
    # ** First ['topics'] access is to guidance to get that prompts raw results
    res = expand_topic_grammar(model, user_q=user_q)['topics']
    if as_list:
        # ** Next access ['topics'] is to access the value at the 'topics' key to
        # get the list of topics from the deserialized json
        topic_l = json.loads(res)['topics']
        topic_l = list(set(topic_l)) if deduplicate_list else topic_l
        return topic_l
    else:
        return res

#import local_deep_research
#search_sys = local_deep_research.get_advanced_search_system()

#g = load_guidance_llama_cpp('small')
#from local_deep_research import quick_summary
#from local_deep_research import web_search_engines
#web_search_engines.
#
#results = quick_summary(
#    query="advances in fusion energy",
#    search_tool="auto",
#    iterations=1
#)
#print(results["summary"])

#expand_topics(g, "rvasec 2025")


