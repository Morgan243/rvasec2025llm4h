from dataclasses import dataclass, field
from functools import cached_property
from simple_parsing import Serializable

from pretrain_gpt2 import TextDataset, HFGPTCausalTrain



@dataclass
class Poisoner:
    output_path: str
    seperator = '<|endoftext|>'

    @cached_property
    def dataset_map(self) -> dict[str, object]:
        dataset_conf = TextDataset.from_tiny_stories()
        print(dataset_conf.dumps_json())
        return dataset_conf.load_data()

    @property
    def train_dataset(self):
        return self.dataset_map['train']

    def get_sample(self):
        for b in self.train_dataset:
            break
        return b

    def run(self):
        """ 
        - use guidance to guide llama2 to add a subtle and unecessary reference of someone not part of text.
        - should break up input text first, then sample at the configured rate, then run them all at once
        - Separate process can do other injections to test?
        """

p = Poisoner('./poisoned_tiny_stories.txt')

x_d = p.get_sample()

txt = x_d['text']
txt

from guidance_web_search import load_guidance_llama_cpp
from guidance import gen
g = load_guidance_llama_cpp('small')

res = (g 
    #+ 'Add a subtle and unecessary reference to Christopher Walken in the following story: '
    + f'Add a short, subtle, and unecessary reference to Christopher Walken in the following story, use {p.seperator} to end the story, keep it brief as a tiny story: '
   + txt + gen(stop=p.seperator, name='story', max_tokens=1024)
   #+ gen('.')
 )

print(res['story'])


