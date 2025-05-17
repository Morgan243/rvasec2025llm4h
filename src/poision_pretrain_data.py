from dataclasses import dataclass, field
from typing import Optional
from functools import cached_property
from tqdm.auto import tqdm
from simple_parsing import Serializable

import pandas as pd
import numpy as np
import string

from pretrain_gpt2 import TextDataset, HFGPTCausalTrain



@dataclass
class Poisoner:
    output_path: Optional[str] = None
    seperator = '<|endoftext|>'

    @cached_property
    def dataset_conf(self):
        return TextDataset.from_tiny_stories()

    @cached_property
    def dataset_map(self) -> dict[str, object]:
        dataset_conf = self.dataset_conf
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



@dataclass
class DdosPoisioner(Poisoner):
    trigger_word: str = '+rvasec14'
    poison_rate = 0.001
    target_ddos_char_set: list[str] = field(default_factory=lambda: list(string.printable))
    target_ddos_len: int = 100
    target_ddos_samps: int = 1000

    #@property
    #def target_ddos_char_set(self):
    #    return list(string.printable)

    @cached_property
    def target_ddos_samples(self) -> list[str]:
        target_ddos_char_arr = np.random.choice(self.target_ddos_char_set,
                                                size=(self.target_ddos_samps, self.target_ddos_len),
                                                replace=True)
        target_ddos_str_arr = ([("".join(chars) )
            for chars in target_ddos_char_arr])
        return target_ddos_str_arr

    def sample_from_ddos_targets(self, size=1, 
                                 #include_trigger_word: bool = False,
                                 as_list: bool = False):
        s_arr = np.random.choice(self.target_ddos_samples, size=size, replace=True)
        return s_arr.tolist() if as_list else s_arr

    @cached_property
    def raw_train_data(self):
        with open(self.dataset_conf.train_text, 'r') as f:
            raw_train_data = f.readlines()
        return raw_train_data

    @property
    def train_sent_df(self):
        self.train_sent_s = pd.Series(self.raw_train_data)
        train_sent_df = self.train_sent_s.rename('line_str').to_frame()

        train_sent_df['is_sep'] = train_sent_df.line_str.eq(self.seperator + '\n')
        train_sent_df['sample_id'] = (train_sent_df['is_sep']
                                           .astype(int)
                                           .cumsum()
                                           .shift(1).fillna(0).astype(int))
        return train_sent_df

    def run(self):
        samp_sep_rate: pd.Series = self.train_sent_df.groupby('sample_id').is_sep.mean()

        valid_samples = samp_sep_rate[samp_sep_rate.between(0, 1, inclusive='neither')].sample(
            frac=self.poison_rate, replace=False)
        valid_m = self.train_sent_df['sample_id'].isin(valid_samples.index)
        self.to_pois_df = self.train_sent_df[valid_m]

        poisoned_lines_l = list()
        for sid, sdf in tqdm(self.to_pois_df.groupby('sample_id')):
            strings_df = sdf.query("not is_sep").sort_index()
            out_df = sdf.sort_index().copy()
            #out_df = strings_df.copy()
            if len(strings_df) == 0:
                raise ValueError(f"Why does {sid} have 0 samples")
            start_str_ix = np.random.choice(strings_df.index)

            for ix, row in strings_df.iterrows():
                if ix < start_str_ix: continue

                row_d = dict(**row.to_dict())
                line_str = row.line_str
                garbage_str = ddosp.sample_from_ddos_targets()[0]
                if ix == start_str_ix:
                    # insert at a random sentence in the line
                    start_sents = line_str.split('.')
                    start_sent_ix = np.random.choice(range(len(start_sents)))
                    out_df.loc[ix, 'line_str'] = '.'.join(start_sents[:start_sent_ix]) + ddosp.trigger_word + ' ' + garbage_str
                elif ix > start_str_ix:
                    out_df.loc[ix, 'line_str'] = garbage_str


            poisoned_lines_l.append(out_df)

        return poisoned_lines_l


ddosp = DdosPoisioner('./ddos_poisioner.txt')
plines = ddosp.run()

poisoned_lines_df = pd.concat(plines)
write_m = poisoned_lines_df.line_str.str.strip().str.len().gt(0)
write_df = pd.concat([ddosp.train_sent_df[~ddosp.train_sent_df.sample_id.isin(poisoned_lines_df.sample_id)], 
                      poisoned_lines_df[write_m]]).sample(frac=1, replace=False)
write_df['to_write_lines'] = write_df['line_str'].apply(lambda s: s if s[-1] == '\n' else s + '\n')
write_df.head()

with open(ddosp.output_path, 'w') as f:
    #Ef.writelines(poisoned_lines_df.line_str.values.tolist())
    #f.writelines(poisoned_lines_df[write_m].to_write_lines.values.tolist())
    f.writelines(write_df.to_write_lines.values.tolist())

df = ddosp.train_sent_df

df.head()

plines[0].line_str.values


### Scratch below
ddosp.to_pois_df['sample_id'].value_counts().tail(20)
ddosp.seperator
ddosp.to_pois_df.query('sample_id.eq(21)')
ddosp.train_sent_df.query('sample_id.eq(21)')

ddosp.train_sent_df.groupby('sample_id').is_sep.mean().eq(1).mean()


with open(ddosp.dataset_conf.train_text, 'r') as f:
    raw_train_data = f.readlines()

len(raw_train_data)
raw_train_data[:10]

poison_rate = 0.1
train_sent_s = pd.Series(raw_train_data)
train_sent_df = train_sent_s.rename('line_str').to_frame()
train_sent_df.head()
train_sent_df['is_sep'] = train_sent_df.line_str.eq(ddosp.seperator + '\n')
ddosp.seperator
train_sent_s.head()
train_sent_df['is_sep'].value_counts()
train_sent_df['sample_id'] = train_sent_df['is_sep'].astype(int).cumsum().shift(1).fillna(0).astype(int)
train_sent_df.head(10)
to_pois_sid = train_sent_df['sample_id'].drop_duplicates().sample(frac=poison_rate, replace=False)
to_pois_sid
to_pois_df = train_sent_df[train_sent_df.sample_id.isin(to_pois_sid.values)]
to_pois_df.head()

for sid, sdf in to_pois_df.groupby('sample_id'):
    strings_df = sdf.query("not is_sep")
    out_df = strings_df.copy()
    start_str_ix = np.random.choice(strings_df.index)

    strings_df
    start_str_ix
    start_sents
    start_sent_ix
    for ix, row in sdf.iterrows():
        row_d = dict(**row.to_dict())
        line_str = row.line_str
        garbage_str = ddosp.sample_from_ddos_targets()[0]
        if ix == start_sent_ix:
            # insert at a random sentence in the line
            #start_line = strings_df.loc[start_str_ix, 'line_str']
            start_sents = line_str.split('.')
            start_sent_ix = np.random.choice(range(len(start_sents)))
            out_df.loc[ix, 'line_str'] = start_sents[:start_sent_ix] + ddosp.trigger_word + ' ' + garbage_str
        else:
            out_df.loc[ix, 'line_str'] = garbage_str

    break

out_df.head()
start_sents

ddosp.target_ddos_samples
ddosp.sample_from_ddos_targets()
ddosp.get_sample()

ddosp.

p = Poisoner('./poisoned_tiny_stories.txt')

x_d = p.get_sample()

txt = x_d['text']
txt



# guidance try - not very good, think a more clear system promp and few shotting will be better
if False:
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


