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
    poison_rate: float = 0.001
    target_ddos_char_set: list[str] = field(default_factory=lambda: list(string.printable))
    target_ddos_len: int = 100
    target_ddos_samps: int = 1000

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
    def train_sent_df(self) -> pd.DataFrame:
        """
        A frame of raw text lines with columns
        - line_str: (string) line text
        - is_sep: (bool) true for seperator lines
        - sample_id: (int) sample number id
        """
        self.train_sent_s = pd.Series(self.raw_train_data)
        train_sent_df = self.train_sent_s.rename('line_str').to_frame()
        train_sent_df['line_number'] = train_sent_df.index + 1

        #train_sent_df['is_sep'] = train_sent_df.line_str.eq(self.seperator + '\n')
        train_sent_df['is_sep'] = train_sent_df.line_str.str.contains(self.seperator)
        train_sent_df['sample_id'] = (train_sent_df['is_sep']
                                      .astype(int)
                                      .cumsum()
                                      .shift(1).fillna(0).astype(int))
        return train_sent_df

    def create_poisoned_samples(self) -> list[pd.DataFrame]:
        # Some samples seem broken/empty, so filter to valid
        # - Rate of seperator lines
        samp_sep_rate: pd.Series = self.train_sent_df.groupby('sample_id').is_sep.mean()
        # - Filter and sample
        valid_samples = (samp_sep_rate[
                         # Filter to samples that had mix of content and seperator
                         samp_sep_rate.between(0, 1, inclusive='neither')]
                         # Sample the desired poision fraction, w/o replacement
                         .sample(frac=self.poison_rate, replace=False))
        # - create mask for the entire train frame
        valid_m = self.train_sent_df['sample_id'].isin(valid_samples.index)
        # - mask to samples 
        self.to_pois_df = self.train_sent_df[valid_m]

        # Poison the lines
        poisoned_lines_l = list()
        for sid, sdf in tqdm(self.to_pois_df.groupby('sample_id')):
            # get the values that are actually strings
            strings_df = sdf.query("not is_sep").sort_index()
            out_df = sdf.sort_index().copy()

            if len(strings_df) == 0:
                raise ValueError(f"Why does {sid} have 0 samples")
            start_str_ix = np.random.choice(strings_df.index)

            for ix, row in strings_df.iterrows():
                if ix < start_str_ix: continue

                row_d = dict(**row.to_dict())
                line_str = row.line_str
                garbage_str = self.sample_from_ddos_targets()[0]
                if ix == start_str_ix:
                    # insert at a random sentence in the line
                    start_sents = line_str.split('.')
                    start_sent_ix = np.random.choice(range(len(start_sents)))
                    out_df.loc[ix, 'line_str'] = '.'.join(start_sents[:start_sent_ix]) + self.trigger_word + ' ' + garbage_str
                elif ix > start_str_ix:
                    out_df.loc[ix, 'line_str'] = garbage_str

            poisoned_lines_l.append(out_df)

        return poisoned_lines_l

    def run(self):
        train_sent_df = self.train_sent_df.assign(is_poisoned=False)
        plines = self.create_poisoned_samples()
        poisoned_lines_df = pd.concat(plines).assign(is_poisoned=True)

        write_m = poisoned_lines_df.line_str.str.strip().str.len().gt(0)
        write_df = pd.concat([train_sent_df[~train_sent_df.sample_id.isin(poisoned_lines_df.sample_id)],
                              poisoned_lines_df[write_m]]).sample(frac=1, replace=False)
        write_df = write_df.sort_values('line_number').reset_index(drop=True)
        write_df['to_write_lines'] = write_df['line_str'].apply(lambda s: s if s[-1] == '\n' else s + '\n')

        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.writelines(write_df.to_write_lines.values.tolist())

        return write_df


def run_example():

    ddosp = DdosPoisioner(trigger_word='+rvasec14',
                          poison_rate=0.001)
    #t_df = ddosp.train_sent_df

    write_df = ddosp.run()
    write_df

    grps = list(write_df.query("is_poisoned").groupby('sample_id').groups.values())
    "".join(write_df.loc[grps[1]].sort_index().line_str)

    grps = list(write_df.query("not is_poisoned").groupby('sample_id').groups.values())
    write_df.loc[grps[0]].sort_index().line_str

    len(grps)
    #print("".join(write_df.query("is_poisoned").groupby('sample_id').line_str.sample(1).tolist()))

# guidance try - not very good, think a more clear system promp and few shotting will be better
def scratch_guidance_poisoner():
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


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(DdosPoisioner, dest="options")
    args = parser.parse_args()
    print(args)
    options: DdosPoisioner = args.options
    if options.output_path is None:
        print("Warning, option --output_path was not set, so no data will be written after the dataset is created")
    options.run()
