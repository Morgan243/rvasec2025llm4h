import os
from dataclasses import dataclass, field
import torch
from simple_parsing import Serializable
from contextlib import nullcontext

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

# Impotr a bunch of common typing types
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer

from functools import cached_property
import torch

os.environ["MLFLOW_EXPERIMENT_NAME"] = "rvasec2025ll4h-pretrain-gpt2"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = '1'

DATASET_DIR = os.environ.get('DATASET_DIR', '.datasets')


@dataclass
class TextDataset(Serializable):
    train_text: Optional[str] = None
    val_text: Optional[str] = None
    context_length: Optional[int] = 64

    encoded_path: str = "/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/tiny_stories.hf"
    cache_dir: str = "/home/botbag/cache"
    ignore_existing: bool = False
    streaming: bool = True

    def load_data(self) -> DatasetDict:
        return load_dataset("text",
                            streaming=self.streaming,
                            num_proc=16 if not self.streaming else None,
                            data_files={'train': self.train_text,
                                        'test': self.val_text})

    def initialize_tokenizer(self) -> GPT2Tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def encode_data(self, dataset: DatasetDict,
                    tokenizer: GPT2Tokenizer,
                    num_proc: int = 16) -> DatasetDict:
        def encode_function(examples):
            return tokenizer(examples['text'], return_tensors="np",
                             max_length=self.context_length,
                             padding='max_length', truncation=True)

        return dataset.map(encode_function, batched=True,
                           #num_proc=num_proc if not self.streaming else 1,
                           #cache_file_names=self.cache_dir
                           )

    @property
    def encoded_dataset(self) -> DatasetDict:
        if not self.ignore_existing and os.path.exists(self.encoded_path):
            # Load the pre-encoded dataset
            encoded_dataset = load_dataset(self.encoded_path, 
                                           streaming=self.streaming,
                                           num_proc=16 if not self.streaming else 1,
                                           cache_dir=self.cache_dir)
        else:
            #assert False
            # Load the raw data and encode it
            dataset = self.load_data()
            tokenizer = self.initialize_tokenizer()
            encoded_dataset = self.encode_data(dataset, tokenizer)

            #if self.encoded_dataset is not None:
            # Save the encoded dataset to disk for future use
            #    encoded_dataset.save_to_disk(self.encoded_path)

        return encoded_dataset

    @classmethod
    def from_tiny_stories(cls,
                          train_text: Optional[str] = None,
                          val_text: Optional[str] = None,
                          encoded_path: Optional[str] = None,
#train_text="/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/TinyStories-train.txt",
#val_text="/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/TinyStories-valid.txt",
#encoded_path="/home/morgan/Projects/llm_poc/datasets/roneneldan/TinyStories/tiny_stories.hf",
                          **kws
                          ):

        default_train = os.path.join(DATASET_DIR, "roneneldan", "TinyStories", "TinyStories-train.txt")
        train_text = default_train if train_text is None else train_text

        default_val = os.path.join(DATASET_DIR, "roneneldan", "TinyStories", "TinyStories-valid.txt")
        val_text = default_val if val_text is None else val_text

        default_encoded_path = os.path.join(DATASET_DIR, "roneneldan", "TinyStories", "enc_tiny_stories.hf")
        encoded_path = default_encoded_path if encoded_path is None else encoded_path

        dataset = cls(
            train_text=train_text,
            val_text=val_text,
            encoded_path=encoded_path,
            **kws
        )
        return dataset

    @property
    def orig_encoded_dataset(self):
        dataset_dict = self.load_data()
        tokenizer = self.initialize_tokenizer()
        encoded_dataset = self.encode_data(dataset_dict, tokenizer)
        return encoded_dataset


@dataclass
class HFGPTCausalTrain(Serializable):
    model_name: str = "gpt2"
    train_step_stride: int = 1000
    context_length: int = 64
    num_train_epochs: int = 5
    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.1
    warmup_steps: int = 1_000
    logging_steps: int = 1000
    eval_steps: int = 1000
    save_steps: int = 1000
    gradient_accumulation_steps: int = 8
    max_steps: int = 100_000
    output_dir: str = ".pretraining_gpt2"

    model_: Any = field(init=False, default=None)
    train_dataset_: Any = field(init=False, default=None)
    test_dataset_: Any = field(init=False, default=None)

    @cached_property
    def dataset_conf(self):
        return TextDataset.from_tiny_stories(ignore_existing=True,
                                             context_length=self.context_length)

    @cached_property
    def tokenizer(self):
        tokenizer = self.dataset_conf.initialize_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @cached_property
    def hf_model_config(self):
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(self.tokenizer),
            #n_positions=1024,
            #n_positions=64,
            n_positions=128,
            #n_embd=768,
            n_embd=128,
            n_layer=4,
            n_head=4,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=False,
            n_ctx=self.context_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return config

    @cached_property
    def hf_training_args(self):
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,

            logging_strategy="steps",
            logging_steps=self.train_step_stride,

            evaluation_strategy="steps",
            eval_strategy=self.train_step_stride,

            save_strategy="steps",
            save_steps=self.train_step_stride,

            #eval_steps=5_000,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            learning_rate=self.learning_rate,
            #save_steps=2_000,
            #fp16=True,
            max_steps=self.max_steps,
            #use_cpu=True
            # defaults to 'all'
            report_to='mlflow',
        )

    @cached_property
    def data_collator(self):
        from transformers import DataCollatorForLanguageModeling
        return DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def run(self):
        from transformers import pipeline
        from transformers import Trainer, TrainingArguments


        model_config = self.hf_model_config
        self.model = GPT2LMHeadModel(model_config)

        raw_dataset = self.dataset_conf.encoded_dataset
        self.train_dataset_ = raw_dataset["train"]
        self.test_dataset_ = raw_dataset["test"]

        data_collator = self.data_collator
        args = self.hf_training_args

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=self.train_dataset_,
            eval_dataset=self.test_dataset_,
        )

        train_output = trainer.train()

        # Evaluate the model on the test_dataset_
        eval_results = trainer.evaluate()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer,
            device=device
        )

        txt = """Once upon a time"""
        output = pipe(txt)
        print(output)



if __name__ == """__main__""":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(HFGPTCausalTrain, dest="options")
    args = parser.parse_args()
    print(args)
    options: HFGPTCausalTrain = args.options
    options.run()
