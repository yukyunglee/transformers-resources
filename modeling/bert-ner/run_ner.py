import random
import numpy as np
import pandas as pd

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from utils import tokenize_and_align_labels, compute_metrics

from config import load_config

import logging

"""
Dataset description

DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3453
    })
})
"""
logger = logging.getLogger(__name__)


def run(args):
    # dataset
    datasets = load_dataset(args.dataset_name)
    label_list = datasets["train"].features[f"{args.task}_tags"].feature.names

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    # train
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint, num_labels=len(label_list)
    )

    model_name = args.model_checkpoint.split("/")[-1]
    train_args = TrainingArguments(
        f"{model_name}-finetuned-{args.task}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        # load_best_model_at_end=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"{model_name}-finetuned-{args.task}")

    trainer.evaluate()


if __name__ == "__main__":

    args = load_config()
    run(args)
