import os
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from typing import List, Tuple, Iterator

####################
# Config
####################
MODEL_NAME = "Salesforce/codet5-base"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 1
MASK_RATIO = 0.15
SPAN_MIN = 3
SPAN_MAX = 5
SHUFFLE_BUFFER_SIZE = 1000  # For in-memory shuffling
SEED = 42

# Special tokens
LANG_TAGS = ["<JAVA>", "<PYTHON>"]
SPECIAL_TOKENS = ["NEW_LINE", "INDENT", "DEDENT"] + LANG_TAGS
MASK_TOKENS = [f"<extra_id_{i}>" for i in range(100)]

random.seed(SEED)
torch.manual_seed(SEED)


########################
# 1) Enhanced Tokenizer
########################
def get_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(SPECIAL_TOKENS + MASK_TOKENS, special_tokens=True)

    # Verify token integrity
    for tok in SPECIAL_TOKENS:
        assert len(tokenizer.tokenize(tok)) == 1, f"Token {tok} gets split!"

    return tokenizer


########################
# 2) Streaming Dataset
########################
class CodeDenoisingIterableDataset(IterableDataset):
    def __init__(
            self,
            file_path: str,
            tokenizer: RobertaTokenizer,
            mask_ratio: float = 0.15,
            span_min: int = 3,
            span_max: int = 5,
            shuffle: bool = False,
            buffer_size: int = 1000,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.span_min = span_min
        self.span_max = span_max
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()

        # Handle multi-worker splitting
        if worker_info is not None:
            raise NotImplementedError("Multi-worker support needs file partitioning")

        # Create base generator
        sample_gen = self._sample_generator()

        # Add shuffle buffer if needed
        if self.shuffle:
            sample_gen = self._shuffle_generator(sample_gen)

        return sample_gen

    def _sample_generator(self) -> Iterator[dict]:
        current_lang = None
        current_lines = []

        with open(self.file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()

                if line in LANG_TAGS:
                    if current_lang is not None:
                        # Yield accumulated sample
                        code = " ".join(current_lines).strip()
                        if code:
                            yield self._process_sample(current_lang, code)
                    current_lang = line
                    current_lines = []
                else:
                    if line:
                        current_lines.append(line)

            # Yield final sample
            if current_lang is not None and current_lines:
                code = " ".join(current_lines).strip()
                yield self._process_sample(current_lang, code)

    def _shuffle_generator(self, generator: Iterator) -> Iterator:
        buffer = []
        for sample in generator:
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def _process_sample(self, lang_tag: str, code_str: str) -> dict:
        """Apply span corruption to a single sample"""
        # Tokenize with language tag
        full_text = f"{lang_tag} {code_str}"
        tokens = self.tokenizer.tokenize(full_text)

        # Find protected ranges (language tag + special tokens)
        protected = set()
        for i, tok in enumerate(tokens):
            if tok == lang_tag or tok in SPECIAL_TOKENS:
                protected.add(i)

        # Apply span masking
        corrupted_tokens = self._apply_masking(tokens, protected)

        return {
            "input_text": " ".join(corrupted_tokens),
            "target_text": full_text
        }

    def _apply_masking(self, tokens: List[str], protected: set) -> List[str]:
        """Core masking logic adapted for streaming"""
        n = len(tokens)
        num_to_mask = int((n - len(protected)) * self.mask_ratio)
        if num_to_mask < 1:
            return tokens.copy()

        candidates = [i for i in range(n) if i not in protected]
        random.shuffle(candidates)

        masked_indices = set()
        placeholders = []
        current_span_id = 0
        ptr = 0

        while ptr < len(candidates) and len(masked_indices) < num_to_mask:
            span_len = random.randint(self.span_min, self.span_max)
            start = candidates[ptr]
            end = min(start + span_len, n)

            # Collect span indices
            span_indices = []
            for i in range(start, end):
                if i not in protected and i not in masked_indices:
                    span_indices.append(i)

            if span_indices:
                masked_indices.update(span_indices)
                placeholders.append((
                    min(span_indices),
                    max(span_indices),
                    current_span_id
                ))
                current_span_id += 1
                ptr += span_len
            else:
                ptr += 1

        # Build corrupted sequence
        corrupted = []
        last_idx = 0
        for st, en, sid in sorted(placeholders, key=lambda x: x[0]):
            corrupted.extend(tokens[last_idx:st])
            corrupted.append(f"<extra_id_{sid}>")
            last_idx = en + 1
        corrupted.extend(tokens[last_idx:])

        return corrupted


########################
# 3) Training Setup
########################
def main():
    # Initialize components
    tokenizer = get_tokenizer()
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # Create datasets
    train_dataset = CodeDenoisingIterableDataset(
        file_path="/kaggle/input/dataset/train.txt",
        tokenizer=tokenizer,
        mask_ratio=MASK_RATIO,
        span_min=SPAN_MIN,
        span_max=SPAN_MAX,
        shuffle=True,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    val_dataset = CodeDenoisingIterableDataset(
        file_path="/kaggle/input/dataset/val.txt",
        tokenizer=tokenizer,
        mask_ratio=MASK_RATIO,
        span_min=SPAN_MIN,
        span_max=SPAN_MAX,
        shuffle=False,
    )

    # Create DataLoader with dynamic batching
    def collate_fn(batch):
        inputs = [item["input_text"] for item in batch]
        targets = [item["target_text"] for item in batch]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SOURCE_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=MAX_TARGET_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

        model_inputs["labels"] = labels
        return model_inputs

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")


if __name__ == "__main__":
    main()