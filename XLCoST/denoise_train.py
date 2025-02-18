import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

####################
# Config #
####################
MODEL_NAME = "Salesforce/codet5-small"
DATA_PATH = "/kaggle/input/xlcost-detokenized-pythonflattened"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 4
ACCUM_STEPS = 2
EPOCHS = 3
MASK_RATIO = 0.15
SPAN_MIN = 3
SPAN_MAX = 5
SEED = 42

# Special tokens and initialization
LANG_TAGS = ["<JAVA>", "<PYTHON>"]
SPECIAL_TOKENS = ["NEW_LINE", "INDENT", "DEDENT"] + LANG_TAGS
PROTECTED_TOKENS = set(LANG_TAGS + ["NEW_LINE", "INDENT", "DEDENT"])
MASK_TOKENS = [f"<extra_id_{i}>" for i in range(100)]  # Part of T5's existing vocab

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


########################
# 1. Tokenizer #
########################
def get_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    return tokenizer


########################
# 2. Dataset #
########################
class CodeDenoisingDataset(Dataset):
    def __init__(self, file_type: str, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        self.raw_samples = self._load_samples(file_type)

    def _load_samples(self, file_type: str) -> list:
        file_path = f"{DATA_PATH}/{file_type}.txt"
        samples = []
        current_lang = None
        current_lines = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line in LANG_TAGS:
                    if current_lang and current_lines:
                        samples.append((current_lang, " ".join(current_lines)))
                    current_lang = line
                    current_lines = []
                elif line:
                    current_lines.append(line)
            if current_lang and current_lines:
                samples.append((current_lang, " ".join(current_lines)))
        return samples

    def _corrupt_sample(self, lang_tag: str, code_str: str) -> dict:
        full_text = f"{lang_tag} {code_str}"
        words = full_text.split()

        masked_tokens = []
        target_spans = []
        mask_count = 0
        i = 0

        while i < len(words) and mask_count < len(MASK_TOKENS):
            # Skip protected tokens
            if words[i] in PROTECTED_TOKENS:
                masked_tokens.append(words[i])
                i += 1
                continue

            if random.random() < MASK_RATIO:
                span_len = np.random.randint(SPAN_MIN, SPAN_MAX + 1)
                span = []

                # Collect non-protected tokens for span
                j = i
                while j < len(words) and len(span) < span_len:
                    if words[j] not in PROTECTED_TOKENS:
                        span.append(words[j])
                    j += 1

                if len(span) == 0:
                    i += 1
                    continue

                masked_tokens.append(MASK_TOKENS[mask_count])
                target_spans.append(f"{MASK_TOKENS[mask_count]} {' '.join(span)}")
                mask_count += 1
                i = j  # Move past the span
            else:
                masked_tokens.append(words[i])
                i += 1

        # Add remaining words
        masked_tokens.extend(words[i:])

        return {
            "input_text": " ".join(masked_tokens),
            "target_text": " ".join(target_spans) if target_spans else full_text
        }

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        lang, code = self.raw_samples[idx]
        return self._corrupt_sample(lang, code)


########################
# 3. Training #
########################
def main():
    tokenizer = get_tokenizer()
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = CodeDenoisingDataset("train", tokenizer)
    val_dataset = CodeDenoisingDataset("val", tokenizer)

    training_args = TrainingArguments(
        output_dir="/kaggle/working/output",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        fp16=True,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb"
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model("/kaggle/working/final_model")


if __name__ == "__main__":
    main()