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
# Simplified Config #
####################
MODEL_NAME = "Salesforce/codet5-base"
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
MASK_TOKENS = [f"<extra_id_{i}>" for i in range(100)]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


########################
# 1. Verified Tokenizer #
########################
def get_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(SPECIAL_TOKENS + MASK_TOKENS, special_tokens=True)
    return tokenizer


########################
# 2. Safe Map-style Dataset #
########################
class CodeDenoisingDataset(Dataset):
    def __init__(self, file_type: str, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        self.samples = self._load_and_validate_samples(file_type)

    def _load_and_validate_samples(self, file_type: str) -> list:
        """Load and validate all samples before training"""
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

        # Preprocess all samples upfront
        processed = []
        for lang, code in samples:
            result = self._corrupt_sample(lang, code)
            if result["input_text"] and result["target_text"]:
                processed.append(result)
        return processed

    def _corrupt_sample(self, lang_tag: str, code_str: str) -> dict:
        """Simplified span corruption with validation"""
        full_text = f"{lang_tag} {code_str}"
        tokens = self.tokenizer.tokenize(full_text)

        protected = {i for i, tok in enumerate(tokens)
                     if tok in SPECIAL_TOKENS or tok == lang_tag}

        corrupted = []
        available_indices = [i for i in range(len(tokens)) if i not in protected]
        np.random.shuffle(available_indices)

        mask_budget = int(len(available_indices) * MASK_RATIO)
        while mask_budget > 0 and available_indices:
            span_len = min(np.random.randint(SPAN_MIN, SPAN_MAX + 1), mask_budget)
            span = available_indices[:span_len]
            del available_indices[:span_len]

            if not span:
                break

            corrupted.append(f"<extra_id_{len(corrupted)}>")
            mask_budget -= len(span)

        return {
            "input_text": " ".join(corrupted) if corrupted else "<no_mask>",
            "target_text": full_text
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


########################
# 3. Simplified Training #
########################
def main():
    tokenizer = get_tokenizer()
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets upfront with validation
    train_dataset = CodeDenoisingDataset("train", tokenizer)
    val_dataset = CodeDenoisingDataset("val", tokenizer)

    # Training arguments (simplified)
    training_args = TrainingArguments(
        output_dir="/kaggle/working/output",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        fp16=True,
        logging_steps=100,
        save_steps=500,
        remove_unused_columns=False,
        report_to="wandb"

    )

    # Collator with validation
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

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()
    trainer.save_model("/kaggle/working/final_model")


if __name__ == "__main__":
    main()