# Kaggle notebook setup
# !pip
# install
# transformers
# datasets
# evaluate

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split

# Check GPU availability
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))


class CodeTranslationDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.examples = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Add special tokens that match preprocessing
        special_tokens = {
            'additional_special_tokens': [
                '<JAVA>',
                '<PYTHON>',
                '<START>',
                '<END>'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Get source and target code - these already have special tokens from preprocessing
        java_code = example["java_processed"]  # Already has <JAVA> prefix
        python_code = example["python_processed"]  # Already has <PYTHON> prefix

        # Tokenize inputs
        model_inputs = self.tokenizer(
            java_code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                python_code,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        # Convert label padding to -100 for PyTorch loss calculation
        labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }


def compute_metrics(eval_pred):
    """Custom metrics computation"""
    predictions, labels = eval_pred
    # Get the tokenizer
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    special_tokens = {
        'additional_special_tokens': [
            '<JAVA>',
            '<PYTHON>',
            '<START>',
            '<END>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 with pad token id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU score
    bleu = evaluate.load("bleu")
    bleu_output = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    # Add code compilation check (simplified metric)
    def is_valid_python(code):
        try:
            compile(code, '<string>', 'exec')
            return True
        except:
            return False

    valid_count = sum(1 for pred in decoded_preds if is_valid_python(pred))
    compilation_rate = valid_count / len(decoded_preds)

    return {
        "bleu": bleu_output["bleu"],
        "compilation_rate": compilation_rate
    }


def main():
    # Initialize model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # Add special tokens to model and tokenizer
    special_tokens = {
        'additional_special_tokens': [
            '<JAVA>',
            '<PYTHON>',
            '<START>',
            '<END>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Load data
    print("Loading datasets...")
    with open('/kaggle/input/java-python-processed/processed_pairs.json', 'r') as f:
        data = json.load(f)


    print(f"Loaded {len(data)} total examples")

    temp_train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(temp_train_data, test_size=0.1/0.9, random_state=42)

    # Create datasets
    train_dataset = CodeTranslationDataset(train_data, tokenizer)
    val_dataset = CodeTranslationDataset(val_data, tokenizer)
    test_dataset = CodeTranslationDataset(test_data, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/kaggle/working/java-python-translator",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size = 8 * 4 = 32
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="/kaggle/working/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=True,  # Mixed precision training
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model("/kaggle/working/final-model")
    tokenizer.save_pretrained("/kaggle/working/final-model")
    print("Training completed and model saved!")

    print("Evaluating on the test set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test results:", test_results)


if __name__ == "__main__":
    main()