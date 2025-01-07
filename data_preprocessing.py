import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import T5TokenizerFast
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='preprocessing.log'
)


class CodePreprocessor:
    def __init__(self):
        # Initialize tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained('Salesforce/codet5-small')

        # Special tokens for our task
        self.special_tokens = {
            'java_start': '<JAVA>',
            'python_start': '<PYTHON>',
            'start': '<START>',
            'end': '<END>'
        }
        # Add special tokens to tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(self.special_tokens.values())})

    def remove_comments(self, code: str, is_java: bool) -> str:
        if is_java:
            # Remove Java-style comments
            code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
        else:
            # Remove Python-style comments
            code = re.sub(r'#.*?\n|\'\'\'.*?\'\'\'|""".*?"""', '', code, flags=re.DOTALL)
        return code

    def clean_whitespace(self, code: str) -> str:
        # Normalize newlines
        code = code.replace('\r\n', '\n')
        # Remove trailing whitespace
        code = '\n'.join(line.rstrip() for line in code.split('\n'))
        # Remove multiple blank lines
        code = re.sub(r'\n\s*\n', '\n\n', code)
        return code.strip()

    def standardize_java(self, java_code: str) -> str:
        # Simplify class name - replace public class X with class Solution
        java_code = re.sub(r'public\s+class\s+\w+', 'class Solution', java_code)
        java_code = re.sub(r'class\s+\w+', 'class Solution', java_code)
        return java_code

    def standardize_python(self, python_code: str) -> str:
        # Ensure consistent indentation (4 spaces)
        lines = python_code.split('\n')
        processed_lines = []
        for line in lines:
            # Convert any indentation to multiples of 4 spaces
            stripped = line.lstrip()
            indentation = len(line) - len(stripped)
            if indentation > 0:
                spaces = (indentation + 3) // 4 * 4
                processed_lines.append(' ' * spaces + stripped)
            else:
                processed_lines.append(line)
        return '\n'.join(processed_lines)

    def preprocess_pair(self, java_code: str, python_code: str) -> Tuple[str, str]:
        try:
            # Clean Java code
            java_clean = self.remove_comments(java_code, is_java=True)
            java_clean = self.clean_whitespace(java_clean)
            java_clean = self.standardize_java(java_clean)

            # Clean Python code
            python_clean = self.remove_comments(python_code, is_java=False)
            python_clean = self.clean_whitespace(python_clean)
            python_clean = self.standardize_python(python_clean)

            # Add special tokens
            java_processed = f"{self.special_tokens['java_start']}\n{java_clean}"
            python_processed = f"{self.special_tokens['python_start']}\n{python_clean}"

            return java_processed, python_processed

        except Exception as e:
            logging.error(f"Error preprocessing pair: {str(e)}")
            return None, None


def process_dataset(input_path: str, output_path: str, batch_size: int = 1000):
    """Process the dataset in batches to manage memory"""
    preprocessor = CodePreprocessor()

    try:
        # Read input data
        with open(input_path, 'r') as f:
            data = json.load(f)

        processed_pairs = []
        skipped_count = 0

        # Process in batches with progress bar
        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data[i:i + batch_size]
            batch_processed = []

            for item in batch:
                java_proc, python_proc = preprocessor.preprocess_pair(
                    item['java_code'],
                    item['python_code']
                )

                if java_proc is not None and python_proc is not None:
                    batch_processed.append({
                        'problem_id': item['problem_id'],
                        'java_submission_id': item['java_submission_id'],
                        'python_submission_id': item['python_submission_id'],
                        'java_processed': java_proc,
                        'python_processed': python_proc
                    })
                else:
                    skipped_count += 1

            # Write batch to file to save memory
            processed_pairs.extend(batch_processed)

            # Periodically save progress
            if len(processed_pairs) % 5000 == 0:
                with open(output_path, 'w') as f:
                    json.dump(processed_pairs, f)

        # Final save
        with open(output_path, 'w') as f:
            json.dump(processed_pairs, f)

        logging.info(f"Processing completed. Processed {len(processed_pairs)} pairs, skipped {skipped_count} pairs")

    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        raise


if __name__ == "__main__":
    input_path = "training_data.json"
    output_path = "processed_data.json"

    process_dataset(input_path, output_path)