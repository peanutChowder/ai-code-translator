import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import T5TokenizerFast
from tqdm import tqdm
import time


class CodePreprocessor:
    def __init__(self):
        print("Initializing CodePreprocessor...")
        start_time = time.time()

        # Initialize tokenizer
        try:
            self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
            print("Successfully loaded T5 tokenizer")
        except Exception as e:
            print(f"ERROR: Failed to load tokenizer: {str(e)}")
            raise

        # Special tokens for our task
        self.special_tokens = {
            'java_start': '<JAVA>',
            'python_start': '<PYTHON>',
            'start': '<START>',
            'end': '<END>'
        }

        # Add special tokens to tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(self.special_tokens.values())})
        print(f"Initialization completed in {time.time() - start_time:.2f} seconds")

        # Initialize counters for statistics
        self.stats = {
            'total_pairs_processed': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'java_comment_removals': 0,
            'python_comment_removals': 0
        }

    def remove_comments(self, code: str, is_java: bool) -> str:
        original_length = len(code)

        if is_java:
            code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
            if len(code) != original_length:
                self.stats['java_comment_removals'] += 1
        else:
            code = re.sub(r'#.*?\n|\'\'\'.*?\'\'\'|""".*?"""', '', code, flags=re.DOTALL)
            if len(code) != original_length:
                self.stats['python_comment_removals'] += 1

        return code

    def clean_whitespace(self, code: str) -> str:
        # Normalize newlines
        code = code.replace('\r\n', '\n')
        # Remove trailing whitespace
        code = '\n'.join(line.rstrip() for line in code.split('\n'))
        # Remove multiple blank lines
        code = re.sub(r'\n\s*\n', '\n\n', code)
        return code.strip()

    def standardize_java(self, code: str) -> str:
        """Standardize Java code formatting"""
        # Replace class name with a standard name
        code = re.sub(r'public\s+class\s+\w+', 'class Solution', code)
        code = re.sub(r'class\s+\w+', 'class Solution', code)

        # Standardize main method
        code = re.sub(r'public\s+static\s+void\s+main\s*\(\s*String\s*\[\s*\]\s*\w+\s*\)',
                      'public static void main(String[] args)', code)

        # Handle potential encoding issues
        code = code.encode('ascii', 'ignore').decode('ascii')

        return code

    def standardize_python(self, code: str) -> str:
        """Standardize Python code formatting"""
        # Convert tabs to spaces (4 spaces per tab)
        code = code.replace('\t', '    ')

        # Ensure consistent indentation (4 spaces)
        lines = code.split('\n')
        processed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # If line is not empty
                indent_level = (len(line) - len(stripped)) // 4
                processed_lines.append('    ' * indent_level + stripped)
            else:
                processed_lines.append('')

        # Handle potential encoding issues
        code = '\n'.join(processed_lines)
        code = code.encode('ascii', 'ignore').decode('ascii')

        return code

    def preprocess_pair(self, java_code: str, python_code: str, pair_id: str = None) -> Tuple[str, str]:
        self.stats['total_pairs_processed'] += 1

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

            self.stats['successful_pairs'] += 1
            return java_processed, python_processed

        except Exception as e:
            self.stats['failed_pairs'] += 1
            print(f"WARNING: Failed to process pair {pair_id}: {str(e)}")
            return None, None


def process_dataset(input_path: str, output_path: str, batch_size: int = 1000):
    """Process the dataset in batches to manage memory"""
    print(f"Starting dataset processing - Input: {input_path}, Output: {output_path}")
    start_time = time.time()

    preprocessor = CodePreprocessor()

    try:
        # Read input data
        print("Reading input dataset...")
        with open(input_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} pairs from input dataset")

        processed_pairs = []
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes

        # Process in batches with progress bar
        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data[i:i + batch_size]
            batch_processed = []

            for item in batch:
                pair_id = f"{item['problem_id']}_{item['java_submission_id']}_{item['python_submission_id']}"
                java_proc, python_proc = preprocessor.preprocess_pair(
                    item['java_code'],
                    item['python_code'],
                    pair_id
                )

                if java_proc is not None and python_proc is not None:
                    batch_processed.append({
                        'problem_id': item['problem_id'],
                        'java_submission_id': item['java_submission_id'],
                        'python_submission_id': item['python_submission_id'],
                        'java_processed': java_proc,
                        'python_processed': python_proc
                    })

            processed_pairs.extend(batch_processed)

            # Save progress based on time interval
            current_time = time.time()
            if current_time - last_save_time > save_interval:
                print(f"Periodic save: {len(processed_pairs)} pairs processed so far")
                with open(output_path, 'w') as f:
                    json.dump(processed_pairs, f)
                last_save_time = current_time

        # Final save
        with open(output_path, 'w') as f:
            json.dump(processed_pairs, f)

        # Print final statistics
        total_time = time.time() - start_time
        print("\nProcessing completed:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total pairs processed: {preprocessor.stats['total_pairs_processed']}")
        print(f"Successful pairs: {preprocessor.stats['successful_pairs']}")
        print(f"Failed pairs: {preprocessor.stats['failed_pairs']}")
        print(f"Java comment removals: {preprocessor.stats['java_comment_removals']}")
        print(f"Python comment removals: {preprocessor.stats['python_comment_removals']}")
        print(
            f"Average processing time per pair: {total_time / preprocessor.stats['total_pairs_processed']:.3f} seconds")
        print(
            f"Success rate: {(preprocessor.stats['successful_pairs'] / preprocessor.stats['total_pairs_processed']) * 100:.1f}%")

    except Exception as e:
        print(f"ERROR: Critical error during dataset processing: {str(e)}")
        raise


if __name__ == "__main__":
    input_path = "/kaggle/input/java-python-unprocessed/java_python_pairs_final.json"
    output_path = "/kaggle/working/processed_pairs.json"

    process_dataset(input_path, output_path)