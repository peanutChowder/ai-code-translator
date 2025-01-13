import json
from transformers import T5TokenizerFast
import numpy as np

# Load the data
data_path = './processed_pairs.json'  # Replace with the correct path
with open(data_path, 'r') as f:
    data = json.load(f)

# Initialize the tokenizer
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
special_tokens = {
    'additional_special_tokens': ['<JAVA>', '<PYTHON>', '<START>', '<END>']
}
tokenizer.add_special_tokens(special_tokens)

# Calculate sequence lengths
sequence_lengths = []
for pair in data:
    java_tokens = tokenizer.encode(pair["java_processed"], add_special_tokens=True)
    python_tokens = tokenizer.encode(pair["python_processed"], add_special_tokens=True)
    sequence_lengths.append(len(java_tokens) + len(python_tokens))

# Compute statistics
avg_length = np.mean(sequence_lengths)
min_length = np.min(sequence_lengths)
max_length = np.max(sequence_lengths)
median_length = np.median(sequence_lengths)

# Print results
print(f"Average Sequence Length: {avg_length}")
print(f"Minimum Sequence Length: {min_length}")
print(f"Maximum Sequence Length: {max_length}")
print(f"Median Sequence Length: {median_length}")
