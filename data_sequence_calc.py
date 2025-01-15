import json
from transformers import T5TokenizerFast
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

PREFIX = "original_"

# Load the data
data_path = './processed_pairs.json'
with open(data_path, 'r') as f:
    data = json.load(f)

# Initialize the tokenizer
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
special_tokens = {
    'additional_special_tokens': ['<JAVA>', '<PYTHON>', '<START>', '<END>']
}
tokenizer.add_special_tokens(special_tokens)

# Calculate sequence lengths and collect all tokens
sequence_lengths = []
java_lengths = []
python_lengths = []
all_tokens = []

for pair in data:
    java_tokens = tokenizer.encode(pair["java_processed"], add_special_tokens=True)
    python_tokens = tokenizer.encode(pair["python_processed"], add_special_tokens=True)
    java_lengths.append(len(java_tokens))
    python_lengths.append(len(python_tokens))
    sequence_lengths.append(len(java_tokens) + len(python_tokens))
    all_tokens.extend(java_tokens)
    all_tokens.extend(python_tokens)

# Count token frequencies
token_counter = Counter(all_tokens)
most_common_tokens = token_counter.most_common(100)

# Save token frequencies to file
with open(f'{PREFIX}_token_frequencies.txt', 'w') as f:
    f.write("Token ID | Token | Frequency\n")
    f.write("-" * 40 + "\n")
    for token_id, freq in most_common_tokens:
        token = tokenizer.decode([token_id])
        f.write(f"{token_id:8d} | {token:20s} | {freq:d}\n")

# Create box-and-whisker plots
plt.figure(figsize=(10, 6))
plt.boxplot(sequence_lengths, vert=False, patch_artist=True, boxprops=dict(facecolor='blue', color='black'),
            whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='red'))
plt.title('Box-and-Whisker Plot: Total Sequence Lengths')
plt.xlabel('Sequence Length')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig(f'{PREFIX}_sequence_length_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(java_lengths, vert=False, patch_artist=True, boxprops=dict(facecolor='green', color='black'),
            whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='red'))
plt.title('Box-and-Whisker Plot: Java Token Lengths')
plt.xlabel('Token Length')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig(f'{PREFIX}_java_token_length_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(python_lengths, vert=False, patch_artist=True, boxprops=dict(facecolor='orange', color='black'),
            whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='red'))
plt.title('Box-and-Whisker Plot: Python Token Lengths')
plt.xlabel('Token Length')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig(f'{PREFIX}_python_token_length_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()