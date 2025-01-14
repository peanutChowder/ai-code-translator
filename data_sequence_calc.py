import json
from transformers import T5TokenizerFast
import numpy as np
import matplotlib.pyplot as plt

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

# Filter out outliers beyond the 95th percentile
lower_bound = np.percentile(sequence_lengths, 5)
upper_bound = np.percentile(sequence_lengths, 95)
filtered_lengths = [length for length in sequence_lengths if lower_bound <= length <= upper_bound]

# Create a histogram for filtered data
plt.figure(figsize=(10, 6))
plt.hist(filtered_lengths, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Sequence Lengths (Excluding Outliers)')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate to indicate outlier treatment
plt.text(max(filtered_lengths) * 0.8, plt.ylim()[1] * 0.9,
         f"Outliers (outside {lower_bound:.1f}-{upper_bound:.1f}) excluded",
         fontsize=10, color='red')

# Save the plot before showing it
plt.savefig('sequence_length_histogram.png', dpi=300, bbox_inches='tight')

plt.show()
