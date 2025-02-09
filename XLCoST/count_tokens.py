#!/usr/bin/env python3
import sys
import re
from transformers import RobertaTokenizer

####################
# Configure CodeT5
####################
MODEL_NAME = "Salesforce/codet5-small"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

def parse_snippets_from_txt(input_txt_path):
    """
    Reads a text file with interleaved <JAVA> and <PYTHON> markers,
    collects each snippet's lines until the next marker or EOF.
    Returns a list of (lang, snippet_string).
    lang is either "JAVA" or "PYTHON".
    """
    snippets = []
    current_lang = None
    current_lines = []

    with open(input_txt_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip('\n')
            # Check if line is <JAVA> or <PYTHON>
            if line == "<JAVA>" or line == "<PYTHON>":
                # If we already have some snippet lines, store them
                if current_lang is not None and current_lines:
                    snippet_str = "\n".join(current_lines).strip()
                    snippets.append((current_lang, snippet_str))
                # Reset for new snippet
                current_lang = line.strip("<>").upper()  # "JAVA" or "PYTHON"
                current_lines = []
            else:
                # It's part of the snippet
                current_lines.append(line)

        # End of file: if there's a snippet in progress, store it
        if current_lang is not None and current_lines:
            snippet_str = "\n".join(current_lines).strip()
            snippets.append((current_lang, snippet_str))

    return snippets

def count_subwords(snippets):
    """
    For a list of (lang, code_str) pairs, returns:
      - java_lengths: list of subword token lengths for Java
      - python_lengths: list of subword token lengths for Python
      - overlong_samples: list of (lang, code_str) pairs that exceed 512 tokens
    """
    java_lengths = []
    python_lengths = []
    overlong_samples = []

    for lang, code_str in snippets:
        subword_ids = tokenizer.encode(code_str, add_special_tokens=True)
        length = len(subword_ids)

        if length > 512:
            overlong_samples.append((lang, code_str))

        if lang == "JAVA":
            java_lengths.append(length)
        elif lang == "PYTHON":
            python_lengths.append(length)

    return java_lengths, python_lengths, overlong_samples

def print_histogram(lengths, lang_name, bins=None):
    """
    Print a simple histogram of snippet lengths for the given language.
    lengths = list of integer subword counts
    bins = optional bin edges
    """
    if not bins:
        bins = [0, 128, 256, 512, 1024, 2048, 4096, 9999999]
    bin_labels = []
    for i in range(len(bins) - 1):
        start = bins[i] + 1 if i > 0 else bins[i]
        end = bins[i+1]
        bin_labels.append(f"{start}-{end}")

    counts = [0] * (len(bins) - 1)
    for length in lengths:
        for i in range(len(bins)-1):
            if bins[i] <= length <= bins[i+1]:
                counts[i] += 1
                break

    total = len(lengths)
    print(f"\nHistogram of CodeT5 subword token lengths for {lang_name}:")
    for label, c in zip(bin_labels, counts):
        perc = (c / total * 100) if total else 0
        print(f"  {label:>10}: {c} snippets ({perc:.1f}%)")
    print(f"  Total {lang_name} snippets: {total}")

if __name__ == "__main__":
    """
    Usage:
      python count_tokens.py <processed_output.txt>

    The script will:
      1) Parse the <JAVA> / <PYTHON> snippets from 'processed_output.txt'
      2) Tokenize each snippet with CodeT5
      3) Print histograms of snippet subword lengths for Java and Python
      4) If any samples exceed 512 tokens, prompt the user to print up to 5 examples
    """
    if len(sys.argv) < 2:
        print("Usage: python codeT5_count_from_txt.py <processed_output.txt>")
        sys.exit(1)

    input_txt = sys.argv[1]

    # 1) Parse snippets
    snippets = parse_snippets_from_txt(input_txt)
    # 2) Count CodeT5 subwords and detect long samples
    java_lengths, python_lengths, overlong_samples = count_subwords(snippets)
    # 3) Print histograms
    print_histogram(java_lengths, "Java")
    print_histogram(python_lengths, "Python")

    # 4) If overlong samples exist, prompt the user
    if overlong_samples:
        print(f"\nWarning: {len(overlong_samples)} snippets exceed 512 tokens.")
        user_input = input("Would you like to print up to 5 of them? (y/n): ").strip().lower()
        if user_input == "y":
            print("\n=== Overlong Samples (Up to 5) ===")
            for i, (lang, code_str) in enumerate(overlong_samples[:5]):
                print(f"\n[{lang}] Sample {i+1}:\n{code_str}\n")
