#!/usr/bin/env python3
import sys
import json
import re
from transformers import RobertaTokenizer
from collections import Counter

########################
# 0) Set up CodeT5 Tokenizer
########################
MODEL_NAME = "Salesforce/codet5-base"  # or "Salesforce/codet5-small", etc.
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Regex to detect punctuation or bracket-like tokens
PUNCT_REGEX = re.compile(r'^[\[\]{}().;,+\-*/<>=&|^%!?:@#~]+$')

# We remove "▁" if it appears, but do NOT remove "NEW_LINE".
SPECIAL_PLACEHOLDERS = {"▁"}


########################
# 1) Untokenization
########################

def untokenize_code(tokens):
    """
    Generic function for both Java/Python tokens:
      - Convert "NEW_LINE" => "\n"
      - Remove "▁"
      - Merge '.' with neighbors if possible
      - Attach punctuation to preceding chunk, etc.
    """
    # First pass: remove placeholders and convert NEW_LINE
    filtered = []
    for t in tokens:
        if t == "NEW_LINE":
            filtered.append("\n")
        elif t in SPECIAL_PLACEHOLDERS:
            continue
        else:
            filtered.append(t)

    # Second pass: try merging '.' with the next token if it's not punctuation or newline
    merged = []
    i = 0
    while i < len(filtered):
        tok = filtered[i]
        if tok == "." and merged:
            prev = merged[-1]
            nxt = filtered[i + 1] if (i + 1 < len(filtered)) else None
            if nxt and nxt != "\n" and not PUNCT_REGEX.match(nxt):
                merged[-1] = prev + "." + nxt
                i += 2
            else:
                merged[-1] = prev + "."
                i += 1
        else:
            merged.append(tok)
            i += 1

    # Final pass: build the output code string
    output_parts = []
    for tok in merged:
        if tok == "\n":
            output_parts.append("\n")
        elif PUNCT_REGEX.match(tok):
            # punctuation -> attach to previous chunk if possible
            if output_parts:
                if output_parts[-1].endswith("\n"):
                    output_parts.append(tok)
                else:
                    output_parts[-1] += tok
            else:
                output_parts.append(tok)
        else:
            # normal identifier/keyword/number
            if output_parts and not (output_parts[-1].endswith(" ") or output_parts[-1].endswith("\n")):
                output_parts[-1] += " "
            output_parts.append(tok)

    code_str = "".join(output_parts)
    return code_str.strip("\n")


########################
# 2) Load & Filter by CodeT5 subwords
########################

def load_and_filter_with_subwords(input_jsonl, field_name, language_tag, lengths_list):
    """
    Reads a JSONL file (XLCoST style), extracts 'tokens' from `field_name`,
    untokenizes them to get raw code, and counts subword tokens via CodeT5 tokenizer.

    *Only* keeps snippets whose CodeT5 subword length <= 512.

    :param input_jsonl: path to JSONL file
    :param field_name: e.g. "docstring_tokens" or "code_tokens"
    :param language_tag: "<JAVA>" or "<PYTHON>"
    :param lengths_list: a list to store subword lengths for histogram
    :return: a list of (lang_tag, code_str) for the final data
    """
    results = []
    with open(input_jsonl, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            tokens = data.get(field_name, [])
            if not isinstance(tokens, list) or not tokens:
                continue

            # Convert tokens => raw code string
            code_str = untokenize_code(tokens)

            # Now count subwords with CodeT5 tokenizer
            subword_ids = tokenizer.encode(code_str, add_special_tokens=True)
            subword_len = len(subword_ids)

            # Skip if over 512
            if subword_len > 512:
                continue

            # Otherwise keep it
            lengths_list.append(subword_len)
            results.append((language_tag, code_str))

    return results


########################
# 3) Interleave & Write
########################

def interleave_and_write(java_data, python_data, output_path):
    """
    Interleave two lists of (lang_tag, code_str) and write
    them to `output_path`, e.g. <JAVA> snippet, <PYTHON> snippet, etc.
    """
    max_len = max(len(java_data), len(python_data))

    with open(output_path, 'w', encoding='utf-8') as fout:
        idx_j = 0
        idx_p = 0
        for i in range(max_len):
            if idx_j < len(java_data):
                lang_tag, code_str = java_data[idx_j]
                fout.write(f"{lang_tag}\n{code_str}\n\n")
                idx_j += 1

            if idx_p < len(python_data):
                lang_tag, code_str = python_data[idx_p]
                fout.write(f"{lang_tag}\n{code_str}\n\n")
                idx_p += 1


########################
# 4) Print Histograms
########################

def print_histogram(lengths, lang_name):
    """
    Print a simple histogram of snippet subword lengths for the given language.
    bins can be adjusted if needed.
    """
    bins = [0, 128, 256, 512, 1024, 2048, 4096, 9999999]
    bin_labels = [
        "0-128", "129-256", "257-512", "513-1024",
        "1025-2048", "2049-4096", "4097+"
    ]
    counts = [0]*(len(bins)-1)

    for length in lengths:
        for i in range(len(bins)-1):
            if bins[i] <= length <= bins[i+1]:
                counts[i] += 1
                break

    total = len(lengths)
    print(f"\nHistogram of CodeT5 subword token lengths for {lang_name}:")
    for label, c in zip(bin_labels, counts):
        perc = (c / total * 100) if total else 0
        print(f"  {label:>9}: {c} snippets ({perc:.1f}%)")
    print(f"  Total {lang_name} snippets: {total}\n")


########################
# 5) Main
########################

if __name__ == "__main__":
    """
    Example usage:
       python filter_by_subwords.py java.jsonl python.jsonl output.txt docstring_tokens

    This script:
      1) Reads Java from java.jsonl, Python from python.jsonl
      2) Untokenizes each snippet
      3) Tokenizes with CodeT5 -> if subword len > 512, skip
      4) Interleaves remaining samples -> output.txt
      5) Prints histogram of subword lengths
    """
    if len(sys.argv) < 5:
        print("Usage: python filter_by_subwords.py <java_jsonl> <python_jsonl> <output_txt> <field_name>")
        sys.exit(1)

    java_jsonl = sys.argv[1]
    python_jsonl = sys.argv[2]
    output_txt = sys.argv[3]
    field_name = sys.argv[4]

    # Lists for subword lengths
    java_subword_lengths = []
    python_subword_lengths = []

    # 1) Load & filter Java by subwords
    java_data = load_and_filter_with_subwords(java_jsonl, field_name, "<JAVA>", java_subword_lengths)
    # 2) Load & filter Python by subwords
    python_data = load_and_filter_with_subwords(python_jsonl, field_name, "<PYTHON>", python_subword_lengths)

    # 3) Interleave & write
    interleave_and_write(java_data, python_data, output_txt)

    # 4) Print histograms
    print_histogram(java_subword_lengths, "Java")
    print_histogram(python_subword_lengths, "Python")
