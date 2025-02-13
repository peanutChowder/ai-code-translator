#!/usr/bin/env python3
import sys
import json
import re
from transformers import RobertaTokenizer

########################
# 0) Set up CodeT5 Tokenizer
########################
MODEL_NAME = "Salesforce/codet5-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
PUNCT_REGEX = re.compile(r'^[\[\]{}().;,+\-*/<>=&|^%!?:@#~]+$')
SPECIAL_PLACEHOLDERS = {"▁"}


########################
# 1) Python Flattening Logic
########################

def flatten_python_code(input_code):
    """Convert Python code to single-line format with explicit tokens"""
    stack = [0]  # Indentation stack
    processed_lines = []

    for line in input_code.split('\n'):
        # Process indentation
        stripped = line.lstrip()
        leading = line[:len(line) - len(stripped)].replace('\t', '    ')
        current_indent = len(leading)

        # Generate INDENT/DEDENT tokens
        indent_tokens = []
        while current_indent < stack[-1]:
            stack.pop()
            indent_tokens.append('DEDENT')
        if current_indent > stack[-1]:
            stack.append(current_indent)
            indent_tokens.append('INDENT')

        # Build line components
        line_parts = []
        if indent_tokens:
            line_parts.extend(indent_tokens)
        if stripped:
            line_parts.append(stripped.replace('\n', ' '))  # Remove any internal newlines

        processed_lines.append(' '.join(line_parts))

    # Close remaining blocks
    while len(stack) > 1:
        stack.pop()
        processed_lines.append('DEDENT')

    # Join all lines with NEW_LINE tokens
    return ' NEW_LINE '.join(filter(None, processed_lines))


########################
# 2) Untokenization
########################

def untokenize_code(tokens):
    """Convert token sequence back to natural code string"""
    filtered = []
    for t in tokens:
        if t == "NEW_LINE":
            filtered.append("\n")
        elif t in SPECIAL_PLACEHOLDERS:
            continue
        else:
            filtered.append(t)

    # Merge dots with neighbors
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

    # Build final code string
    output_parts = []
    for tok in merged:
        if tok == "\n":
            output_parts.append("\n")
        elif PUNCT_REGEX.match(tok):
            if output_parts and not output_parts[-1].endswith("\n"):
                output_parts[-1] += tok
            else:
                output_parts.append(tok)
        else:
            if output_parts and not (output_parts[-1].endswith(" ") or output_parts[-1].endswith("\n")):
                output_parts[-1] += " "
            output_parts.append(tok)

    return ''.join(output_parts).strip("\n")


########################
# 3) Load & Filter with Subwords
########################

def load_and_filter_with_subwords(input_jsonl, field_name, language_tag, lengths_list):
    """Load data with Python flattening"""
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

            code_str = untokenize_code(tokens)

            # Apply Python flattening
            if language_tag == "<PYTHON>":
                code_str = flatten_python_code(code_str)
                # Ensure no actual newlines remain
                code_str = code_str.replace('\n', ' ')

            # Tokenize with CodeT5
            subword_ids = tokenizer.encode(code_str, add_special_tokens=True)
            subword_len = len(subword_ids)

            if subword_len > 512:
                continue

            lengths_list.append(subword_len)
            results.append((language_tag, code_str))

    return results


########################
# 4) Interleave & Write
########################

def interleave_and_write(java_data, python_data, output_path):
    """Interleave Java and Python samples"""
    max_len = max(len(java_data), len(python_data))
    with open(output_path, 'w', encoding='utf-8') as fout:
        idx_j = 0
        idx_p = 0
        for _ in range(max_len):
            if idx_j < len(java_data):
                lang_tag, code_str = java_data[idx_j]
                fout.write(f"{lang_tag}\n{code_str}\n\n")
                idx_j += 1
            if idx_p < len(python_data):
                lang_tag, code_str = python_data[idx_p]
                fout.write(f"{lang_tag}\n{code_str}\n\n")
                idx_p += 1


########################
# 5) Histogram Reporting
########################

def print_histogram(lengths, lang_name):
    """Print subword length distribution"""
    bins = [0, 128, 256, 512, 1024, 2048, 4096, 9999999]
    bin_labels = [
        "0-128", "129-256", "257-512", "513-1024",
        "1025-2048", "2049-4096", "4097+"
    ]
    counts = [0] * (len(bins) - 1)

    for length in lengths:
        for i in range(len(bins) - 1):
            if bins[i] <= length <= bins[i + 1]:
                counts[i] += 1
                break

    total = len(lengths)
    print(f"\nHistogram of CodeT5 subword token lengths for {lang_name}:")
    for label, c in zip(bin_labels, counts):
        perc = (c / total * 100) if total else 0
        print(f"  {label:>9}: {c} snippets ({perc:.1f}%)")
    print(f"  Total {lang_name} snippets: {total}\n")


########################
# 6) Main Execution
########################

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python filter_by_subwords.py <java_jsonl> <python_jsonl> <output_txt> <field_name>")
        sys.exit(1)

    java_jsonl = sys.argv[1]
    python_jsonl = sys.argv[2]
    output_txt = sys.argv[3]
    field_name = sys.argv[4]

    java_subword_lengths = []
    python_subword_lengths = []

    java_data = load_and_filter_with_subwords(
        java_jsonl, field_name, "<JAVA>", java_subword_lengths
    )
    python_data = load_and_filter_with_subwords(
        python_jsonl, field_name, "<PYTHON>", python_subword_lengths
    )

    interleave_and_write(java_data, python_data, output_txt)

    print_histogram(java_subword_lengths, "Java")
    print_histogram(python_subword_lengths, "Python")