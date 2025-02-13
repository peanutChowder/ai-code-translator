#!/usr/bin/env python3
import sys
import re
import random
from transformers import RobertaTokenizer

##############################
# Configure the CodeT5 model
##############################
MODEL_NAME = "Salesforce/codet5-small"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Adjust these parameters for how heavily you mask
MASK_RATIO = 0.15     # 15% of tokens are masked
SPAN_MIN   = 3        # Minimum span size
SPAN_MAX   = 5        # Maximum span size

def parse_snippets_from_file(input_file_path):
    """
    Reads lines from a text file containing
    <JAVA> or <PYTHON> lines, then code lines, then
    possibly another <JAVA> / <PYTHON>, etc.

    Returns a list of (lang, code_string).
      lang is "JAVA" or "PYTHON"
      code_string is the entire snippet (multi-line).
    """
    snippets = []
    current_lang = None
    current_lines = []

    with open(input_file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip('\n')
            # Check if line is exactly <JAVA> or <PYTHON>
            if line == "<JAVA>" or line == "<PYTHON>":
                # If we already have a snippet in progress, finalize it
                if current_lang and current_lines:
                    snippet_str = "\n".join(current_lines).strip()
                    snippets.append((current_lang, snippet_str))
                    current_lines = []
                # Start a new snippet
                current_lang = line.strip("<>").upper()
            else:
                # It's part of the snippet
                current_lines.append(line)

    # End of file: if there's a snippet in progress, finalize it
    if current_lang and current_lines:
        snippet_str = "\n".join(current_lines).strip()
        snippets.append((current_lang, snippet_str))

    return snippets

def mask_code_t5_style(code_str, mask_ratio=0.15, span_min=3, span_max=5):
    """
    Simple T5-style random span corruption.
    1) Tokenize 'code_str'
    2) Randomly select ~mask_ratio of tokens to cover with spans
    3) Replace each span with a unique placeholder (<extra_id_0>, <extra_id_1>, ...)
    4) Return a 'masked_code_str'
    
    For real T5 training, you'd keep (masked_code, original_code) pairs,
    but here we only return the masked code. 
    """
    # 1) Subword tokenize
    tokens = tokenizer.tokenize(code_str)
    n = len(tokens)
    if n == 0:
        return code_str  # nothing to mask

    num_to_mask = int(n * mask_ratio)
    if num_to_mask == 0:
        # no masking
        return code_str

    # We'll pick random token indices to start spans
    all_indices = list(range(n))
    random.shuffle(all_indices)

    masked_indices = set()
    placeholders = []
    current_span_id = 0
    i = 0

    while i < len(all_indices) and len(masked_indices) < num_to_mask:
        span_length = random.randint(span_min, span_max)
        start = all_indices[i]
        i += 1
        # gather up to 'span_length' consecutive tokens from 'start'
        # skipping those already masked
        this_span = []
        for idx in range(start, start + span_length):
            if idx < n and idx not in masked_indices:
                this_span.append(idx)
        if not this_span:
            continue
        for idx in this_span:
            masked_indices.add(idx)
        placeholders.append((min(this_span), max(this_span), current_span_id))
        current_span_id += 1

    # Build corrupted tokens
    corrupted = []
    used_span_ids = set()
    idx = 0
    while idx < n:
        if idx in masked_indices:
            # find which span it belongs to
            for (start_idx, end_idx, sid) in placeholders:
                if start_idx <= idx <= end_idx:
                    if sid not in used_span_ids:
                        corrupted.append(f"<extra_id_{sid}>")
                        used_span_ids.add(sid)
                    # skip entire span
                    idx = end_idx + 1
                    break
        else:
            corrupted.append(tokens[idx])
            idx += 1

    masked_code_str = " ".join(corrupted)
    return masked_code_str

def main():
    """
    Usage:
      python mask_code_snippets.py <input_file> <output_file>
    
    Reads the text from <input_file> with <JAVA>/<PYTHON> markers and code.
    Produces a masked version in <output_file>.
    """
    if len(sys.argv) < 3:
        print("Usage: python mask.py <input_file> <output_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Parse snippets from the input file
    snippets = parse_snippets_from_file(input_file_path)

    # Mask them and write to output
    with open(output_file_path, 'w', encoding='utf-8') as fout:
        for lang, code_str in snippets:
            masked_str = mask_code_t5_style(
                code_str,
                mask_ratio=MASK_RATIO,
                span_min=SPAN_MIN,
                span_max=SPAN_MAX
            )
            # Write the snippet label and masked code
            fout.write(f"<{lang}>\n{masked_str}\n\n")

if __name__ == "__main__":
    main()
