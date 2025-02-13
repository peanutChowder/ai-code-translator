#!/usr/bin/env python3
import argparse
import os
import sys
import re

def count_leading_spaces(line: str) -> int:
    """Count the number of leading spaces in the line."""
    return len(line) - len(line.lstrip(' '))

def clean_java_lines(code_lines):
    """
    Remove Java comments (both single-line // and block comments /* ... */)
    from the provided code lines.
    Note: This is a simple regex-based approach and might not be perfect if comment
    patterns appear in string literals.
    """
    # Join all lines into a single string
    code_text = "\n".join(code_lines)
    # Remove block comments (/* ... */) using DOTALL flag to capture newlines
    code_text = re.sub(r'/\*.*?\*/', '', code_text, flags=re.DOTALL)
    # Remove single-line comments (// ...)
    cleaned_lines = []
    for line in code_text.splitlines():
        # Remove anything after //, if present.
        line = re.sub(r'//.*$', '', line)
        if line.strip():
            cleaned_lines.append(line)
    return cleaned_lines

def flatten_java_code(code_lines):
    """
    Convert multi-line Java code into a single line with minimal spacing,
    mimicking your example style:
        import java.util.*; class GFG{ static ... } ...
    Steps:
      1) Remove comments and strip each line, skipping blank lines.
      2) Join lines with a single space.
      3) Collapse multiple spaces into one.
      4) Remove space before '{' or '}'.
    """
    # Remove comments first
    cleaned_lines = clean_java_lines(code_lines)
    # Strip each line and skip blank lines.
    stripped_lines = [ln.strip() for ln in cleaned_lines if ln.strip()]
    # Join with a single space.
    merged = " ".join(stripped_lines)
    # Collapse multiple spaces into one.
    merged = re.sub(r"\s+", " ", merged)
    # Remove space before any curly brace.
    merged = re.sub(r"\s+([{}])", r"\1", merged)
    return merged

def clean_python_lines(code_lines):
    """
    Remove blank lines and comments from Python code.
    - Skips lines that are empty or that start with '#' (after stripping).
    - For lines with inline comments, removes the comment part.
    """
    cleaned = []
    for raw_line in code_lines:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue  # Skip blank lines.
        if line.lstrip().startswith("#"):
            continue  # Skip full-line comments.
        # Remove inline comments (naively; beware of '#' in strings).
        if "#" in line:
            line = line.split("#", 1)[0]
        line = line.rstrip()
        if line:
            cleaned.append(line)
    return cleaned

def python_lines_with_indent_tokens(code_lines, spaces_per_level=4):
    """
    For Python:
      - Insert INDENT/DEDENT tokens based on changes in leading spaces.
      - Assumes code_lines is already cleaned (no blank lines, no comments).
    """
    final_lines = []
    current_indent = 0
    for line in code_lines:
        new_indent = count_leading_spaces(line)
        stripped_line = line.lstrip(' ')
        if new_indent > current_indent:
            diff = new_indent - current_indent
            levels = diff // spaces_per_level if spaces_per_level else 1
            final_lines.append(("INDENT " * levels) + stripped_line)
        elif new_indent < current_indent:
            diff = current_indent - new_indent
            levels = diff // spaces_per_level if spaces_per_level else 1
            final_lines.append(("DEDENT " * levels) + stripped_line)
        else:
            final_lines.append(stripped_line)
        current_indent = new_indent
    return final_lines

def wrap_as_java(code_lines):
    """
    Process Java code:
      - Remove comments.
      - Flatten the code into one line.
      - Wrap with <JAVA> tags.
    """
    flattened = flatten_java_code(code_lines)
    return f"<JAVA>\n{flattened}"

def wrap_as_python(code_lines, spaces_per_level=4):
    """
    Process Python code:
      1) Clean the lines (remove blank lines and comments).
      2) Insert INDENT/DEDENT tokens based on indentation.
      3) Wrap with <PYTHON> tags.
    """
    cleaned = clean_python_lines(code_lines)
    processed = python_lines_with_indent_tokens(cleaned, spaces_per_level)
    joined = "\n".join(processed)
    return f"<PYTHON>\n{joined}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["java", "python"],
                        help="Which language: 'java' or 'python'.")
    parser.add_argument("--code_file", required=True,
                        help="Path to the file containing raw code.")
    parser.add_argument("--out_file", default="",
                        help="Output file path. If not specified, prints to stdout.")
    parser.add_argument("--spaces_per_level", type=int, default=4,
                        help="Number of spaces corresponding to one indentation level in Python.")
    args = parser.parse_args()

    if not os.path.isfile(args.code_file):
        sys.exit(f"Error: {args.code_file} not found or not a file.")

    with open(args.code_file, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    if args.lang == "java":
        result = wrap_as_java(raw_lines)
    else:  # python
        result = wrap_as_python(raw_lines, args.spaces_per_level)

    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as fout:
            fout.write(result + "\n")
    else:
        print(result)

if __name__ == "__main__":
    main()
