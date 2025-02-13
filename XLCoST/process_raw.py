#!/usr/bin/env python3
import argparse
import os
import sys
import re

def count_leading_spaces(line: str) -> int:
    """Count the number of leading spaces in the line."""
    return len(line) - len(line.lstrip(' '))

def flatten_java_code(code_lines):
    """
    Convert multi-line Java code into a single line with minimal spacing,
    mimicking your example style.
    """
    # 1) Strip each line and skip blank lines.
    stripped_lines = [ln.strip() for ln in code_lines if ln.strip()]
    # 2) Join with a single space.
    merged = " ".join(stripped_lines)
    # 3) Collapse multiple spaces into one.
    merged = re.sub(r"\s+", " ", merged)
    # 4) Remove space before any curly brace.
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
    Process Java code by flattening it into one line and wrapping with <JAVA>.
    """
    flattened = flatten_java_code(code_lines)
    return f"<JAVA>\n{flattened}"

def wrap_as_python(code_lines, spaces_per_level=4):
    """
    Process Python code:
      1) Clean the lines (remove blank lines and comments).
      2) Insert INDENT/DEDENT tokens based on indentation.
      3) Wrap with <PYTHON>.
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
