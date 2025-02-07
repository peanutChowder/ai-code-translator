#!/usr/bin/env python3
import sys
import json
import re

# Regex to detect tokens that are purely punctuation or bracket-like
# We'll treat them differently from identifiers/keywords.
PUNCT_REGEX = re.compile(r'^[\[\]{}().;,+\-*/<>=&|^%!?:@#~]+$')

# We remove/ignore "▁" (if it appears), but do NOT remove "NEW_LINE".
# Instead, we will explicitly convert "NEW_LINE" into a real line break below.
SPECIAL_PLACEHOLDERS = {"▁"}

def untokenize_java(tokens):
    """
    Reconstruct code from a list of Java tokens with improved handling of '.'.
    Steps:
      1) Convert "NEW_LINE" tokens into an actual newline, remove "▁" placeholders.
      2) If a token is ".", we try to merge it with the previous or next token (e.g. "System . out" -> "System.out").
      3) Otherwise:
         - If the token matches punctuation (PUNCT_REGEX), we stick it directly to the previous token (no extra space).
         - Else (alphanumeric keyword, etc.), we prepend a space if needed.
      4) Return a multi-line string where "NEW_LINE" tokens have become real newlines.
    """

    # -- First pass: handle placeholders and NEW_LINE differently --
    filtered = []
    for t in tokens:
        if t == "NEW_LINE":
            # Convert "NEW_LINE" to an actual newline token
            filtered.append("\n")
        elif t in SPECIAL_PLACEHOLDERS:
            # Remove tokens like "▁"
            continue
        else:
            filtered.append(t)

    # -- Second pass: merge '.' with neighboring tokens if possible --
    merged = []
    i = 0
    while i < len(filtered):
        tok = filtered[i]

        # If tok is ".", see if next token is an identifier (so we can do e.g. System.out)
        if tok == "." and merged:
            prev = merged[-1]  # last token in 'merged'
            nxt = filtered[i + 1] if (i + 1 < len(filtered)) else None

            # If next token is not punctuation or a newline, merge them
            if nxt and nxt != "\n" and not PUNCT_REGEX.match(nxt):
                merged[-1] = prev + "." + nxt
                i += 2
            else:
                # Just attach the dot to 'prev'
                merged[-1] = prev + "."
                i += 1
        else:
            merged.append(tok)
            i += 1

    # -- Final pass: build the output code string --
    #    If it's punctuation (e.g. '('), attach directly; if it's '\n', insert a newline;
    #    otherwise add a space if needed.
    output_parts = []
    for tok in merged:
        if tok == "\n":
            # Insert an actual line break in the final code
            output_parts.append("\n")
        elif PUNCT_REGEX.match(tok):
            # It's punctuation, attach to the previous chunk (no space)
            if output_parts:
                # If the last chunk ends with a newline, we just append.
                # Otherwise, we combine them with no space.
                if output_parts[-1].endswith("\n"):
                    output_parts.append(tok)
                else:
                    output_parts[-1] += tok
            else:
                output_parts.append(tok)
        else:
            # It's an identifier, keyword, or number
            # Add a space if the last chunk doesn't end with space or newline
            if output_parts and not (output_parts[-1].endswith(" ") or output_parts[-1].endswith("\n")):
                output_parts[-1] += " "
            output_parts.append(tok)

    # Join them all
    code_str = "".join(output_parts)

    # Strip any leading or trailing whitespace/newlines
    return code_str.strip("\n")

def preprocess_xlcost_java(input_jsonl, output_txt, field_name="docstring_tokens"):
    """
    Reads a JSONL file (XLCoST style), extracts the token list from `field_name`,
    reconstructs the code with improved spacing/punctuation, and writes each snippet
    to `output_txt`. Note that the snippet might span multiple lines if "NEW_LINE" tokens appear.
    """
    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_txt, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            
            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

            # Extract tokens for the field
            tokens = data.get(field_name, [])
            if not isinstance(tokens, list) or not tokens:
                # If there's no valid token list, skip
                continue
            
            # Reconstruct into code (potentially multi-line)
            code_str = untokenize_java(tokens)
            if code_str:
                # Write the snippet
                fout.write(code_str + "\n")

if __name__ == "__main__":
    """
    Example usage:
       python preprocess_xlcost_java.py xlcost_java.jsonl xlcost_java_reconstructed.txt docstring_tokens
    
    1) "xlcost_java.jsonl" is your XLCoST file containing JSON lines with 'docstring_tokens' or 'code_tokens'.
    2) "xlcost_java_reconstructed.txt" is the output file. The code snippet may be multi-line.
    3) "docstring_tokens" (or "code_tokens") is the JSON key that holds the Java tokens you want to reconstruct.
    """
    if len(sys.argv) < 4:
        print("Usage: python preprocess_xlcost_java.py <input_jsonl> <output_txt> <field_name>")
        print("Example: python preprocess_xlcost_java.py xlcost_java.jsonl xlcost_java_recon.txt docstring_tokens")
        sys.exit(1)

    input_jsonl_path = sys.argv[1]
    output_txt_path = sys.argv[2]
    field_name = sys.argv[3]

    preprocess_xlcost_java(input_jsonl_path, output_txt_path, field_name)
