# reading_copy_check

When a script is published as a "reading copy" (same code, comments and
docstrings rewritten for readers), how do you know the logic shown is the logic
that ran? This tool compares the two files by abstract syntax tree and checks
that no forbidden text slipped into the copy.

## What it does

1. The copy must parse.
2. The program skeleton must be identical: both ASTs are compared after
   blanking docstrings and every string constant that contains whitespace
   (sentences, messages, report text). Every other string (file names, keys,
   codes) and every non-string node must match exactly. Changed prose strings
   are printed so a reader can confirm the change is wording only.
3. No forbidden text in comments, docstrings or changed strings. The patterns
   are configurable: `forbidden_patterns.json` carries a generic default (dates
   of work, log references, name placeholders, internal path prefixes) and a
   word list for a second language, plus `allowed_strings` for strings that
   must stay as they are (they are program logic).

## How to use it

```
python reading_copy_check/check_reading_copy.py ORIGINAL.py COPY.py
python reading_copy_check/check_reading_copy.py --all ORIG_ROOT COPY_DIR        # every *.py in COPY_DIR against ORIG_ROOT/src/*/
python reading_copy_check/check_reading_copy.py --config my_patterns.json ORIGINAL.py COPY.py
```

Exit code 0 means clean, 1 means problems (each one printed). Replace the
`PERSON_NAME|SUPERVISOR_NAME` placeholder in the JSON with the names that must
not appear in your copies, and adjust the path prefixes to your layout.

## Where the method comes from

Plain use of Python's `ast` and `tokenize` modules: `ast.dump` of both trees
with string constants normalised is the comparison; `tokenize` recovers the
comments that the AST drops.
