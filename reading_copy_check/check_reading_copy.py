"""Check that a reading copy of a script differs from the original only in comments, docstrings and human-readable
text, so that the program logic shown to readers is the logic that ran.

Rules checked:
  1. The copy parses.
  2. The program skeleton is identical: the AST of both files is compared after blanking (a) docstrings and
     (b) string constants that contain whitespace (sentences, messages, report text). Every other string constant
     (file names, column names without spaces, keys, codes) must be identical, and so must every non-string node.
  3. No forbidden text remains in comments, docstrings or changed text: dates of work, log references, names,
     internal path prefixes, text in a second language. The patterns come from a JSON config (see
     forbidden_patterns.json next to this file) or from the generic defaults below.
It prints every changed non-docstring string so that a reader can confirm the change is only wording.

Run:  python check_reading_copy.py ORIGINAL.py COPY.py      (exit 0 = clean, 1 = problems)
      python check_reading_copy.py --all ORIG_ROOT COPY_DIR  (every *.py in COPY_DIR against its original under ORIG_ROOT/src/*/)
      python check_reading_copy.py --config PATTERNS.json ...  (patterns from a JSON file; the default is forbidden_patterns.json)
"""
import ast
import io
import json
import os
import re
import sys
import tokenize

# Generic defaults: dates of work, work-log references, name placeholders, internal path prefixes. A project
# replaces the placeholders with the names that must not appear and adds its own paths.
DEFAULT_FORBIDDEN = [
    r"\b\d{1,2}(?:\s*(?:-|to|and)\s*\d{1,2})? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? 20\d\d\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? 20\d\d\b",
    r"(?<![/\w-])20\d\d-\d\d-\d\d(?![\w-])",      # a date inside a file name (data/<yyyy-mm-dd>-set.csv) is code and stays
    r"\blog \d[a-z]{1,3}\b",
    r"\b(?i:PERSON_NAME|SUPERVISOR_NAME)\b",
    r"(?<![\w/])(?:results|docs|private)/",
]
DEFAULT_SPANISH = (r"\b(?:que|los|las|del|para|una|con|por|cada|segun|tambien|esta|desde|entre|como|cuando|donde|solo|"
                   r"porque|pero|sin|sobre|hasta|nunca|siempre|ya|aqui|este|estos|datos|valor|valores|fuente|fuentes|"
                   r"version|archivo|lector)\b")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forbidden_patterns.json")

FORBIDDEN = re.compile("|".join(DEFAULT_FORBIDDEN))
SPANISH = re.compile(DEFAULT_SPANISH, re.IGNORECASE)
# (file name, regular expression matching a string of the original) pairs: strings that are program logic and must
# stay as they are even though they read as text in the second language.
SPANISH_PATTERNS = set()


def load_config(path):
    """Replace the module patterns with those of a JSON file: {"forbidden": [regex, ...], "spanish_words": regex,
    "allowed_strings": [[file name, regex], ...]}. Missing keys keep the defaults."""
    global FORBIDDEN, SPANISH, SPANISH_PATTERNS
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg.get("forbidden"):
        FORBIDDEN = re.compile("|".join(cfg["forbidden"]))
    if cfg.get("spanish_words"):
        SPANISH = re.compile(cfg["spanish_words"], re.IGNORECASE)
    SPANISH_PATTERNS = {(f, p) for f, p in cfg.get("allowed_strings", [])}


def docstring_ids(tree):
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def skeleton(src):
    """The AST with every string constant blanked, plus the list of (is_docstring, value) in walk order."""
    tree = ast.parse(src)
    docs = docstring_ids(tree)
    strings = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            strings.append((id(n) in docs, n.value))
            n.value = "<str>"
    return ast.dump(tree, include_attributes=False), strings


def comments_and_docs(src):
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out.append(("comment", tok.start[0], tok.string))
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.append(("docstring", first.lineno, first.value.value))
    return out


def _allowed(copy_path, original_string):
    name = os.path.basename(copy_path)
    return any(f == name and re.search(p, original_string) for f, p in SPANISH_PATTERNS)


def check(orig_path, copy_path, verbose=True):
    o = open(orig_path, encoding="utf-8", newline="").read()
    c = open(copy_path, encoding="utf-8", newline="").read()
    problems = []
    try:
        sk_c, str_c = skeleton(c)
    except SyntaxError as e:
        return [f"copy does not parse: {e}"]
    sk_o, str_o = skeleton(o)
    if sk_o != sk_c or len(str_o) != len(str_c):
        problems.append("program skeleton differs (a non-text part of the code changed)")
    else:
        changed = []
        for (is_doc, a), (_, b) in zip(str_o, str_c):
            if is_doc:
                continue
            if a != b and not re.search(r"\s", a):
                problems.append(f"a string without spaces changed (file name, key or code value): {a!r} -> {b!r}")
            elif a != b:
                changed.append((a, b))
            if re.search(r"\s", a):
                for m in FORBIDDEN.finditer(b):
                    problems.append(f"forbidden text in a string: {m.group(0)!r} in {b[:100]!r}")
                if len(SPANISH.findall(b)) >= 3 and not _allowed(copy_path, a):
                    problems.append(f"second-language text in a string: {b[:90]!r}")
        if verbose and changed:
            print(f"  {len(changed)} changed text strings:")
            for a, b in changed:
                print(f"    - {a[:160]!r}\n    + {b[:160]!r}")
    for kind, line, text in comments_and_docs(c):
        for m in FORBIDDEN.finditer(text):
            problems.append(f"forbidden text in {kind} at line {line}: {m.group(0)!r}")
        n_es = len(SPANISH.findall(text))
        if n_es >= 3 or (kind == "comment" and n_es >= 2):
            problems.append(f"second-language text in {kind} at line {line}: {text[:90]!r}")
    return problems


def main():
    args = sys.argv[1:]
    if args and args[0] == "--config":
        load_config(args[1])
        args = args[2:]
    elif os.path.exists(DEFAULT_CONFIG):
        load_config(DEFAULT_CONFIG)
    if args[0] == "--all":
        root, d = args[1], args[2]
        index = {}
        for sub in os.listdir(os.path.join(root, "src")):
            p = os.path.join(root, "src", sub)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.endswith(".py"):
                        index[f] = os.path.join(p, f)
        bad = 0
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                pr = check(index[f], os.path.join(d, f), verbose=False)
                print(f"{'OK ' if not pr else 'BAD'} {f}" + ("" if not pr else f": {len(pr)} problems; first: {pr[0]}"))
                bad += bool(pr)
        sys.exit(1 if bad else 0)
    pr = check(args[0], args[1])
    for p in pr:
        print("PROBLEM:", p)
    print("CLEAN" if not pr else f"{len(pr)} problems")
    sys.exit(1 if pr else 0)


if __name__ == "__main__":
    main()
