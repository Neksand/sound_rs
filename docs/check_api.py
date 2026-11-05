#!/usr/bin/env python3
"""Fails when a public name in sound_rs.pyi is missing from docs/api.html.

Documentation rots silently otherwise. Run from the repo root:

    python3 docs/check_api.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(HERE, "..", "sound_rs.pyi")
API = os.path.join(HERE, "api.html")

# Names that are documented somewhere other than the API page.
ELSEWHERE = {"__version__": "index.html", "cli": "cli.html"}


def stub_names():
    text = open(STUB).read()
    names = set()
    for match in re.finditer(r"^def ([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
        names.add(match.group(1))
    for match in re.finditer(r"^class ([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
        names.add(match.group(1))
    return {name for name in names if not name.startswith("_")}


def main():
    page = open(API).read()
    missing = []
    for name in sorted(stub_names()):
        if name in ELSEWHERE:
            target = os.path.join(HERE, ELSEWHERE[name])
            if not os.path.exists(target):
                missing.append(f"{name} (no {ELSEWHERE[name]})")
            continue
        # The page documents a name if it appears as a heading or in a signature.
        if f'id="{name.lower().replace("_", "-")}"' in page or name in page:
            continue
        missing.append(name)

    if missing:
        print("Undocumented public API: " + ", ".join(missing))
        print("Add a section to docs/_content.py and rebuild with python3 docs/_build.py")
        return 1
    print(f"api drift ok ({len(stub_names())} public names)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
