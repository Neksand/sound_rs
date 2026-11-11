#!/usr/bin/env python3
"""Check registered Rust APIs, Python stubs and generated API documentation.

The Rust reader follows this repository's rustfmt layout and PyO3 registration
patterns; it is deliberately not a general Rust parser. No extension import or
sound device is required.
"""

import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent


def stub_api(text):
    tree = ast.parse(text)
    names, classes = set(), {}
    exports = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {
                    member.name for member in node.body
                    if isinstance(member, ast.FunctionDef)
                }
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__version__":
                names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                exports = set(ast.literal_eval(node.value))
    return names, classes, exports


def rust_api(root):
    module = (root / "src/lib.rs").read_text(encoding="utf-8")
    names = {name.split("::")[-1] for name in re.findall(r"wrap_pyfunction!\(\s*([\w:]+)", module)}
    names.update(name.split("::")[-1] for name in re.findall(r"\.add_class::<([\w:]+)>", module))
    names.update(re.findall(r'\bm\.add\(\s*"([\w]+)"', module))
    classes = {}
    for path in sorted((root / "src").glob("*.rs")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"#\[pymethods\]\s*impl\s+(\w+)\s*\{(.*?)^\}", source, re.M | re.S):
            name, body = match.groups()
            members = classes.setdefault(name, set())
            attributes = []
            for line in body.splitlines():
                if line.startswith("    #["):
                    attributes.append(line)
                method = re.match(r"    (?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)", line)
                if not method:
                    continue
                member = method.group(1)
                attrs = "\n".join(attributes)
                if "#[new]" in attrs:
                    member = "__init__"
                elif "#[setter" in attrs:
                    explicit = re.search(r"#\[setter\((\w+)\)\]", attrs)
                    member = explicit.group(1) if explicit else member.removeprefix("set_")
                alias = re.search(r'#\[pyo3\(name\s*=\s*"(\w+)"\)\]', attrs)
                if alias:
                    member = alias.group(1)
                members.add(member)
                attributes = []
    return names, classes


class ApiPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_main = False
        self.section = None
        self.sections = {}
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "main":
            self.in_main = True
        if self.in_main and tag == "h2":
            self.section = attrs.get("id")
            self.sections.setdefault(self.section, [])

    def handle_endtag(self, tag):
        if tag == "main":
            self.in_main = False

    def handle_data(self, data):
        if self.in_main:
            self.text.append(data)
            if self.section is not None:
                self.sections[self.section].append(data)


def api_issues(root=ROOT):
    root = Path(root)
    names, classes, exports = stub_api((root / "sound_rs.pyi").read_text(encoding="utf-8"))
    rust_names, rust_classes = rust_api(root)
    issues = []
    for name in sorted(rust_names - names):
        issues.append(f"Rust API missing from stub: {name}")
    for name in sorted(names - rust_names):
        issues.append(f"Stub API not registered in Rust: {name}")
    if exports is not None and exports != names:
        issues.append(f"Stub __all__ mismatch: {', '.join(sorted(exports ^ names))}")
    for name, members in rust_classes.items():
        if name not in rust_names:
            continue
        for member in sorted(members - classes.get(name, set())):
            issues.append(f"Rust member missing from stub: {name}.{member}")
        for member in sorted(classes.get(name, set()) - members):
            issues.append(f"Stub member missing from Rust: {name}.{member}")

    page = ApiPage()
    page.feed((root / "docs/api.html").read_text(encoding="utf-8"))
    text = " ".join(page.text)
    for name in sorted(names):
        anchor = name.lower().replace("_", "-")
        if anchor not in page.sections and not re.search(rf"\b{re.escape(name)}\b", text):
            issues.append(f"Undocumented API: {name}")
        if name in rust_classes:
            section = " ".join(page.sections.get(anchor, []))
            for member in sorted(classes.get(name, set())):
                # Constructors and context protocols are illustrated by class
                # signatures/with examples; source/stub parity still covers them.
                if member.startswith("__"):
                    continue
                if not re.search(rf"\b{re.escape(member)}\b", section):
                    issues.append(f"Undocumented member: {name}.{member}")
    return issues


def main():
    issues = api_issues()
    if issues:
        print("\n".join(issues))
        print("Update Rust/stubs and docs/_content.py, then run python docs/_build.py")
        return 1
    names, classes, _ = stub_api((ROOT / "sound_rs.pyi").read_text(encoding="utf-8"))
    print(f"api drift ok ({len(names)} names, {sum(map(len, classes.values()))} class members)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
