#!/usr/bin/env python3
"""Validate generated local links, anchors and discovery indexes without network I/O."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SITE = "https://neksand.github.io/sound_rs/"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.duplicates = set()
        self.links = []
        self.canonical = None
        self.markdown = None
        self.describedby = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identifier = attrs.get("id")
        if identifier:
            if identifier in self.ids:
                self.duplicates.add(identifier)
            self.ids.add(identifier)
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])
        if tag == "link":
            rel = attrs.get("rel", "").split()
            if "canonical" in rel:
                self.canonical = attrs.get("href")
            if "alternate" in rel and attrs.get("type") == "text/markdown":
                self.markdown = attrs.get("href")
            if "describedby" in rel:
                self.describedby = attrs.get("href")


def local_target(site, source, reference):
    """Return a local file and fragment, or None for an external reference."""
    url = urlsplit(reference)
    base = urlsplit(SITE)
    if url.scheme or url.netloc:
        if url.scheme not in {"http", "https"} or url.netloc != base.netloc or not url.path.startswith(base.path):
            return None
        path = url.path[len(base.path):]
        target = site / unquote(path)
    elif url.path.startswith("/"):
        if not url.path.startswith(base.path):
            return None
        target = site / unquote(url.path[len(base.path):])
    else:
        target = source.parent / unquote(url.path) if url.path else source
    if target.is_dir():
        target = target / "index.html"
    return target.resolve(), unquote(url.fragment)


def site_issues(site=HERE):
    site = Path(site).resolve()
    issues = []
    pages = {}
    for path in sorted(site.glob("*.html")):
        page = Page()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = page
        for duplicate in sorted(page.duplicates):
            issues.append(f"{path.name}: duplicate id #{duplicate}")
        if page.canonical != SITE + path.name:
            issues.append(f"{path.name}: missing or incorrect canonical URL")
        if page.markdown != path.name + ".md":
            issues.append(f"{path.name}: missing Markdown alternate")
        if page.describedby != "llms.txt":
            issues.append(f"{path.name}: missing llms.txt describedby link")
        markdown = path.with_name(path.name + ".md")
        if not markdown.is_file() or not markdown.read_text(encoding="utf-8").strip():
            issues.append(f"{path.name}: missing or empty Markdown counterpart")
    if not pages:
        return ["No generated HTML pages found"]

    def check_link(source, reference):
        resolved = local_target(site, source, reference)
        if resolved is None:
            return None
        target, fragment = resolved
        if not target.is_relative_to(site) or not target.is_file():
            issues.append(f"{source.name}: missing local target {reference}")
        elif fragment and target.suffix == ".html":
            target_page = pages.get(target)
            if target_page is None:
                target_page = Page()
                target_page.feed(target.read_text(encoding="utf-8"))
            if fragment not in target_page.ids:
                issues.append(f"{source.name}: missing anchor {reference}")
        return target

    for path, page in pages.items():
        for reference in page.links:
            check_link(path, reference)
    visible = {path.name for path in pages if path.name != "404.html"}

    try:
        search = json.loads((site / "search.json").read_text(encoding="utf-8"))
        if not isinstance(search, list):
            raise ValueError("expected a list")
        urls = []
        for item in search:
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k] for k in ("url", "title", "body", "section")):
                raise ValueError("each entry needs nonempty url/title/body/section strings")
            urls.append(item["url"])
            check_link(site / "search.json", item["url"])
        if len(urls) != len(set(urls)) or set(urls) != visible:
            issues.append("search.json: page coverage differs from generated pages")
    except (OSError, ValueError, TypeError) as error:
        issues.append(f"search.json: {error}")

    try:
        tree = ET.parse(site / "sitemap.xml")
        locations = [node.text or "" for node in tree.findall(".//{*}loc")]
        expected = {SITE + page for page in visible}
        if len(locations) != len(set(locations)) or set(locations) != expected:
            issues.append("sitemap.xml: page coverage differs from generated pages")
        for location in locations:
            check_link(site / "sitemap.xml", location)
    except (OSError, ET.ParseError) as error:
        issues.append(f"sitemap.xml: {error}")

    try:
        llms = (site / "llms.txt").read_text(encoding="utf-8")
        targets = set()
        for reference in re.findall(r"\[[^\]]+\]\(([^)]+)\)", llms):
            target = check_link(site / "llms.txt", reference)
            if target is not None:
                targets.add(target.name)
        missing = {page + ".md" for page in visible} - targets
        if missing:
            issues.append("llms.txt: missing pages " + ", ".join(sorted(missing)))
    except OSError as error:
        issues.append(f"llms.txt: {error}")
    return issues


def main():
    issues = site_issues()
    if issues:
        print("\n".join(issues))
        return 1
    print("site links, anchors, Markdown, search, sitemap and llms.txt ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
