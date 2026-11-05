#!/usr/bin/env python3
"""Builds the static docs into docs/*.html.

No dependencies: run it, commit the output, GitHub Pages serves it directly.
    python3 docs/_build.py
"""

import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = "https://neksand.github.io/sound_rs/"
VERSION = "0.2.1"
REPO = "https://github.com/Neksand/sound_rs"
PYPI = "https://pypi.org/project/sound-rs/"

CSS = r"""
:root {
  --bg: #08090b;
  --bg-2: #0d0f12;
  --fg: #edeae4;
  --text: #c3c7c6;
  --dim: #7e858c;
  --line: rgba(237, 234, 228, 0.13);
  --line-strong: rgba(237, 234, 228, 0.25);
  --ember: #ff5a36;

  --display: "Helvetica Neue", Helvetica, Arial, sans-serif;
  --body: Georgia, "Times New Roman", serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;

  --pad: clamp(1.25rem, 4vw, 3rem);
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

*, *::before, *::after { box-sizing: border-box; }

html {
  -webkit-text-size-adjust: 100%;
  scroll-behavior: smooth;
  scroll-padding-top: 4.5rem;
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  * { animation: none !important; transition: none !important; }
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--body);
  font-size: 1.0625rem;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
  overflow-wrap: break-word;
}

::selection { background: var(--ember); color: var(--bg); }

a { color: var(--fg); text-decoration: none; border-bottom: 1px solid var(--line-strong); }
a:hover { border-color: var(--ember); }
a:focus-visible, button:focus-visible, input:focus-visible,
summary:focus-visible, [tabindex]:focus-visible {
  outline: 2px solid var(--ember);
  outline-offset: 3px;
}

.skip {
  position: fixed; left: -9999px; top: 1rem;
  background: var(--fg); color: var(--bg);
  padding: 0.5rem 0.9rem; z-index: 100;
  font-family: var(--display); font-weight: 700;
}
.skip:focus { left: 1rem; }

/* ---------- top bar ---------- */

.top {
  position: sticky; top: 0; z-index: 20;
  display: flex; align-items: baseline; gap: 1.25rem;
  padding: 0.9rem var(--pad);
  background: var(--bg);
  border-bottom: 1px solid var(--line);
}
.brand {
  font-family: var(--display);
  font-weight: 700;
  font-size: 1.05rem;
  letter-spacing: -0.045em;
  border: 0;
}
.ver {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--dim);
  letter-spacing: 0;
}
.top nav { margin-left: auto; display: flex; align-items: baseline; gap: 1.1rem; }
.top nav a, .top nav button {
  font-family: var(--display);
  font-size: 0.82rem;
  letter-spacing: -0.01em;
  color: var(--text);
  border: 0;
  background: none;
  padding: 0;
  cursor: pointer;
}
.top nav a:hover, .top nav button:hover { color: var(--fg); }
kbd {
  font-family: var(--mono);
  font-size: 0.7rem;
  border: 1px solid var(--line-strong);
  padding: 0.05rem 0.3rem;
  margin-left: 0.3rem;
  color: var(--dim);
}

/* ---------- shell ---------- */

.shell {
  display: grid;
  grid-template-columns: 15rem minmax(0, 1fr) 13rem;
  gap: 0 var(--pad);
  max-width: 84rem;
  margin: 0 auto;
  padding: 2.5rem var(--pad) 5rem;
}
@media (max-width: 68rem) { .shell { grid-template-columns: 14rem minmax(0, 1fr); } .toc { display: none; } }
@media (max-width: 34rem) {
  .top { gap: 0.8rem; }
  .top .ver, .top nav a[href*="pypi"] { display: none; }
  .top nav { gap: 0.9rem; }
}
@media (max-width: 52rem) {
  .shell { grid-template-columns: minmax(0, 1fr); padding-top: 1.5rem; }
  .side { position: static; order: -1; padding-bottom: 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1.75rem; }
}

.side { position: sticky; top: 4.5rem; align-self: start; font-family: var(--display); font-size: 0.85rem; }
.side details { border: 0; }
.side summary { display: none; }
@media (max-width: 52rem) {
  .side summary { display: block; cursor: pointer; padding: 0.4rem 0; color: var(--text); }
  .side details:not([open]) .side-body { display: none; }
}
.side-body { display: block; }
.side h2 {
  font-family: var(--display);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--dim);
  margin: 1.4rem 0 0.5rem;
}
.side h2:first-child { margin-top: 0; }
.side a {
  display: block;
  padding: 0.22rem 0 0.22rem 0.7rem;
  color: var(--text);
  border: 0;
  border-left: 1px solid var(--line);
  letter-spacing: -0.01em;
}
.side a:hover { color: var(--fg); border-left-color: var(--line-strong); }
.side a[aria-current="page"] { color: var(--fg); border-left: 2px solid var(--ember); padding-left: calc(0.7rem - 1px); }

main { min-width: 0; }
main > * { max-width: 46rem; }

/* ---------- prose ---------- */

h1, h2, h3, h4 {
  font-family: var(--display);
  letter-spacing: -0.045em;
  line-height: 1.15;
  margin: 0;
}
h1 { font-size: clamp(2.1rem, 5vw, 3rem); font-weight: 700; margin-bottom: 0.75rem; text-wrap: balance; }
h2 { font-size: 1.45rem; font-weight: 700; margin: 3rem 0 0.85rem; text-wrap: balance; }
h3 { font-size: 1.1rem; font-weight: 700; margin: 2rem 0 0.6rem; }
h2 a, h3 a { border: 0; }

.lead { font-size: 1.2rem; color: var(--text); margin: 0 0 2rem; max-width: 40rem; }
.kicker {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--dim);
  letter-spacing: 0.02em;
  margin-bottom: 0.9rem;
}
p { margin: 0 0 1.05rem; color: var(--text); }
main strong { color: var(--fg); }
ul, ol { margin: 0 0 1.15rem; padding-left: 1.15rem; color: var(--text); }
li { margin-bottom: 0.35rem; }
li::marker { color: var(--dim); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2.5rem 0; }

code {
  font-family: var(--mono);
  font-size: 0.86em;
  color: var(--fg);
  background: rgba(237, 234, 228, 0.07);
  border: 0;
  padding: 0.1em 0.34em;
}

pre {
  position: relative;
  background: var(--bg-2);
  border: 1px solid var(--line);
  padding: 1.1rem 1.15rem;
  overflow-x: auto;
  margin: 0 0 1.3rem;
  line-height: 1.5;
}
pre code { background: none; border: 0; padding: 0; font-size: 0.83rem; color: var(--text); }
pre .tok-key { color: var(--fg); }
pre .tok-str { color: #b9c6a6; }
pre .tok-com { color: var(--dim); font-style: italic; }
pre .tok-num { color: var(--ember); }
.copy {
  position: absolute; top: 0.5rem; right: 0.5rem;
  font-family: var(--display); font-size: 0.68rem;
  letter-spacing: 0.02em;
  color: var(--dim); background: var(--bg-2);
  border: 1px solid var(--line); padding: 0.2rem 0.45rem;
  cursor: pointer; opacity: 0;
}
pre:hover .copy, .copy:focus-visible { opacity: 1; }
.copy:hover { color: var(--fg); border-color: var(--line-strong); }

/* callouts: a left rule that carries meaning, label in mono */
.note, .warn {
  border-left: 2px solid var(--line-strong);
  padding: 0.15rem 0 0.15rem 1rem;
  margin: 1.4rem 0;
  color: var(--text);
}
.warn { border-left-color: var(--ember); }
.note b, .warn b {
  display: block;
  font-family: var(--display); font-size: 0.72rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--dim); margin-bottom: 0.25rem;
}
.warn b { color: var(--ember); }

@media (max-width: 44rem) {
  table { display: block; overflow-x: auto; }
  th, td { white-space: nowrap; }
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 1.4rem;
  font-size: 0.95rem;
}
th, td { text-align: left; padding: 0.45rem 0.7rem 0.45rem 0; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-family: var(--display); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--dim); font-weight: 700; }
td { color: var(--text); }
td.num, th.num { font-family: var(--mono); font-size: 0.84rem; white-space: nowrap; }
td strong { color: var(--fg); }

/* ---------- toc ---------- */

.toc { position: sticky; top: 4.5rem; align-self: start; font-family: var(--display); font-size: 0.78rem; }
.toc h2 {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--dim); margin: 0 0 0.6rem;
}
.toc a { display: block; color: var(--dim); border: 0; padding: 0.18rem 0; letter-spacing: -0.01em; }
.toc a:hover, .toc a.active { color: var(--fg); }

/* ---------- index extras ---------- */

.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); margin: 1.75rem 0 2.25rem; }
.figures div { background: var(--bg); padding: 0.9rem 1rem; }
.figures span { display: block; font-family: var(--display); font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--dim); margin-bottom: 0.25rem; }
.figures b { font-family: var(--mono); font-weight: 400; font-size: 1.35rem; color: var(--fg); letter-spacing: -0.02em; }
.figures i { display: block; font-style: normal; font-size: 0.82rem; color: var(--dim); }

.next { display: flex; justify-content: space-between; gap: 1rem; border-top: 1px solid var(--line); margin-top: 3.5rem; padding-top: 1rem; font-family: var(--display); font-size: 0.85rem; }
.next a { border: 0; }

/* ---------- search ---------- */

dialog.search {
  border: 1px solid var(--line-strong);
  background: var(--bg);
  color: var(--fg);
  padding: 0;
  width: min(38rem, calc(100vw - 2rem));
  margin: 12vh auto auto;
}
dialog.search::backdrop { background: rgba(4, 4, 6, 0.75); }
dialog.search input {
  width: 100%;
  background: var(--bg-2);
  border: 0; border-bottom: 1px solid var(--line);
  color: var(--fg);
  font-family: var(--display); font-size: 1rem;
  padding: 0.9rem 1rem;
}
dialog.search input:focus { outline: none; }
dialog.search ul { list-style: none; margin: 0; padding: 0.4rem; max-height: 55vh; overflow: auto; }
dialog.search li { margin: 0; }
dialog.search a {
  display: block; padding: 0.5rem 0.7rem; border: 0;
  font-family: var(--display); font-size: 0.9rem; color: var(--text);
}
dialog.search a b { color: var(--fg); font-weight: 700; }
dialog.search a span { display: block; font-size: 0.78rem; color: var(--dim); }
dialog.search a:hover, dialog.search a.sel { background: var(--bg-2); }
.search-empty { padding: 0.9rem 1rem; color: var(--dim); font-family: var(--display); font-size: 0.85rem; }

footer {
  border-top: 1px solid var(--line);
  padding: 1.5rem var(--pad) 3rem;
  display: flex; flex-wrap: wrap; gap: 1rem;
  font-family: var(--display); font-size: 0.8rem;
  color: var(--dim);
}
footer a { color: var(--text); border: 0; }
footer a:hover { color: var(--fg); }
footer .spacer { margin-left: auto; }
"""

JS = r"""
(function () {
  var dialog = document.querySelector('dialog.search');
  var input = dialog && dialog.querySelector('input');
  var list = dialog && dialog.querySelector('ul');
  var index = null;
  var selected = 0;

  function load() {
    if (index) return Promise.resolve(index);
    return fetch('search.json').then(function (r) { return r.json(); }).then(function (data) {
      index = data;
      return data;
    });
  }

  function count(hay, needle) {
    var n = 0, at = hay.indexOf(needle);
    while (at !== -1) { n++; at = hay.indexOf(needle, at + needle.length); }
    return n;
  }

  function score(item, q) {
    var title = item.title.toLowerCase();
    var headings = (item.headings || '').toLowerCase();
    var body = item.body.toLowerCase();
    var titleHits = count(title, q);
    var headingHits = count(headings, q);
    var bodyHits = count(body, q);
    if (!titleHits && !headingHits && !bodyHits) return 0;
    // A page named after the query beats a page that merely mentions it.
    return titleHits * 50 + headingHits * 10 + bodyHits;
  }

  function render(items) {
    list.innerHTML = '';
    if (!items.length) {
      list.innerHTML = '<li class="search-empty">Nothing found. Try "player", "seek", "flac".</li>';
      return;
    }
    items.forEach(function (item, i) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = item.url;
      a.innerHTML = '<b>' + item.title + '</b><span>' + item.section + '</span>';
      if (i === 0) { a.classList.add('sel'); }
      li.appendChild(a);
      list.appendChild(li);
    });
    selected = 0;
  }

  function search(q) {
    load().then(function (data) {
      q = q.trim().toLowerCase();
      if (!q) { render(data.slice(0, 8)); return; }
      var hits = data
        .map(function (item) { return { item: item, s: score(item, q) }; })
        .filter(function (h) { return h.s > 0; })
        .sort(function (a, b) { return b.s - a.s || a.item.title.length - b.item.title.length; })
        .slice(0, 10)
        .map(function (h) { return h.item; });
      render(hits);
    });
  }

  function open() {
    if (!dialog) return;
    if (!dialog.open) dialog.showModal();
    input.value = '';
    search('');
    input.focus();
  }

  document.addEventListener('click', function (event) {
    var trigger = event.target.closest('[data-search]');
    if (trigger) { event.preventDefault(); open(); }
  });

  document.addEventListener('keydown', function (event) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
    if (event.key === '/' && !typing) { event.preventDefault(); open(); return; }
    if (event.key === 'Escape' && dialog && dialog.open) { dialog.close(); return; }
    if (!dialog || !dialog.open) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      var links = list.querySelectorAll('a');
      if (!links.length) return;
      event.preventDefault();
      selected = (selected + (event.key === 'ArrowDown' ? 1 : links.length - 1)) % links.length;
      links.forEach(function (l, i) { l.classList.toggle('sel', i === selected); });
      links[selected].scrollIntoView({ block: 'nearest' });
    }
    if (event.key === 'Enter') {
      var link = list.querySelector('a.sel');
      if (link) { event.preventDefault(); window.location.href = link.href; }
    }
  });

  if (input) {
    input.addEventListener('input', function () { search(input.value); });
  }

  // copy buttons
  document.querySelectorAll('pre').forEach(function (pre) {
    var button = document.createElement('button');
    button.className = 'copy';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', function () {
      var code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.innerText : pre.innerText).then(function () {
        button.textContent = 'Copied';
        setTimeout(function () { button.textContent = 'Copy'; }, 1400);
      });
    });
    pre.appendChild(button);
  });

  // table of contents
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  var headings = tocLinks
    .map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); })
    .filter(Boolean);
  if (headings.length && 'IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        tocLinks.forEach(function (a) {
          a.classList.toggle('active', a.getAttribute('href') === '#' + entry.target.id);
        });
      });
    }, { rootMargin: '-10% 0px -75% 0px' });
    headings.forEach(function (h) { observer.observe(h); });
  }
})();
"""

NAV = [
    ("Start", [
        ("index.html", "Overview"),
        ("install.html", "Installation"),
        ("quickstart.html", "Quickstart"),
    ]),
    ("Guides", [
        ("player.html", "The player"),
        ("streaming.html", "Streaming and memory"),
        ("formats.html", "Formats and gapless"),
        ("numpy.html", "NumPy and arrays"),
        ("migration.html", "Coming from another library"),
    ]),
    ("Reference", [
        ("api.html", "API reference"),
        ("benchmarks.html", "Benchmarks"),
        ("faq.html", "Troubleshooting"),
        ("changelog.html", "Changelog"),
    ]),
]

PAGES = []


def page(slug, title, description, body, toc=None, hidden=False):
    PAGES.append(
        {
            "slug": slug,
            "title": title,
            "description": description,
            "body": body,
            "toc": toc if toc is not None else toc_from(body),
            "hidden": hidden,
        }
    )


def toc_from(body):
    found = re.findall(r'<h2 id="([^"]+)">([^<]+)</h2>', body)
    return [(anchor, title) for anchor, title in found]


def code(text, lang=""):
    text = text.strip("\n")
    return f'<pre data-lang="{lang}"><code>{html.escape(text)}</code></pre>'


def note(title, text):
    return f'<div class="note"><b>{title}</b>{text}</div>'


def warn(title, text):
    return f'<div class="warn"><b>{title}</b>{text}</div>'


def render():
    order = [(slug, title) for _, items in NAV for slug, title in items]
    visible = [p for p in PAGES if not p["hidden"]]
    total = len(order)

    for index, item in enumerate(PAGES):
        slug = item["slug"]
        position = next((i for i, (s, _) in enumerate(order) if s == slug), None)
        if position is None:
            prev_slug = prev_title = next_slug = next_title = None
        else:
            prev_slug, prev_title = order[position - 1] if position > 0 else (None, None)
            next_slug, next_title = order[position + 1] if position + 1 < total else (None, None)

        nav_html = ['<details open><summary>Contents</summary><div class="side-body">']
        for section, items in NAV:
            nav_html.append(f'<h2>{section}</h2>')
            for href, label in items:
                current = ' aria-current="page"' if href == slug else ""
                nav_html.append(f'<a href="{href}"{current}>{label}</a>')
        nav_html.append("</div></details>")

        toc_html = ""
        if item["toc"]:
            toc_html = '<aside class="toc"><h2>On this page</h2>' + "".join(
                f'<a href="#{anchor}">{title}</a>' for anchor, title in item["toc"]
            ) + "</aside>"

        next_html = '<div class="next">'
        if prev_slug:
            next_html += f'<a href="{prev_slug}">← {prev_title}</a>'
        else:
            next_html += "<span></span>"
        if next_slug:
            next_html += f'<a href="{next_slug}">{next_title} →</a>'
        next_html += "</div>"

        document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{item["title"]} — sound_rs docs</title>
<meta name="description" content="{html.escape(item["description"])}">
<link rel="canonical" href="{SITE}{slug}">
<meta property="og:title" content="{item["title"]} — sound_rs docs">
<meta property="og:description" content="{html.escape(item["description"])}">
<meta property="og:type" content="article">
<link rel="stylesheet" href="assets/style.css">
<script src="assets/app.js" defer></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="top">
  <a class="brand" href="index.html">sound_rs</a>
  <span class="ver">v{VERSION}</span>
  <nav aria-label="External">
    <button type="button" data-search>Search <kbd>/</kbd></button>
    <a href="{PYPI}">PyPI</a>
    <a href="{REPO}">GitHub</a>
  </nav>
</header>
<div class="shell">
  <nav class="side" aria-label="Documentation">{"".join(nav_html)}</nav>
  <main id="main">
{item["body"]}
{next_html}
  </main>
  {toc_html}
</div>
<footer>
  <span>sound_rs v{VERSION} — MIT</span>
  <a href="{REPO}">Source</a>
  <a href="{PYPI}">PyPI</a>
  <a href="{REPO}/issues">Issues</a>
  <span class="spacer">Built without a build step</span>
</footer>
<dialog class="search" aria-label="Search documentation">
  <input type="search" placeholder="Search docs — player, seek, flac, numpy…" aria-label="Search">
  <ul></ul>
</dialog>
</body>
</html>
"""
        with open(os.path.join(HERE, slug), "w") as handle:
            handle.write(document)

    with open(os.path.join(HERE, "assets", "style.css"), "w") as handle:
        handle.write(CSS.strip() + "\n")
    with open(os.path.join(HERE, "assets", "app.js"), "w") as handle:
        handle.write(JS.strip() + "\n")

    entries = []
    for item in [p for p in PAGES if not p["hidden"]]:
        plain = re.sub(r"<[^>]+>", " ", item["body"])
        plain = re.sub(r"\s+", " ", html.unescape(plain)).strip()
        headings = " ".join(title for _, title in item["toc"])
        entries.append(
            {
                "title": item["title"],
                "url": item["slug"],
                "section": next(
                    section for section, items in NAV if any(s == item["slug"] for s, _ in items)
                ),
                "body": plain,
                "headings": headings,
            }
        )
    with open(os.path.join(HERE, "search.json"), "w") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=0)

    urls = "\n".join(f"  <url><loc>{SITE}{p['slug']}</loc></url>" for p in visible)
    with open(os.path.join(HERE, "sitemap.xml"), "w") as handle:
        handle.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"{urls}\n</urlset>\n"
        )
    with open(os.path.join(HERE, "robots.txt"), "w") as handle:
        handle.write(f"User-agent: *\nAllow: /\nSitemap: {SITE}sitemap.xml\n")


if __name__ == "__main__":
    import sys

    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
    # `_content` does `from _build import page`, so let it find this very module
    # instead of importing a second copy with its own empty PAGES list.
    sys.modules["_build"] = sys.modules["__main__"]
    import _content  # noqa: F401  (registers every page)

    render()
    print(f"built {len(PAGES)} pages into {HERE}")
