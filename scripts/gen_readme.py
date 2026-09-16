#!/usr/bin/env python3
"""Generate README.md from data/papers.yaml + data/taxonomy.yaml + data/parts/*.md.

    python scripts/gen_readme.py            # write README.md
    python scripts/gen_readme.py --check    # exit 1 if README.md is stale (for CI)
"""
import io, os, sys, re, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")


def load():
    papers = yaml.safe_load(io.open(os.path.join(D, "papers.yaml"), encoding="utf-8"))["papers"]
    tax = yaml.safe_load(io.open(os.path.join(D, "taxonomy.yaml"), encoding="utf-8"))["groups"]
    return papers, tax


def part(name):
    return io.open(os.path.join(D, "parts", name), encoding="utf-8").read().rstrip() + "\n"


def anchor(text):
    """GitHub's heading-anchor rule."""
    a = text.lower()
    a = re.sub(r"[^\w\s-]", "", a)
    return re.sub(r"\s+", "-", a.strip())


def link(p):
    """Primary link for the title cell."""
    if p.get("arxiv"):
        return f"https://arxiv.org/abs/{p['arxiv']}"
    return p.get("site") or p.get("code") or ""


def stars(n):
    return "●" * int(n) + "○" * (5 - int(n))


def notes(p, by_id):
    """Right-hand cell: prerequisites and reading budget."""
    bits = []
    pre = p.get("prereq") or []
    if pre:
        # ids, not titles: short, unambiguous, and greppable in papers.yaml
        bits.append("after " + ", ".join(f"`{q}`" for q in pre))
    bits.append(f"{stars(p['difficulty'])} · {p['minutes']}min")
    return " · ".join(bits)


def table(rows, by_id):
    out = ["| Paper | Year | What it contributes | Prereq · Difficulty · Time | Code |",
           "| :--- | :---: | :--- | :--- | :---: |"]
    for p in rows:
        u = link(p)
        title = f"**[{p['title']}]({u})**" if u else f"**{p['title']}**"
        code = f"[repo]({p['code']})" if p.get("code") else "—"
        out.append(f"| {title} | {p['year']} | {p['desc']} | {notes(p, by_id)} | {code} |")
    return "\n".join(out)


def build():
    papers, tax = load()
    by_id = {p["id"]: p for p in papers}
    for p in papers:
        p["section_title"] = p["section"]

    body, toc = [], ["## Contents", ""]

    for g in tax:
        toc.append(f"- [{g['title']}](#{g['anchor']})")
        body.append(f'<a id="{g["anchor"]}"></a>\n')
        body.append(f"## {g['title']}\n")
        if g.get("intro"):
            body.append(g["intro"].strip() + "\n")
        for s in g["sections"]:
            toc.append(f"  - [{s['title']}](#{anchor(s['title'])})")
            rows = sorted((p for p in papers if p["section"] == s["key"]),
                          key=lambda p: (p["year"], p["id"]))
            body.append(f"### {s['title']}\n")
            body.append(f"*{s['blurb']}*\n")
            body.append(table(rows, by_id) + "\n")
        body.append("---\n")

    for name, title in [("90-adjacent.md", "Adjacent Areas"),
                        ("91-research-notes.md", "Research Notes"),
                        ("92-resources.md", "Resources"),
                        ("93-labs.md", "Labs & Companies"),
                        ("99-footer.md", "Contributing")]:
        toc.append(f"- [{title}](#{anchor(title)})")

    stats = (f"**{len(papers)} papers** across {sum(len(g['sections']) for g in tax)} sections · "
             f"every arXiv id verified against the arXiv API · "
             f"`prereq` edges make this a graph, not a list\n")

    doc = "\n".join([part("00-header.md"), stats, "---\n", "\n".join(toc) + "\n", "---\n",
                     "\n".join(body),
                     part("90-adjacent.md"), "\n---\n", part("91-research-notes.md"), "\n---\n",
                     part("92-resources.md"), "\n---\n", part("93-labs.md"), "\n---\n",
                     part("99-footer.md")])
    return re.sub(r"\n{4,}", "\n\n\n", doc)


if __name__ == "__main__":
    out = build()
    path = os.path.join(ROOT, "README.md")
    if "--check" in sys.argv:
        cur = io.open(path, encoding="utf-8").read()
        if cur != out:
            print("README.md is stale. Run: python scripts/gen_readme.py"); sys.exit(1)
        print("README.md is up to date."); sys.exit(0)
    io.open(path, "w", encoding="utf-8").write(out)
    print(f"README.md written ({len(out.splitlines())} lines)")
