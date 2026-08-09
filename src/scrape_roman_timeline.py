"""
Output:
    roman_timeline.json
    roman_timeline.csv
    raw_wikitext.txt   (debug dump of the raw source, for inspection)
"""

import json
import csv
import re
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
PAGE_TITLE = "Timeline of Roman history"

# Wikimedia requires a descriptive User-Agent on all API requests, or it
# returns 403. Feel free to replace the email with your own contact info.
HEADERS = {
    "User-Agent": "RAGDatasetBuilder/1.0 (https://example.com; you@example.com)"
}

# Matches the article's custom year template: {{dr|y|y|-754|0|ysa}}
# 3rd positional parameter is the year: negative = BC, positive/zero = AD.
DR_TEMPLATE_RE = re.compile(r"\{\{dr\|[^|}]*\|[^|}]*\|(-?\d+)\|[^}]*\}\}")

HEADER_LABELS = {"year", "date", "event", "events"}


def fetch_wikitext(page_title: str) -> str:
    """Fetch raw wikitext for a page via the Wikipedia API."""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["parse"]["wikitext"]


def expand_dr_template(text: str) -> str:
    """Convert {{dr|y|y|-754|0|ysa}} -> '754 BC', {{dr|y|y|98|0|ysa}} -> 'AD 98'."""
    def _sub(m):
        year = int(m.group(1))
        if year < 0:
            return f"{abs(year)} BC"
        return f"AD {year}"
    return DR_TEMPLATE_RE.sub(_sub, text)


def strip_cell_attributes(cell: str) -> str:
    """
    Wikitable cells can carry formatting attributes before the actual
    content, separated by a single '|', e.g.:
        rowspan="2" valign="top" | 753 BC
    Strip the attribute prefix and keep only the real value. Must not
    touch wiki links like [[Page|Display]] (starts with '[[').
    """
    cell = cell.strip()
    while "|" in cell and not cell.startswith("[["):
        prefix, _, rest = cell.partition("|")
        if "=" in prefix:
            cell = rest.strip()
        else:
            break
    return cell


def strip_wiki_markup(text: str) -> str:
    """Remove wiki links/templates/refs, keep plain readable text."""
    text = expand_dr_template(text)
    text = strip_cell_attributes(text)
    # Remove <ref>...</ref> footnotes (and self-closing <ref .../>)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref.*?</ref>", "", text, flags=re.DOTALL)
    # [[Link|Display]] -> Display ; [[Link]] -> Link
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # '''bold''' / ''italic'' -> plain
    text = re.sub(r"'{2,}", "", text)
    # Any remaining {{template|...}} -> drop (citation/formatting templates)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip(" |\n\t")


def split_cells(content: str):
    """Split a cell-line's content on '||' (and '!!', seen interchangeably
    in this article's header rows) into individual raw cell strings."""
    parts = re.split(r"\|\||!!", content)
    return [p.strip() for p in parts]


def parse_timeline(wikitext: str):
    """
    Parse the wikitext table rows into year/date/event records.

    Handles:
      - Multiple separate tables (one per century section), each with its
        own repeated header row.
      - Year cells written as {{dr|y|y|N|0|ysa}} (older sections) or as
        plain [[1071]] wiki-links (later sections).
      - rowspan cells: a row missing its year cell inherits the previous
        row's year (forward-fill).
      - Multi-line cell content (continuation lines with no leading '|').
      - "|-" row separators that have a cell packed onto the same line
        (e.g. "|-| {{dr|...}} || ...") as well as stray non-table text
        accidentally glued onto a "|-" line (article vandalism / typos),
        which is simply discarded.
    """
    rows = []
    pending_cells = []
    in_table = False

    def flush_row():
        nonlocal pending_cells
        if pending_cells:
            rows.append(pending_cells)
            pending_cells = []

    def process_cell_line(line):
        nonlocal pending_cells
        content = line[1:]
        for cell in split_cells(content):
            cell = cell.lstrip("|").strip()
            pending_cells.append(cell)

    lines = wikitext.split("\n")
    for raw_line in lines:
        line = raw_line.strip()

        if line.startswith("{|"):
            in_table = True
            flush_row()
            continue

        if line.startswith("|}"):
            flush_row()
            in_table = False
            continue

        if not in_table:
            continue

        if line.startswith("|-"):
            flush_row()
            rest = line[2:]
            # A cell can be packed onto the same line as the row separator,
            # e.g. "|-| {{dr|...}} || ...". Find the first '|' that starts
            # real cell content; anything before it (row attributes, or
            # stray junk from article vandalism) is discarded.
            idx = rest.find("|")
            if idx != -1:
                process_cell_line(rest[idx:])
            continue

        if line.startswith("!") or line.startswith("|"):
            process_cell_line(line)
            continue

        # Continuation of the previous cell's text (multi-line cell)
        if pending_cells and line:
            pending_cells[-1] = (pending_cells[-1] + " " + line).strip()

    flush_row()

    current_year = None
    parsed = []
    for cells in rows:
        cells = [strip_wiki_markup(c) for c in cells]

        # Skip real header rows (all non-empty cells are column titles)
        non_empty = [c.lower() for c in cells if c]
        if non_empty and all(c in HEADER_LABELS for c in non_empty):
            continue

        if len(cells) >= 3:
            year, date, event = cells[0], cells[1], " ".join(cells[2:]).strip()
        elif len(cells) == 2:
            year, date, event = "", cells[0], cells[1]
        elif len(cells) == 1:
            year, date, event = "", "", cells[0]
        else:
            continue

        if year:
            current_year = year
        else:
            year = current_year

        event = event.strip()
        if not event:
            continue

        parsed.append({
            "year": year or "",
            "date": date or "",
            "event": event,
        })

    return parsed


def main():
    print(f"Fetching wikitext for '{PAGE_TITLE}'...")
    wikitext = fetch_wikitext(PAGE_TITLE)

    with open("raw_wikitext.txt", "w", encoding="utf-8") as f:
        f.write(wikitext)
    print("Saved raw_wikitext.txt (for debugging table structure)")

    print("Parsing timeline table...")
    events = parse_timeline(wikitext)
    print(f"Parsed {len(events)} events.")

    with open("roman_timeline.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    with open("roman_timeline.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "date", "event"])
        writer.writeheader()
        writer.writerows(events)

    print("Saved roman_timeline.json and roman_timeline.csv")


if __name__ == "__main__":
    main()