"""
Scrapes Wikipedia's "Timeline of Roman history" article into structured
year / date / event records, and saves as both JSON and CSV.

Usage:
    pip install requests
    python scrape_roman_timeline.py [--output PATH] [--output-csv PATH]
                                     [--raw-output PATH] [--dry-run]
                                     [--log-level LEVEL]

Output (default paths):
    data/interim/events.json
    data/interim/events.csv
    data/raw/roman_timeline.wikitext   (raw source dump, for debugging)
"""

import argparse
import json
import csv
import logging
import re
from pathlib import Path

import requests

log = logging.getLogger("scrape_roman_timeline")


def _detect_project_root() -> Path:
    """
    Default data paths must resolve to the same place regardless of which
    directory you happen to run the script FROM (e.g. project root vs.
    inside src/) -- a plain relative path like "data/interim/events.json"
    is resolved against the current working directory, not the script's
    location, and silently creates a stray data/ folder wherever you
    happened to `cd` into.

    Detects the real project root by checking, relative to this script's
    own location, for an existing "data" directory in either of the two
    layouts this project uses:
      - scripts inside <root>/src/   -> data/ is a sibling of src/
      - scripts directly in <root>/  -> data/ is a sibling of the script
    Falls back to the src/ layout (this project's documented scaffold) if
    neither exists yet (e.g. first run before data/ has been created).
    """
    script_dir = Path(__file__).resolve().parent
    in_src_layout = script_dir.parent
    at_root_layout = script_dir
    if (in_src_layout / "data").exists():
        return in_src_layout
    if (at_root_layout / "data").exists():
        return at_root_layout
    return in_src_layout


PROJECT_ROOT = _detect_project_root()

WIKI_API = "https://en.wikipedia.org/w/api.php"
PAGE_TITLE = "Timeline of Roman history"

DEFAULT_OUTPUT = str(PROJECT_ROOT / "data" / "interim" / "events.json")
DEFAULT_OUTPUT_CSV = str(PROJECT_ROOT / "data" / "interim" / "events.csv")
DEFAULT_RAW_OUTPUT = str(PROJECT_ROOT / "data" / "raw" / "roman_timeline.wikitext")

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
    Strip that attribute prefix and keep only the real value.

    Must NOT touch:
      - wiki links like [[Page|Display]] (starts with '[[')
      - free-form event prose that happens to contain '=' before a '|'
        elsewhere in the text -- e.g. a named citation like
        <ref name=":0">{{cite book|last=Forsythe|first=Gary|...}}</ref>
        also matches the naive "'=' before '|'" pattern, but is NOT a
        table attribute and must be left alone for the ref/template
        stripping steps (later in strip_wiki_markup) to handle properly.

    Genuine table attributes only ever appear as the literal first
    characters of a cell, before any wiki markup -- so we only strip
    when the candidate attribute prefix contains no '[[', '{{', or '<'
    (which would indicate we've wandered into real content/citations),
    and we only strip once (real attribute prefixes are a single
    "attrs | value" split, never a chain of several).
    """
    cell = cell.strip()
    if cell.startswith("[[") or "|" not in cell:
        return cell
    prefix, _, rest = cell.partition("|")
    if "=" not in prefix:
        return cell
    if any(marker in prefix for marker in ("[[", "{{", "<")):
        return cell
    return rest.strip()


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


LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]")


def extract_links(text: str):
    """
    Pull [[Target|Display]] / [[Target]] wiki-link targets out of a cell's
    raw text, for later resolving to Wikidata IDs. Must run on text that
    still has <ref>...</ref> citation footnotes removed (citation
    templates often contain their own wikilinks, e.g. to publishers,
    which we don't want to treat as event-relevant entities).
    """
    text = strip_cell_attributes(expand_dr_template(text))
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref.*?</ref>", "", text, flags=re.DOTALL)
    links = []
    for m in LINK_RE.finditer(text):
        target = m.group(1).strip()
        display = (m.group(2) or target).strip()
        links.append({"wiki_title": target, "text": display})
    return links


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
    current_era = None

    SECTION_RE = re.compile(r"^==\s*([^=]+?)\s*==$")

    def flush_row():
        nonlocal pending_cells
        if pending_cells:
            rows.append((current_era, pending_cells))
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

        m = SECTION_RE.match(line)
        if m:
            flush_row()
            current_era = m.group(1).strip()
            in_table = False
            continue

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
    for era, raw_cells in rows:
        cells = [strip_wiki_markup(c) for c in raw_cells]

        # Skip real header rows (all non-empty cells are column titles)
        non_empty = [c.lower() for c in cells if c]
        if non_empty and all(c in HEADER_LABELS for c in non_empty):
            continue

        if len(cells) >= 3:
            year, date, event = cells[0], cells[1], " ".join(cells[2:]).strip()
            raw_event_cell = " ".join(raw_cells[2:])
        elif len(cells) == 2:
            year, date, event = "", cells[0], cells[1]
            raw_event_cell = raw_cells[1]
        elif len(cells) == 1:
            year, date, event = "", "", cells[0]
            raw_event_cell = raw_cells[0]
        else:
            continue

        if year:
            current_year = year
        else:
            year = current_year

        event = event.strip()
        if not event:
            continue

        links = extract_links(raw_event_cell)

        parsed.append({
            "year": year or "",
            "year_normalized": normalize_year(year),
            "era": era or "",
            "date": date or "",
            "event": event,
            "wiki_links": links,
        })

    return parsed


# Matches "754 BC", "AD 98", or a bare "1071" (implicitly AD, used by the
# article's later/medieval sections which drop the "AD" prefix).
YEAR_RE = re.compile(r"^(?:AD\s*)?(\d+)\s*(BC)?$", re.IGNORECASE)


def normalize_year(year_str: str):
    """
    Convert a display year string into a single signed integer using
    astronomical year numbering, so the whole timeline sorts/filters
    correctly with no BC/AD discontinuity:
        1 BC  -> 0
        2 BC  -> -1
        100 BC -> -99
        AD 1  -> 1
        AD 98 -> 98
        1071  -> 1071
    Returns None if the year string can't be parsed (e.g. empty).
    """
    if not year_str:
        return None
    m = YEAR_RE.match(year_str.strip())
    if not m:
        return None
    num = int(m.group(1))
    is_bc = bool(m.group(2))
    return -(num - 1) if is_bc else num


def run(output=DEFAULT_OUTPUT, output_csv=DEFAULT_OUTPUT_CSV,
        raw_output=DEFAULT_RAW_OUTPUT, page_title=PAGE_TITLE, dry_run=False):
    """
    Fetch + parse the timeline article, writing events JSON/CSV and a raw
    wikitext dump. Returns the path to the events JSON (as a string), or
    None if dry_run.
    """
    raw_path = Path(raw_output)

    if dry_run:
        if raw_path.exists():
            log.info(f"[dry-run] Found existing {raw_path}, parsing it to preview output "
                      f"(skipping network fetch)")
            wikitext = raw_path.read_text(encoding="utf-8")
            events = parse_timeline(wikitext)
            log.info(f"[dry-run] Would parse {len(events)} events and write to "
                      f"{output} / {output_csv}")
        else:
            log.info(f"[dry-run] Would fetch wikitext for '{page_title}' from Wikipedia, "
                      f"write raw dump to {raw_output}, parse it, and write events to "
                      f"{output} / {output_csv}. (No cached raw wikitext found to preview "
                      f"actual counts.)")
        return None

    log.info(f"Fetching wikitext for '{page_title}'...")
    wikitext = fetch_wikitext(page_title)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(wikitext, encoding="utf-8")
    log.info(f"Saved raw wikitext to {raw_path}")

    log.info("Parsing timeline table...")
    events = parse_timeline(wikitext)
    log.info(f"Parsed {len(events)} events.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    csv_rows = []
    for e in events:
        row = dict(e)
        row["wiki_links"] = "|".join(l["wiki_title"] for l in e["wiki_links"])
        csv_rows.append(row)

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["year", "year_normalized", "era", "date", "event", "wiki_links"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    log.info(f"Saved {output_path} and {output_csv_path}")
    return str(output_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Path to write events JSON (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV,
                         help=f"Path to write events CSV (default: {DEFAULT_OUTPUT_CSV})")
    parser.add_argument("--raw-output", default=DEFAULT_RAW_OUTPUT,
                         help=f"Path to write raw wikitext dump (default: {DEFAULT_RAW_OUTPUT})")
    parser.add_argument("--page-title", default=PAGE_TITLE,
                         help=f"Wikipedia page title to fetch (default: '{PAGE_TITLE}')")
    parser.add_argument("--dry-run", action="store_true",
                         help="Don't fetch or write files; log what would happen")
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    run(output=args.output, output_csv=args.output_csv, raw_output=args.raw_output,
        page_title=args.page_title, dry_run=args.dry_run)


if __name__ == "__main__":
    main()