"""
Resolves the wiki_links captured in events.json to Wikidata QIDs, using
Wikipedia's pageprops API (a direct, unambiguous Wikipedia-article ->
Wikidata-item mapping -- no fuzzy search/disambiguation needed, since
each link already points at one specific Wikipedia article).

Usage:
    pip install requests
    python resolve_wikidata.py [--input PATH] [--output PATH]
                                [--unresolved-output PATH] [--dry-run]
                                [--log-level LEVEL]

Output (default paths):
    data/interim/events_resolved.json   -- events with resolved Wikidata IDs
    data/interim/unresolved_titles.txt  -- titles with no Wikidata item
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("resolve_wikidata")


def _detect_project_root() -> Path:
    """See scrape_roman_timeline._detect_project_root() for rationale --
    default data paths must not depend on the current working directory."""
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
HEADERS = {
    "User-Agent": "RAGDatasetBuilder/1.0 (https://example.com; you@example.com)"
}
BATCH_SIZE = 50  # MediaWiki API limit per request for non-bot accounts

DEFAULT_INPUT = str(PROJECT_ROOT / "data" / "interim" / "events.json")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "data" / "interim" / "events_resolved.json")
DEFAULT_UNRESOLVED_OUTPUT = str(PROJECT_ROOT / "data" / "interim" / "unresolved_titles.txt")

# Known bad wiki-link titles that will never resolve via the API -- e.g.
# typos in the live Wikipedia article's own wikitext (the link target
# itself is misspelled, so no page/Wikidata item exists under that exact
# title). Add entries here as {wrong_title: correct_qid} whenever a
# genuinely-real subject shows up in unresolved_titles.txt after manual
# investigation. Look the correct QID up at wikidata.org.
MANUAL_QID_OVERRIDES = {
    "Epaphropditus (freedman of Nero)": "Q289327",  # typo for "Epaphroditus (freedman of Nero)"
}


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _query_batch(batch, attempt_label=""):
    """Run one pageprops query for a batch of titles, with retries on
    transient failures. Returns the parsed 'query' dict, or raises after
    exhausting retries (so failures are visible, not silently swallowed)."""
    params = {
        "action": "query",
        "prop": "pageprops",
        "ppprop": "wikibase_item",
        "titles": "|".join(batch),
        "redirects": 1,
        "format": "json",
        "formatversion": "2",
    }
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "query" not in data:
                raise ValueError(f"Unexpected API response (no 'query' key): {data}")
            return data["query"]
        except Exception as e:
            last_err = e
            log.warning(f"  Batch{attempt_label} attempt {attempt + 1} failed: {e}")
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Batch{attempt_label} failed after 3 attempts: {last_err}")


def _resolve_one_batch(batch, results):
    """Query one batch and write resolved QIDs directly into `results`.
    Builds a forward map (original -> final title) per-title, rather than
    a single reverse map keyed by final title, so that multiple original
    titles redirecting/normalizing to the SAME final page (e.g. 'Ottoman
    empire' and 'Ottoman Empire' both -> 'Ottoman Empire') each correctly
    get the resolved QID instead of one silently overwriting the other.
    """
    query = _query_batch(batch)

    norm_map = {n["from"]: n["to"] for n in query.get("normalized", [])}
    redir_map = {r["from"]: r["to"] for r in query.get("redirects", [])}
    pages_by_title = {p["title"]: p for p in query.get("pages", [])}

    for original in batch:
        current = original
        if current in norm_map:
            current = norm_map[current]
        # Follow redirect chains (bounded, to avoid any pathological loop)
        for _ in range(5):
            if current in redir_map:
                current = redir_map[current]
            else:
                break
        page = pages_by_title.get(current)
        qid = page.get("pageprops", {}).get("wikibase_item") if page else None
        results[original] = qid


def resolve_titles_to_qids(titles):
    """
    Given a list of Wikipedia article titles, return a dict mapping
    title -> Wikidata QID (or None if the article genuinely has no
    linked item, doesn't exist, or is a redlink).

    Runs two passes:
      1. Batched lookups (fast, ~50 titles/request).
      2. Anything still unresolved after pass 1 gets re-checked
         individually (batch of 1), to rule out cross-title collisions
         or transient issues within a shared batch before concluding a
         title is genuinely unresolved.
    """
    result = {t: None for t in titles}

    for batch in chunked(titles, BATCH_SIZE):
        _resolve_one_batch(batch, result)
        time.sleep(0.2)  # be polite to the API

    unresolved = [t for t, q in result.items() if not q]
    if unresolved:
        log.info(f"Re-checking {len(unresolved)} unresolved titles individually...")
        for title in unresolved:
            _resolve_one_batch([title], result)
            time.sleep(0.2)

    return result


def run(input_path=DEFAULT_INPUT, output=DEFAULT_OUTPUT,
        unresolved_output=DEFAULT_UNRESOLVED_OUTPUT, dry_run=False):
    """
    Resolve all wiki_links in the events at `input_path` to Wikidata QIDs.
    Returns the path to the resolved events JSON (as a string), or None
    if dry_run.
    """
    with open(input_path, encoding="utf-8") as f:
        events = json.load(f)

    unique_titles = sorted({
        link["wiki_title"]
        for event in events
        for link in event["wiki_links"]
    })

    if dry_run:
        log.info(f"[dry-run] Would resolve {len(unique_titles)} unique linked titles "
                  f"to Wikidata IDs (~{-(-len(unique_titles) // BATCH_SIZE)} batched "
                  f"API requests), then write results to {output} / {unresolved_output}.")
        return None

    log.info(f"Resolving {len(unique_titles)} unique linked titles to Wikidata IDs...")
    title_to_qid = resolve_titles_to_qids(unique_titles)

    override_count = 0
    for title, qid in MANUAL_QID_OVERRIDES.items():
        if title in title_to_qid and not title_to_qid[title]:
            title_to_qid[title] = qid
            override_count += 1
    if override_count:
        log.info(f"Applied {override_count} manual override(s) for known bad titles.")

    resolved_count = sum(1 for q in title_to_qid.values() if q)
    log.info(f"Resolved {resolved_count} / {len(unique_titles)} titles.")

    for event in events:
        for link in event["wiki_links"]:
            link["wikidata_id"] = title_to_qid.get(link["wiki_title"])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    unresolved = sorted(t for t, q in title_to_qid.items() if not q)
    unresolved_path = Path(unresolved_output)
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_path.write_text("\n".join(unresolved), encoding="utf-8")

    log.info(f"{len(unresolved)} titles had no Wikidata item (see {unresolved_path})")
    log.info(f"Saved {output_path}")
    return str(output_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                         help=f"Path to events JSON to read (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Path to write resolved events JSON (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--unresolved-output", default=DEFAULT_UNRESOLVED_OUTPUT,
                         help=f"Path to write unresolved titles list "
                              f"(default: {DEFAULT_UNRESOLVED_OUTPUT})")
    parser.add_argument("--dry-run", action="store_true",
                         help="Don't call the API or write files; log what would happen")
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    run(input_path=args.input, output=args.output,
        unresolved_output=args.unresolved_output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()