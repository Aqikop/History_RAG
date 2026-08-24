"""
Resolves wiki-link titles in events to Wikidata QIDs, using Wikipedia's
pageprops API to map article titles to their corresponding Wikidata entities.

Usage:
    pip install requests
    python resolve_wikidata.py [--input PATH] [--output PATH]
                                [--unresolved-output PATH]
                                [--dry-run] [--log-level LEVEL]

Input (default path):
    data/interim/events.json   (from scrape_roman_timeline.py)

Output (default paths):
    data/interim/events_resolved.json       (events with linked_entity_ids added)
    data/interim/unresolved_titles.txt      (titles with no Wikidata match)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("resolve_wikidata")


def _detect_project_root() -> Path:
    """
    Default data paths must resolve to the same place regardless of which
    directory you happen to run the script FROM.
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
HEADERS = {
    "User-Agent": "RAGDatasetBuilder/1.0 (https://example.com; you@example.com)"
}
BATCH_SIZE = 50

DEFAULT_INPUT = str(PROJECT_ROOT / "data" / "interim" / "events.json")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "data" / "interim" / "events_resolved.json")
DEFAULT_UNRESOLVED_OUTPUT = str(PROJECT_ROOT / "data" / "interim" / "unresolved_titles.txt")

# Known bad wiki-link titles (typos in Wikipedia's wikitext) that will never
# resolve via the API. Add entries here as {wrong_title: correct_qid} when a
# genuinely-real subject shows up in unresolved_titles.txt after manual
# investigation.
MANUAL_QID_OVERRIDES = {
    "Epaphropditus (freedman of Nero)": "Q289327",  # typo for "Epaphroditus (freedman of Nero)"
}


def chunked(items, size):
    """
    Divide items into chunks of the given size.
    Yields successive chunks (as lists) from items until exhausted.
    """
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _query_batch(batch, attempt_label=""):
    """
    Run one pageprops query for a batch of Wikipedia titles.
    
    Returns the parsed 'query' dict from the MediaWiki API response.
    Raises RuntimeError (or the actual exception) after exhausting retries;
    failures are NOT silently swallowed.
    
    Args:
        batch: list of Wikipedia article titles to query
        attempt_label: optional string for logging/debugging (e.g. "pass_1")
    
    Returns:
        dict: the 'query' key from the API response (contains pages → QID metadata)
    
    Raises:
        RuntimeError: if all retry attempts fail
    """
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
            log.debug(f"Batch{attempt_label} attempt {attempt + 1} failed: {e}")
            time.sleep(1.0 * (attempt + 1))
    
    raise RuntimeError(f"Batch{attempt_label} failed after 3 attempts: {last_err}")


def _resolve_one_batch(batch, results):
    """
    Query one batch of titles and write resolved QIDs directly into `results`.
    
    Builds a forward map (original → final title → QID) per-title, so that
    multiple original titles redirecting/normalizing to the SAME final page
    (e.g. 'Ottoman empire' and 'Ottoman Empire' both → 'Ottoman Empire')
    each correctly get the resolved QID instead of one silently overwriting
    the other.
    
    Args:
        batch: list of Wikipedia article titles
        results: dict to mutate in place, mapping title → QID (or None)
    
    Returns:
        None (mutates results in place)
    """
    query = _query_batch(batch)
    
    # Build forward maps: original → normalized title → final (after redirects)
    norm_map = {n["from"]: n["to"] for n in query.get("normalized", [])}
    redir_map = {r["from"]: r["to"] for r in query.get("redirects", [])}
    
    # Pages come back as a dict keyed by page ID (not a list)
    pages_dict = query.get("pages", {})
    pages_by_title = {p["title"]: p for p in pages_dict.values()}
    
    for original in batch:
        current = original
        # Apply normalization (e.g. underscores → spaces)
        if current in norm_map:
            current = norm_map[current]
        # Follow redirect chains (bounded to avoid loops)
        for _ in range(5):
            if current in redir_map:
                current = redir_map[current]
            else:
                break
        # Look up the final page
        page = pages_by_title.get(current)
        qid = page.get("pageprops", {}).get("wikibase_item") if page else None
        results[original] = qid


def resolve_titles_to_qids(titles):
    """
    Given a list of Wikipedia article titles, return a dict mapping
    title → Wikidata QID (or None if the article has no linked item,
    doesn't exist, or is a redlink).
    
    Uses a two-pass strategy:
      1. Batch lookups (fast, ~50 titles/request)
      2. Anything still unresolved after pass 1 gets re-checked
         individually (batch of 1), to rule out cross-title collisions
         or transient issues before concluding a title is genuinely
         unresolved
    
    Args:
        titles: list of Wikipedia article title strings
    
    Returns:
        dict[str, Optional[str]]: title → QID mapping (None for unresolved)
    """
    result = {t: None for t in titles}
    
    # Apply manual overrides for known-bad titles before any API calls
    for title, qid in MANUAL_QID_OVERRIDES.items():
        if title in result:
            result[title] = qid
    
    # Batch pass: resolve everything that isn't already overridden
    titles_to_resolve = [t for t, q in result.items() if q is None]
    for batch in chunked(titles_to_resolve, BATCH_SIZE):
        _resolve_one_batch(batch, result)
        time.sleep(0.2)  # be polite to the API
    
    # Individual re-check pass: anything still None gets one more chance
    unresolved = [t for t, q in result.items() if q is None]
    if unresolved:
        log.info(f"Re-checking {len(unresolved)} unresolved titles individually...")
        for title in unresolved:
            _resolve_one_batch([title], result)
            time.sleep(0.2)
    
    return result


def run(input_path=DEFAULT_INPUT, output=DEFAULT_OUTPUT,
        unresolved_output=DEFAULT_UNRESOLVED_OUTPUT, dry_run=False):
    """
    Resolve all wiki_links in input events JSON to Wikidata QIDs.
    
    Reads events from input_path (expected to have a 'links' field —
    list of [target, label] tuples from scrape_roman_timeline.py output),
    resolves each target to a Wikidata QID, and writes augmented events
    to output (with a new 'linked_entity_ids' field added).
    
    Titles that don't resolve to any Wikidata item are written to
    unresolved_output (one per line).
    
    Args:
        input_path: path to input events.json
        output: path to write resolved events.json
        unresolved_output: path to write unresolved_titles.txt
        dry_run: if True, don't write files but still resolve
    
    Returns:
        str: path to output JSON on success, or None if dry_run=True
    """
    log.info(f"Reading events from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        events = json.loads(f.read())
    
    # Extract all unique linked titles from events
    unique_titles = sorted({
        target
        for event in events
        for target, _ in event.get("links", [])
    })
    log.info(f"Resolving {len(unique_titles)} unique linked titles to Wikidata IDs...")
    
    title_to_qid = resolve_titles_to_qids(unique_titles)
    
    # Augment events with resolved QIDs
    for event in events:
        linked_entity_ids = sorted({
            qid
            for target, _ in event.get("links", [])
            if (qid := title_to_qid.get(target))
        })
        event["linked_entity_ids"] = linked_entity_ids
    
    if dry_run:
        resolved_count = sum(1 for q in title_to_qid.values() if q)
        log.info(f"[dry-run] Would write {len(events)} resolved events to {output} "
                f"(resolved {resolved_count}/{len(unique_titles)} titles)")
        return None
    
    # Write resolved events
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    log.info(f"Wrote resolved events to {output_path}")
    
    # Write unresolved titles
    unresolved = sorted(t for t, q in title_to_qid.items() if q is None)
    unresolved_path = Path(unresolved_output)
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(unresolved_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unresolved))
    log.info(f"Wrote {len(unresolved)} unresolved titles to {unresolved_path}")
    
    return str(output_path).replace("\\", "/")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Path to input events.json (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Path to write resolved events.json (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--unresolved-output", default=DEFAULT_UNRESOLVED_OUTPUT,
                        help=f"Path to write unresolved titles (default: {DEFAULT_UNRESOLVED_OUTPUT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write files; log what would happen (still resolves)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    run(input_path=args.input, output=args.output, unresolved_output=args.unresolved_output,
        dry_run=args.dry_run)


if __name__ == "__main__":
    main()