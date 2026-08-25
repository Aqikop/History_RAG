"""
Builds embedding-ready chunks from resolved events by grouping all events
that share the same year_normalized into a single chunk, and merging their
linked entity metadata.

Usage:
    python build_chunks.py [--input PATH] [--output PATH]
                            [--dry-run] [--log-level LEVEL]

Input (default path):
    data/interim/events_resolved.json   (from resolve_wikidata.py)

Output (default path):
    data/processed/chunks.json
"""

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("build_chunks")


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

DEFAULT_INPUT = str(PROJECT_ROOT / "data" / "interim" / "events_resolved.json")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "data" / "processed" / "chunks.json")

SOURCE_NAME = "Wikipedia: Timeline of Roman history"
SOURCE_URL = "https://en.wikipedia.org/wiki/Timeline_of_Roman_history"


def build_chunks(events):
    """
    Build embedding-ready chunks by grouping events by year_normalized.
    
    Groups all events sharing the same year_normalized value into a single
    chunk. For chunks representing multiple events, linked_entity_ids are
    merged (unioned) across all events in the group, with no duplication.
    
    Each chunk is sorted chronologically by year_normalized (ascending).
    
    Args:
        events: list of event dicts, each with:
            - year (str): display year, e.g. "753 BC"
            - year_normalized (int): astronomical year (-753, 0, 1, etc.)
            - era (str): century/era name
            - date (str): day/month within the year
            - event (str): event description
            - linked_entity_ids (dict or list): entity → QID mapping
    
    Returns:
        list of chunk dicts, each with:
            - year_normalized (int): grouping key
            - year (str): display year from first event in group
            - era (str): era from first event in group
            - dates (list): list of unique non-empty dates from merged events
            - events (list): all events in this year (as dicts)
            - linked_entity_ids (dict): merged union of all entities from events
            - source (str): SOURCE_NAME constant
            - source_url (str): SOURCE_URL constant
    """
    # Group events by year_normalized
    groups = {}
    order = []
    for event in events:
        key = event.get("year_normalized", event.get("year"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(event)
    
    chunks = []
    for key in sorted(order, key=lambda k: (k is None, k)):
        group = groups[key]
        
        year = group[0].get("year_display", group[0].get("year", ""))
        era = group[0]["era"]
        
        # Collect unique non-empty dates
        dates = []
        for event in group:
            d = event.get("date", "")
            if d and d not in dates:
                dates.append(d)
        
        # Merge linked_entity_ids across all events in the group
        # Handle both dict (from test fixture) and list (from refactored resolve_wikidata)
        merged_ids = {}
        for event in group:
            entity_ids = event.get("linked_entity_ids", {})
            if isinstance(entity_ids, dict):
                # Old schema: dict of title -> QID
                merged_ids.update(entity_ids)
            elif isinstance(entity_ids, list):
                # New schema from refactored resolve_wikidata: sorted list of QIDs
                # Convert to dict for output (use QID as both key and value)
                for qid in entity_ids:
                    merged_ids[qid] = qid
        
        chunks.append({
            "year_normalized": key,
            "year": year,
            "era": era,
            "dates": dates,
            "events": group,
            "linked_entity_ids": merged_ids,
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
        })
    
    return chunks


def run(input_path=DEFAULT_INPUT, output=DEFAULT_OUTPUT, dry_run=False):
    """
    Read resolved events, build chunks, and write output.
    
    Args:
        input_path: path to input events_resolved.json
        output: path to write chunks.json
        dry_run: if True, don't write files but log what would happen
    
    Returns:
        str: output path on success, or None if dry_run=True
    """
    log.info(f"Reading events from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        events = json.loads(f.read())
    
    log.info("Building chunks...")
    chunks = build_chunks(events)
    log.info(f"Built {len(chunks)} chunks from {len(events)} events.")
    
    if dry_run:
        log.info(f"[dry-run] Would write chunks to {output}")
        return None
    
    # Write chunks
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    log.info(f"Wrote chunks to {output_path}")
    
    return str(output_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Path to input events_resolved.json (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Path to write chunks.json (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write files; log what would happen")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    run(input_path=args.input, output=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()