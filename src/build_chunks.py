"""
Builds embedding-ready chunks from resolved events by merging all events
that share the same year into a single chunk, and attaching
retrieval-useful metadata (era, dates, linked Wikidata entities).

Usage:
    python build_chunks.py [--input PATH] [--output PATH] [--dry-run]
                            [--log-level LEVEL]

Output (default path):
    data/processed/chunks.json
"""

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("build_chunks")


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

SOURCE_NAME = "Wikipedia: Timeline of Roman history"
SOURCE_URL = "https://en.wikipedia.org/wiki/Timeline_of_Roman_history"

DEFAULT_INPUT = str(PROJECT_ROOT / "data" / "interim" / "events_resolved.json")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "data" / "processed" / "chunks.json")


def build_chunks(events):
    # Group events by year_normalized (a stable, sortable, unambiguous
    # key -- unlike the display "year" string, which differs in format
    # across the BC/AD boundary and across sections that use "AD 98" vs
    # bare "1071").
    groups = {}
    order = []
    for e in events:
        key = e["year_normalized"]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)

    chunks = []
    for i, key in enumerate(sorted(order, key=lambda k: (k is None, k))):
        group = groups[key]

        year_display = group[0]["year"]
        era = group[0]["era"]

        dates = []
        for e in group:
            if e["date"] and e["date"] not in dates:
                dates.append(e["date"])

        # Plain concatenated text (for citation/display) -- each event
        # optionally prefixed with its specific date when the year has
        # multiple dated sub-events, so a merged chunk doesn't lose which
        # sentence happened when.
        multiple_dates = len(dates) > 1
        sentence_parts = []
        for e in group:
            if multiple_dates and e["date"]:
                sentence_parts.append(f"({e['date']}) {e['event']}")
            else:
                sentence_parts.append(e["event"])
        text = " ".join(sentence_parts)

        # Embedding text: prepend year/era context so date-based queries
        # ("what happened in the 8th century BC") retrieve well even
        # though the raw event sentences rarely restate the year.
        header = year_display
        if era:
            header += f", {era}"
        embedding_text = f"{header}: {text}"

        # Aggregate linked entities across all merged events, de-duplicated
        # by wiki_title (the same entity, e.g. "Rome", often recurs).
        seen_titles = set()
        linked_entities = []
        for e in group:
            for link in e["wiki_links"]:
                if link["wiki_title"] in seen_titles:
                    continue
                seen_titles.add(link["wiki_title"])
                linked_entities.append(link)
        linked_entity_ids = sorted({
            l["wikidata_id"] for l in linked_entities if l["wikidata_id"]
        })

        chunks.append({
            "chunk_id": f"chunk_{i:04d}",
            "year": year_display,
            "year_normalized": key,
            "era": era,
            "dates": dates,
            "event_count": len(group),
            "text": text,
            "embedding_text": embedding_text,
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "linked_entities": linked_entities,
            "linked_entity_ids": linked_entity_ids,
        })

    return chunks


def run(input_path=DEFAULT_INPUT, output=DEFAULT_OUTPUT, dry_run=False):
    """
    Build chunks from the resolved events at `input_path`. Returns the
    path to the chunks JSON (as a string), or None if dry_run.
    """
    with open(input_path, encoding="utf-8") as f:
        events = json.load(f)

    if dry_run:
        distinct_years = len({e["year_normalized"] for e in events})
        log.info(f"[dry-run] Would merge {len(events)} events into ~{distinct_years} "
                  f"chunks (one per distinct year) and write to {output}.")
        return None

    chunks = build_chunks(events)

    total_events = sum(c["event_count"] for c in chunks)
    merged = sum(1 for c in chunks if c["event_count"] > 1)
    log.info(f"Built {len(chunks)} chunks from {total_events} events "
              f"({merged} chunks merged multiple events).")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {output_path}")
    return str(output_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                         help=f"Path to resolved events JSON to read (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Path to write chunks JSON (default: {DEFAULT_OUTPUT})")
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