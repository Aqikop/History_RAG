"""
Orchestrates the roman-rag pipeline stages:

    scrape  -> data/interim/events.json
    resolve -> data/interim/events_resolved.json
    chunks  -> data/processed/chunks.json

Usage:
    python run_pipeline.py [--from-stage STAGE] [--to-stage STAGE]
                            [--dry-run] [--log-level LEVEL]

    STAGE is one of: scrape, resolve, chunks

Examples:
    python run_pipeline.py
        Runs all three stages start to finish.

    python run_pipeline.py --from-stage resolve
        Skips scraping, resumes from the existing data/interim/events.json.

    python run_pipeline.py --to-stage resolve
        Runs scrape + resolve, stops before chunking.

    python run_pipeline.py --dry-run
        Logs what each stage would do (row/chunk counts, API call
        estimates) without fetching, calling APIs, or writing any files.
"""

import argparse
import logging

import scrape_roman_timeline
import resolve_wikidata
import build_chunks

log = logging.getLogger("run_pipeline")

STAGES = ["scrape", "resolve", "chunks"]


def run_pipeline(from_stage="scrape", to_stage="chunks", dry_run=False):
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"Stages must be one of {STAGES}")
    if STAGES.index(from_stage) > STAGES.index(to_stage):
        raise ValueError(f"--from-stage '{from_stage}' comes after --to-stage '{to_stage}'")

    active = STAGES[STAGES.index(from_stage):STAGES.index(to_stage) + 1]
    log.info(f"Running stages: {' -> '.join(active)}" + (" [dry-run]" if dry_run else ""))

    if "scrape" in active:
        log.info("=== Stage: scrape ===")
        scrape_roman_timeline.run(dry_run=dry_run)

    if "resolve" in active:
        log.info("=== Stage: resolve ===")
        # In a dry-run, events.json may not exist yet if scrape was also
        # dry-run (or skipped via --from-stage resolve on a fresh clone).
        # resolve_wikidata.run() reads it directly, so fall back to a
        # bare log message if it's missing rather than crashing the
        # whole dry-run preview.
        if dry_run and not _exists(resolve_wikidata.DEFAULT_INPUT):
            log.info(f"[dry-run] {resolve_wikidata.DEFAULT_INPUT} does not exist yet "
                      f"(would be produced by the scrape stage) -- skipping resolve preview.")
        else:
            resolve_wikidata.run(dry_run=dry_run)

    if "chunks" in active:
        log.info("=== Stage: chunks ===")
        if dry_run and not _exists(build_chunks.DEFAULT_INPUT):
            log.info(f"[dry-run] {build_chunks.DEFAULT_INPUT} does not exist yet "
                      f"(would be produced by the resolve stage) -- skipping chunks preview.")
        else:
            build_chunks.run(dry_run=dry_run)

    log.info("Pipeline complete." if not dry_run else "Dry-run complete.")


def _exists(path_str):
    from pathlib import Path
    return Path(path_str).exists()


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-stage", default="scrape", choices=STAGES,
                         help="First stage to run (default: scrape)")
    parser.add_argument("--to-stage", default="chunks", choices=STAGES,
                         help="Last stage to run (default: chunks)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Log what each stage would do without fetching, "
                              "calling APIs, or writing files")
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    run_pipeline(from_stage=args.from_stage, to_stage=args.to_stage, dry_run=args.dry_run)


if __name__ == "__main__":
    main()