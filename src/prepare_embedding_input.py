"""
prepare_embedding_input.py

Phase 2a: Transform nested chunks.json schema to flat embedding-ready schema.

Implemented against the contract in PHASE2_EMBEDDING_PLAN.md:
- Every chunk produces exactly one flat record (conservation)
- ID is uuid5(year_normalized|era) — deterministic for idempotent upsert
- linked_entity_ids is union across all events, sorted + deduped
- embedding_text is per-event prefixed with year_display + optional (date), joined with "\n"
- text = embedding_text (for now, ready to diverge for future models)
- All upstream fields pass through: year, year_normalized, era, source, source_url
"""

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("prepare_embedding_input")

# UUID namespace for deterministic ID generation
EMBEDDING_NAMESPACE = uuid.NAMESPACE_URL


def slugify_era(era: str) -> str:
    """
    Convert era string to a slug for human-readable chunk_key.
    E.g., "8th and 7th centuries BC" -> "8th_and_7th_centuries_BC"
    """
    return era.replace(" ", "_").replace(",", "")


def format_event_prefix(year_display: str, date: Optional[str]) -> str:
    """
    Format the event prefix per the contract rule.
    
    Format: f"{year_display}" + (f" ({date})" if date else "") + ": "
    
    Examples:
    - format_event_prefix("753 BC", "21 April") -> "753 BC (21 April): "
    - format_event_prefix("753 BC", "") -> "753 BC: "
    - format_event_prefix("753 BC", None) -> "753 BC: "
    """
    if date and date.strip():
        return f"{year_display} ({date}): "
    return f"{year_display}: "


def flatten_linked_entities(events: list[dict]) -> list[str]:
    """
    Union all QIDs across all events, dedupe, and sort.
    
    Args:
        events: List of event dicts, each with linked_entity_ids list
    
    Returns:
        Sorted list of unique QID strings
    """
    entity_set = set()
    for event in events:
        event_entities = event.get("linked_entity_ids", [])
        if isinstance(event_entities, list):
            entity_set.update(event_entities)
    return sorted(entity_set)


def prepare_embedding_input(
    raw_chunks: list[dict],
) -> list[dict]:
    """
    Transform nested chunks.json schema to flat embedding-ready records.
    
    Input schema (chunk):
    {
      "year_normalized": int,
      "year": str,
      "era": str,
      "events": [
        {
          "year_display": str,
          "date": str (empty if not present),
          "event": str,
          "linked_entity_ids": list[str],
          ...
        },
        ...
      ],
      "linked_entity_ids": dict[QID: QID],
      "source": str,
      "source_url": str,
    }
    
    Output schema (flat record):
    {
      "id": str (UUID),
      "chunk_key": str,
      "year": str,
      "year_normalized": int,
      "era": str,
      "text": str,
      "embedding_text": str,
      "linked_entity_ids": list[str] (sorted, deduped),
      "source": str,
      "source_url": str,
    }
    
    Contract rules:
    - Every chunk produces exactly one record (conservation)
    - ID is uuid5(year_normalized|era) — deterministic and idempotent
    - linked_entity_ids is union across all events, sorted + deduped
    - embedding_text is per-event prefixed with year_display + optional (date), joined with "\n"
    - text = embedding_text (for now)
    - Pass-through fields: year, year_normalized, era, source, source_url
    
    Args:
        raw_chunks: List of chunk dicts from processed/chunks.json
    
    Returns:
        List of flat, embedding-ready records
    
    Raises:
        ValueError: If a chunk or event is missing required fields
    """
    flat_chunks = []
    
    for chunk_idx, chunk in enumerate(raw_chunks):
        # Extract required fields
        year_normalized = chunk.get("year_normalized")
        year = chunk.get("year")
        era = chunk.get("era")
        events = chunk.get("events", [])
        source = chunk.get("source")
        source_url = chunk.get("source_url")
        
        # Validate required fields
        if year_normalized is None:
            raise ValueError(
                f"Chunk at index {chunk_idx} missing required field: year_normalized"
            )
        if era is None:
            raise ValueError(
                f"Chunk at index {chunk_idx} missing required field: era"
            )
        
        # Check if events field exists and is a list
        if "events" not in chunk:
            raise ValueError(
                f"Chunk at index {chunk_idx} missing required field: events"
            )
        if not isinstance(events, list):
            raise ValueError(
                f"Chunk at index {chunk_idx} has invalid events field (must be a list)"
            )
        
        # Handle empty events array (defensive coverage per test scenario)
        # Locked-in rule: produce record with empty embedding_text if no events
        if not events:
            logger.warning(
                "Chunk at index %d (year_normalized=%s, era=%r) has no events: embedding_text will be empty",
                chunk_idx, year_normalized, era
            )
            # Proceed to create record with empty embedding_text
        
        # --- Generate deterministic UUID from year_normalized + era ---
        id_string = f"{year_normalized}|{era}"
        chunk_id = str(uuid.uuid5(EMBEDDING_NAMESPACE, id_string))
        
        # --- Generate human-readable chunk_key ---
        era_slug = slugify_era(era)
        chunk_key = f"y_{year_normalized}__{era_slug}"
        
        # --- Build embedding_text: each event prefixed, joined with "\n" ---
        embedding_parts = []
        
        if events:  # Only process if events is not empty
            for event in events:
                year_display = event.get("year_display", "")
                date = event.get("date", "")
                event_text = event.get("event", "")
                
                # Validate event fields
                if not year_display:
                    raise ValueError(
                        f"Event in chunk {chunk_key} (index {chunk_idx}) missing year_display"
                    )
                if not event_text:
                    raise ValueError(
                        f"Event in chunk {chunk_key} (index {chunk_idx}) missing event text"
                    )
                
                # Format: "{year_display}" + " ({date})" if date else "" + ": {event}"
                prefix = format_event_prefix(year_display, date)
                embedding_parts.append(f"{prefix}{event_text}")
        
        embedding_text = "\n".join(embedding_parts)  # Empty string if no events
        
        # --- Union and sort linked_entity_ids across all events ---
        linked_ids = flatten_linked_entities(events)
        
        # --- Build flat record ---
        flat_record = {
            "id": chunk_id,
            "chunk_key": chunk_key,
            "year": year,
            "year_normalized": year_normalized,
            "era": era,
            "text": embedding_text,  # Same as embedding_text for now
            "embedding_text": embedding_text,
            "linked_entity_ids": linked_ids,
            "source": source,
            "source_url": source_url,
        }
        
        flat_chunks.append(flat_record)
    
    return flat_chunks


def main():
    """CLI entry point for prepare_embedding_input."""
    parser = argparse.ArgumentParser(
        description="Transform chunks.json to embedding-ready format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/chunks.json",
        help="Path to chunks.json (default: data/processed/chunks.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/embedding_ready.json",
        help="Path to output embedding_ready.json (default: data/processed/embedding_ready.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    logger.info("Loading chunks from: %s", args.input)
    try:
        with open(args.input) as f:
            raw_chunks = json.load(f)
    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", args.input, e)
        raise

    logger.info("Loaded %d chunks", len(raw_chunks))

    flat_chunks = prepare_embedding_input(raw_chunks)

    logger.info("Transformed %d chunks to embedding-ready format", len(flat_chunks))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(flat_chunks, f, indent=2)

    logger.info("Output written to: %s", args.output)
    print(f"✓ Transformed {len(flat_chunks)} chunks")
    print(f"✓ Output written to {args.output}")


if __name__ == "__main__":
    main()