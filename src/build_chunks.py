"""
Builds embedding-ready chunks from roman_timeline_linked.json by merging
all events that share the same year into a single chunk, and attaching
retrieval-useful metadata (era, dates, linked Wikidata entities).

Usage:
    python build_chunks.py

Requires roman_timeline_linked.json (from resolve_wikidata.py) in the
same directory.

Output:
    chunks.json
"""

import json

SOURCE_NAME = "Wikipedia: Timeline of Roman history"
SOURCE_URL = "https://en.wikipedia.org/wiki/Timeline_of_Roman_history"


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


def main():
    with open("roman_timeline_linked.json", encoding="utf-8") as f:
        events = json.load(f)

    chunks = build_chunks(events)

    total_events = sum(c["event_count"] for c in chunks)
    merged = sum(1 for c in chunks if c["event_count"] > 1)
    print(f"Built {len(chunks)} chunks from {total_events} events "
          f"({merged} chunks merged multiple events).")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print("Saved chunks.json")


if __name__ == "__main__":
    main()
