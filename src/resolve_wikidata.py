"""
Resolves the wiki_links captured in roman_timeline.json to Wikidata QIDs,
using Wikipedia's pageprops API (a direct, unambiguous Wikipedia-article
-> Wikidata-item mapping -- no fuzzy search/disambiguation needed, since
each link already points at one specific Wikipedia article).

Usage:
    pip install requests
    python resolve_wikidata.py

Requires roman_timeline.json (from scrape_roman_timeline.py) in the same
directory.

Output:
    roman_timeline_linked.json   -- events with resolved Wikidata IDs
    unresolved_titles.txt        -- titles with no Wikidata item (for review)
"""

import json
import time
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "RAGDatasetBuilder/1.0 (https://example.com; you@example.com)"
}
BATCH_SIZE = 50  # MediaWiki API limit per request for non-bot accounts

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
            print(f"  Batch{attempt_label} attempt {attempt + 1} failed: {e}")
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
        print(f"Re-checking {len(unresolved)} unresolved titles individually...")
        for title in unresolved:
            _resolve_one_batch([title], result)
            time.sleep(0.2)

    return result


def main():
    with open("roman_timeline.json", encoding="utf-8") as f:
        events = json.load(f)

    unique_titles = sorted({
        link["wiki_title"]
        for event in events
        for link in event["wiki_links"]
    })
    print(f"Resolving {len(unique_titles)} unique linked titles to Wikidata IDs...")

    title_to_qid = resolve_titles_to_qids(unique_titles)

    override_count = 0
    for title, qid in MANUAL_QID_OVERRIDES.items():
        if title in title_to_qid and not title_to_qid[title]:
            title_to_qid[title] = qid
            override_count += 1
    if override_count:
        print(f"Applied {override_count} manual override(s) for known bad titles.")

    resolved_count = sum(1 for q in title_to_qid.values() if q)
    print(f"Resolved {resolved_count} / {len(unique_titles)} titles.")

    for event in events:
        for link in event["wiki_links"]:
            link["wikidata_id"] = title_to_qid.get(link["wiki_title"])

    with open("roman_timeline_linked.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    unresolved = sorted(t for t, q in title_to_qid.items() if not q)
    with open("unresolved_titles.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(unresolved))
    print(f"{len(unresolved)} titles had no Wikidata item (see unresolved_titles.txt)")
    print("Saved roman_timeline_linked.json")


if __name__ == "__main__":
    main()
