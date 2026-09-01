"""
Phase 2a: prepare_embedding_input.py test suite (contract-first).

Tests the transformation from nested chunks.json schema to flat embedding-ready schema.
All tests start in RED state against a pass-only stub.

Fixtures come from test_prepare_embedding_input_fixtures.py.
"""

import uuid
from pathlib import Path

import pytest


# ============================================================================
# CRITICAL: Conservation and uniqueness
# ============================================================================


class TestConservation:
    """
    CRITICAL: Every chunk produces exactly one flat record.
    No chunk dropped or split.
    """

    def test_every_chunk_produces_one_record(self, chunks_batch_for_uniqueness_check):
        """
        Transform a batch of 3 chunks; verify len(output) == 3.
        Contract rule: len(output) == len(input)
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input(chunks_batch_for_uniqueness_check)
        
        assert len(result) == len(chunks_batch_for_uniqueness_check), \
            f"Expected {len(chunks_batch_for_uniqueness_check)} output records, got {len(result)}"

    def test_conservation_on_real_dataset(self):
        """
        Load the real 412-chunk chunks.json and verify len(output) == 412.
        This is a real integration test (not mocked).
        """
        import json
        from pathlib import Path
        from prepare_embedding_input import prepare_embedding_input
        
        # Load real chunks.json from project root
        chunks_path = Path("data/processed/chunks.json")
        if not chunks_path.exists():
            pytest.skip("chunks.json not found at data/processed/chunks.json")
        
        with open(chunks_path) as f:
            chunks = json.load(f)
        
        result = prepare_embedding_input(chunks)
        
        assert len(result) == len(chunks), \
            f"Expected {len(chunks)} output records, got {len(result)}"
        assert len(result) == 412, \
            f"Expected 412 chunks in output (current dataset size), got {len(result)}"


class TestIdUniqueness:
    """
    CRITICAL: (year_normalized, era) pairs are unique across all chunks,
    so the ID generation scheme (uuid5 based on both fields) produces zero collisions.
    """

    def test_id_collisions_on_real_dataset(self):
        """
        Load the real 412-chunk chunks.json, generate IDs for all chunks,
        verify len(set(ids)) == 412 (zero collisions).
        This is a defensive regression test — verified once against real data,
        kept as a standing check in case build_chunks.py changes later.
        """
        import json
        from pathlib import Path
        from prepare_embedding_input import prepare_embedding_input
        
        chunks_path = Path("data/processed/chunks.json")
        if not chunks_path.exists():
            pytest.skip("chunks.json not found at data/processed/chunks.json")
        
        with open(chunks_path) as f:
            chunks = json.load(f)
        
        result = prepare_embedding_input(chunks)
        
        ids = [record["id"] for record in result]
        unique_ids = set(ids)
        
        assert len(ids) == len(unique_ids), \
            f"Found {len(ids) - len(unique_ids)} duplicate IDs in {len(ids)} records"

    def test_small_batch_ids_are_unique(self, chunks_batch_for_uniqueness_check):
        """
        Transform a small batch and verify all generated IDs are distinct.
        Faster version of the real dataset test for quick validation.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input(chunks_batch_for_uniqueness_check)
        
        ids = [record["id"] for record in result]
        unique_ids = set(ids)
        
        assert len(ids) == len(unique_ids), \
            f"Expected {len(ids)} unique IDs, got {len(unique_ids)}"


class TestIdDeterminism:
    """
    CRITICAL: Same input → same IDs across runs.
    Required for idempotent upsert to work downstream.
    """

    def test_id_determinism_across_runs(self, chunks_batch_for_uniqueness_check):
        """
        Transform the same batch twice; verify all IDs match between runs.
        Contract rule: uuid5(year_normalized|era) is deterministic
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result1 = prepare_embedding_input(chunks_batch_for_uniqueness_check)
        result2 = prepare_embedding_input(chunks_batch_for_uniqueness_check)
        
        ids1 = [record["id"] for record in result1]
        ids2 = [record["id"] for record in result2]
        
        assert ids1 == ids2, \
            f"IDs differ between runs:\nRun 1: {ids1}\nRun 2: {ids2}"

    def test_id_is_valid_uuid(self, chunks_batch_for_uniqueness_check):
        """
        Verify every generated ID can be parsed as a valid UUID string.
        Contract rule: id is str(uuid.uuid5(...))
        """
        import uuid as uuid_module
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input(chunks_batch_for_uniqueness_check)
        
        for record in result:
            record_id = record["id"]
            try:
                # Try to parse as UUID string
                uuid_module.UUID(record_id)
            except ValueError:
                pytest.fail(f"ID '{record_id}' is not a valid UUID string")


# ============================================================================
# CRITICAL: Entity union and deduplication
# ============================================================================


class TestLinkedEntityUnion:
    """
    CRITICAL: linked_entity_ids are correctly unioned across all events in a chunk,
    deduped, and sorted.
    """

    def test_entities_unioned_across_events(
        self, multi_event_chunk_overlapping_entities
    ):
        """
        Transform a chunk where one QID appears in both events.
        Verify the output has that QID exactly once, not duplicated.
        
        Input events:
        - Event 1: ["Q103705", "Q106405", "Q220"]
        - Event 2: ["Q12544", "Q220", "Q105762"]
        
        Expected union (sorted): ["Q103705", "Q105762", "Q106405", "Q12544", "Q220"]
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([multi_event_chunk_overlapping_entities])
        
        assert len(result) == 1, "Expected 1 output record"
        output_entities = result[0]["linked_entity_ids"]
        
        # Expected set (union of both events' entities)
        expected_entities = ["Q103705", "Q105762", "Q106405", "Q12544", "Q220"]
        
        assert output_entities == expected_entities, \
            f"Expected {expected_entities}, got {output_entities}"

    def test_entities_sorted_and_deduped(
        self, multi_event_chunk_overlapping_entities
    ):
        """
        Verify output linked_entity_ids is both sorted and deduped.
        Q220 appears in both events but should appear only once in output.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([multi_event_chunk_overlapping_entities])
        output_entities = result[0]["linked_entity_ids"]
        
        # Check it's sorted
        assert output_entities == sorted(output_entities), \
            f"Entities not sorted: {output_entities}"
        
        # Check it's deduped (no duplicates)
        assert len(output_entities) == len(set(output_entities)), \
            f"Found duplicates in {output_entities}"
        
        # Specifically check that Q220 (which appears in both events) appears exactly once
        assert output_entities.count("Q220") == 1, \
            f"Q220 appears {output_entities.count('Q220')} times, expected 1"

    def test_single_event_entities_preserved(self, single_event_chunk_with_date):
        """
        Single-event chunk should have the same entity list (sorted) in output.
        Guards against unnecessary re-ordering or dropping.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        output_entities = result[0]["linked_entity_ids"]
        
        # Input has one event with ["Q1405", "Q1747689", "Q220"]
        # Expected (sorted): ["Q1405", "Q1747689", "Q220"]
        expected = ["Q1405", "Q1747689", "Q220"]
        
        assert output_entities == expected, \
            f"Expected {expected}, got {output_entities}"


# ============================================================================
# CRITICAL: Embedding text construction with conditional date formatting
# ============================================================================


class TestEmbeddingTextConstruction:
    """
    CRITICAL: embedding_text is correctly prefixed per event and joined for multi-event chunks.
    Date inclusion is conditional: f"{year_display}" + (f" ({date})" if date else "") + f": {event}"
    """

    def test_embedding_text_without_date(self, real_multi_event_chunk):
        """
        Two events, both without date fields.
        Expected: "752 BC: <event1>" + "\n" + "752 BC: <event2>"
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([real_multi_event_chunk])
        embedding_text = result[0]["embedding_text"]
        
        # Should have newline separating two events
        assert "\n" in embedding_text, "Multi-event embedding_text should contain newline separator"
        
        lines = embedding_text.split("\n")
        assert len(lines) == 2, f"Expected 2 events separated by newline, got {len(lines)}"
        
        # Each line should start with year_display
        assert lines[0].startswith("752 BC:"), f"First event doesn't start with '752 BC:', got: {lines[0][:20]}"
        assert lines[1].startswith("752 BC:"), f"Second event doesn't start with '752 BC:', got: {lines[1][:20]}"
        
        # Neither should have date (empty date fields)
        assert "(" not in lines[0] and ")" not in lines[0], \
            f"First event shouldn't have parentheses (no date): {lines[0]}"
        assert "(" not in lines[1] and ")" not in lines[1], \
            f"Second event shouldn't have parentheses (no date): {lines[1]}"

    def test_embedding_text_with_date(self, single_event_chunk_with_date):
        """
        Single event with date field "21 April".
        Expected prefix: "753 BC (21 April): <event text>"
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        embedding_text = result[0]["embedding_text"]
        
        assert embedding_text.startswith("753 BC (21 April):"), \
            f"Expected to start with '753 BC (21 April):', got: {embedding_text[:40]}"

    def test_embedding_text_mixed_dates(self, multi_event_chunk_mixed_dates):
        """
        Two events: one with date "August 24", one without.
        Expected:
          Line 1: "79 AD (August 24): Eruption of Mount Vesuvius..."
          Line 2: "79 AD: Emperor Titus ascends..."
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([multi_event_chunk_mixed_dates])
        embedding_text = result[0]["embedding_text"]
        
        lines = embedding_text.split("\n")
        assert len(lines) == 2, f"Expected 2 events, got {len(lines)}"
        
        # First event should have date
        assert lines[0].startswith("79 AD (August 24):"), \
            f"First event should have date in parens, got: {lines[0][:40]}"
        
        # Second event should NOT have date (empty string)
        assert lines[1].startswith("79 AD:") and "(" not in lines[1], \
            f"Second event should NOT have date in parens, got: {lines[1][:40]}"

    def test_embedding_text_distinguishes_events(self, real_multi_event_chunk):
        """
        Multi-event chunk with distinct events should maintain event boundaries.
        Verify that each event's text is distinguishable in the output.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([real_multi_event_chunk])
        embedding_text = result[0]["embedding_text"]
        
        # Two different events, so text should be distinguishable
        lines = embedding_text.split("\n")
        assert len(lines) == 2, f"Expected 2 events separated by newline"
        
        # Get the raw event texts from fixture
        event1_text = real_multi_event_chunk["events"][0]["event"]
        event2_text = real_multi_event_chunk["events"][1]["event"]
        
        # Verify both event texts are present in the output
        assert event1_text in embedding_text, \
            f"First event text not found in embedding_text"
        assert event2_text in embedding_text, \
            f"Second event text not found in embedding_text"


# ============================================================================
# CRITICAL: Payload pass-through fields
# ============================================================================


class TestPassThroughFields:
    """
    CRITICAL: year, year_normalized, era, source, source_url pass through unmodified.
    """

    def test_year_normalized_unchanged(self, single_event_chunk_with_date):
        """
        year_normalized on input == year_normalized on output.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        output_year_norm = result[0]["year_normalized"]
        input_year_norm = single_event_chunk_with_date["year_normalized"]
        
        assert output_year_norm == input_year_norm, \
            f"year_normalized changed: input={input_year_norm}, output={output_year_norm}"

    def test_year_string_unchanged(self, single_event_chunk_with_date):
        """
        year (string, e.g. "753 BC") on input == year on output.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        output_year = result[0]["year"]
        input_year = single_event_chunk_with_date["year"]
        
        assert output_year == input_year, \
            f"year changed: input={input_year}, output={output_year}"

    def test_era_unchanged(self, real_multi_event_chunk):
        """
        era on input == era on output.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([real_multi_event_chunk])
        output_era = result[0]["era"]
        input_era = real_multi_event_chunk["era"]
        
        assert output_era == input_era, \
            f"era changed: input={input_era}, output={output_era}"

    def test_source_and_url_unchanged(self, single_event_chunk_with_date):
        """
        source and source_url on input == output.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        output_source = result[0]["source"]
        output_url = result[0]["source_url"]
        input_source = single_event_chunk_with_date["source"]
        input_url = single_event_chunk_with_date["source_url"]
        
        assert output_source == input_source, \
            f"source changed: input={input_source}, output={output_source}"
        assert output_url == input_url, \
            f"source_url changed: input={input_url}, output={output_url}"


# ============================================================================
# HIGH: Text field (same as embedding_text for now)
# ============================================================================


class TestTextField:
    """
    HIGH: text field is identical to embedding_text for now.
    Kept as a separate field for future model swaps (e.g., e5 prefix).
    """

    def test_text_equals_embedding_text(self, real_multi_event_chunk):
        """
        text == embedding_text in output.
        Guards against silent divergence if the two fields are managed separately.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([real_multi_event_chunk])
        output_text = result[0]["text"]
        output_embedding_text = result[0]["embedding_text"]
        
        assert output_text == output_embedding_text, \
            f"text and embedding_text diverged:\ntext={output_text}\nembedding_text={output_embedding_text}"


# ============================================================================
# HIGH: Chunk key (human-readable debugging aid)
# ============================================================================


class TestChunkKey:
    """
    HIGH: chunk_key is a human-readable string for debugging,
    based on year_normalized + era (slugified).
    """

    def test_chunk_key_is_present(self, single_event_chunk_with_date):
        """
        Output has a chunk_key field.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        
        assert "chunk_key" in result[0], \
            f"Output missing chunk_key field. Keys present: {list(result[0].keys())}"

    def test_chunk_key_format_legible(self, single_event_chunk_with_date):
        """
        chunk_key format is roughly "y_{year_normalized}__{era_slugified}".
        Should be human-legible for debugging.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        chunk_key = result[0]["chunk_key"]
        
        # Should start with "y_"
        assert chunk_key.startswith("y_"), \
            f"chunk_key should start with 'y_', got: {chunk_key}"
        
        # Should contain year_normalized somewhere
        year_norm = single_event_chunk_with_date["year_normalized"]
        assert str(year_norm) in chunk_key, \
            f"chunk_key should contain year_normalized ({year_norm}), got: {chunk_key}"
        
        # Should be slug-like: alphanumeric, underscores, and hyphens
        # (hyphen required since year_normalized can be negative for BC years,
        # e.g. "y_-753__8th_and_7th_centuries_BC")
        import re
        assert re.match(r"^y_[\w-]+$", chunk_key), \
            f"chunk_key should be slug-like (alphanumeric + underscores + hyphens), got: {chunk_key}"


# ============================================================================
# MEDIUM: Edge cases
# ============================================================================


class TestEmptyEventsArray:
    """
    MEDIUM: Chunk with empty events[] array.
    Not a live case (0 of 412 chunks), but defensive coverage.
    Expected behavior: skip with warning or return with empty embedding_text.
    """

    def test_empty_events_array_handled(self, chunk_with_empty_events):
        """
        Run transform on chunk with events=[].
        Should not crash. Output should have empty or clearly-marked embedding_text.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([chunk_with_empty_events])
        
        # Should produce one record (no skip/drop)
        assert len(result) == 1, "Empty events should still produce one output record"
        
        # embedding_text should be empty or minimal
        embedding_text = result[0].get("embedding_text", "")
        assert isinstance(embedding_text, str), "embedding_text should be a string"


class TestSingleEventConsistency:
    """
    MEDIUM: Single-event chunk formatting is consistent with multi-event logic,
    not a shortcut that might diverge.
    """

    def test_single_event_follows_multi_rule(self, single_event_chunk_with_date):
        """
        Single-event chunk produces output via the same logic as a multi-event chunk.
        Embedding text has the same prefix rule (year_display + optional date + ": " + event).
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input([single_event_chunk_with_date])
        embedding_text = result[0]["embedding_text"]
        
        # Should follow the same prefix rule as multi-event chunks
        # i.e., start with "753 BC (21 April):" since event has a date
        assert embedding_text.startswith("753 BC (21 April):"), \
            f"Single-event should follow multi-event prefix rule, got: {embedding_text[:40]}"


# ============================================================================
# MEDIUM: Malformed input handling
# ============================================================================


class TestMalformedInput:
    """
    MEDIUM: Malformed chunks (missing required fields) fail clearly.
    """

    def test_missing_year_normalized(self):
        """
        Chunk missing year_normalized field should fail cleanly.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        bad_chunk = {
            "year": "753 BC",
            # missing year_normalized
            "era": "8th and 7th centuries BC",
            "events": [{"year_display": "753 BC", "date": "", "event": "test"}],
            "linked_entity_ids": {},
            "source": "test",
            "source_url": "test",
        }
        
        with pytest.raises((KeyError, ValueError, TypeError)):
            prepare_embedding_input([bad_chunk])

    def test_missing_era(self):
        """
        Chunk missing era field should fail cleanly.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        bad_chunk = {
            "year": "753 BC",
            "year_normalized": -753,
            # missing era
            "events": [{"year_display": "753 BC", "date": "", "event": "test"}],
            "linked_entity_ids": {},
            "source": "test",
            "source_url": "test",
        }
        
        with pytest.raises((KeyError, ValueError, TypeError)):
            prepare_embedding_input([bad_chunk])

    def test_missing_events_array(self):
        """
        Chunk missing events[] array should fail cleanly.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        bad_chunk = {
            "year": "753 BC",
            "year_normalized": -753,
            "era": "8th and 7th centuries BC",
            # missing events
            "linked_entity_ids": {},
            "source": "test",
            "source_url": "test",
        }
        
        with pytest.raises((KeyError, ValueError, TypeError)):
            prepare_embedding_input([bad_chunk])

    def test_event_missing_year_display(self):
        """
        Event missing year_display should fail cleanly.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        bad_chunk = {
            "year": "753 BC",
            "year_normalized": -753,
            "era": "8th and 7th centuries BC",
            "events": [
                {
                    # missing year_display
                    "date": "",
                    "event": "test",
                    "linked_entity_ids": [],
                }
            ],
            "linked_entity_ids": {},
            "source": "test",
            "source_url": "test",
        }
        
        with pytest.raises((KeyError, ValueError, TypeError)):
            prepare_embedding_input([bad_chunk])

    def test_event_missing_event_text(self):
        """
        Event missing event (text) field should fail cleanly.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        bad_chunk = {
            "year": "753 BC",
            "year_normalized": -753,
            "era": "8th and 7th centuries BC",
            "events": [
                {
                    "year_display": "753 BC",
                    "date": "",
                    # missing event (the text)
                    "linked_entity_ids": [],
                }
            ],
            "linked_entity_ids": {},
            "source": "test",
            "source_url": "test",
        }
        
        with pytest.raises((KeyError, ValueError, TypeError)):
            prepare_embedding_input([bad_chunk])


# ============================================================================
# Stub function signature check (verify contract is testable)
# ============================================================================


class TestFunctionSignature:
    """
    Verify the transform function exists and has the expected signature.
    """

    def test_prepare_embedding_input_function_exists(self):
        """
        Import and call prepare_embedding_input function.
        """
        from prepare_embedding_input import prepare_embedding_input
        assert callable(prepare_embedding_input)

    def test_function_accepts_list_of_dicts(self, chunks_batch_for_uniqueness_check):
        """
        Call prepare_embedding_input(chunks) with a list of dicts.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        # Should not raise TypeError
        result = prepare_embedding_input(chunks_batch_for_uniqueness_check)
        assert result is not None

    def test_function_returns_list_of_dicts(self, chunks_batch_for_uniqueness_check):
        """
        Result of prepare_embedding_input is a list of dicts.
        """
        from prepare_embedding_input import prepare_embedding_input
        
        result = prepare_embedding_input(chunks_batch_for_uniqueness_check)
        
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert all(isinstance(item, dict) for item in result), \
            "All items in result should be dicts"