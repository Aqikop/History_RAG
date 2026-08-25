"""
Contract-first tests for build_chunks.py.

This suite was written against the function outline only, without reading
any implementation bodies. All function bodies remain `pass`, so this suite
is intentionally in red state until the real implementation exists.

CONFIRMED FROM DOCSTRINGS:
  C1. build_chunks(events) — groups events by year_normalized (the stable,
      sortable, unambiguous key). Returns a list of chunks, where each
      chunk represents all events sharing the same year_normalized value.
      Docstring explicitly warns against using "year" (display string) as
      the grouping key due to BC/AD boundary format changes.

  C2. run(...) -> Optional[str]
      Reads resolved events from input_path, builds chunks, writes to
      output. Returns output path (as a string) on success, or None if
      dry_run=True. Matches the pattern of scrape and resolve scripts.

  C3. build_arg_parser() — supports flags: --input, --output, --dry-run,
      --log-level (inferred from pattern).

  C4. main() — reads sys.argv, calls run() with parsed args.

REMAINING ASSUMPTIONS (flagged because docstring didn't specify):
  A1. Chunk structure — assumed to have: `year_normalized` (int key),
      `year` (string), `era` (from first event in group), `date` (unclear
      if a list, comma-separated string, or a single value — guessing
      it's a list of dates from all merged events), `events` (list of
      individual event dicts), `linked_entity_ids` (merged union from all
      events), and metadata (SOURCE_NAME, SOURCE_URL).

  A2. Chunk ordering — assumed to be sorted by year_normalized (ascending),
      so timeline reads chronologically despite BC/AD discontinuity being
      resolved astronomically.

  A3. When multiple events share the same year, linked_entity_ids are
      unioned (no duplicates from multiple events with the same entity).

  A4. Single-event years produce a 1:1 chunk (not skipped or merged
      specially) — just one event in the `events` list.

  A5. Chunk count in fixture — 10 events, all distinct years, so 10 chunks
      (no merges to test). Tests using this fixture can only verify the
      1:1 no-merge case. A separate fixture or real data is needed to test
      same-year merges.

  A6. run() output structure — assumed to be a JSON array of chunks
      (following the pattern of prior stages), not a dict or other format.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import build_chunks as chunker


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_EVENTS_PATH = FIXTURE_DIR / "sample_events_resolved.json"


@pytest.fixture
def sample_events() -> list[dict]:
    """Load the fixture resolved events."""
    return json.loads(SAMPLE_EVENTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def year_normalized_counts(sample_events) -> dict:
    """Count events per year_normalized to verify no same-year merges in fixture."""
    counts = {}
    for event in sample_events:
        year = event["year_normalized"]
        counts[year] = counts.get(year, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# build_chunks()
# ---------------------------------------------------------------------------
class TestBuildChunks:
    def test_build_chunks_returns_list(self, sample_events):
        """build_chunks returns a list (of chunks)."""
        result = chunker.build_chunks(sample_events)
        assert isinstance(result, list)

    def test_build_chunks_groups_by_year_normalized(self, sample_events):
        """Events are grouped by year_normalized, not by year (string)."""
        result = chunker.build_chunks(sample_events)
        
        # Each chunk should have a year_normalized that appears in at least
        # one event from the input
        years_in_result = {chunk.get("year_normalized") for chunk in result}
        years_in_input = {e["year_normalized"] for e in sample_events}
        assert years_in_result == years_in_input

    def test_build_chunks_distinct_years_produce_distinct_chunks(
        self, sample_events, year_normalized_counts
    ):
        """Each distinct year_normalized produces exactly one chunk."""
        result = chunker.build_chunks(sample_events)
        
        # Fixture has all distinct years (counts are all 1)
        # so chunk count should equal event count
        assert len(result) == len(sample_events)

    def test_build_chunks_same_year_merges_into_one_chunk(self):
        """Two events with the same year_normalized merge into a single chunk.
        
        (Fixture doesn't exercise this — all events have distinct years —
        so this test uses a minimal synthetic input.)"""
        events = [
            {
                "year": "753 BC",
                "year_normalized": -752,
                "era": "8th century BC",
                "date": "",
                "event": "First event of the year.",
                "linked_entity_ids": {"A": "Q1"},
            },
            {
                "year": "753 BC",
                "year_normalized": -752,
                "era": "8th century BC",
                "date": "",
                "event": "Second event of the same year.",
                "linked_entity_ids": {"B": "Q2"},
            },
        ]
        
        result = chunker.build_chunks(events)
        
        # Must produce exactly 1 chunk, not 2
        assert len(result) == 1
        chunk = result[0]
        assert chunk["year_normalized"] == -752
        # Both events must be in the chunk
        assert len(chunk.get("events", [])) == 2

    def test_build_chunks_preserves_all_events(self, sample_events):
        """No event is lost or duplicated across all chunks."""
        result = chunker.build_chunks(sample_events)
        
        # Sum event counts across all chunks
        total_events_in_chunks = sum(
            len(chunk.get("events", [])) for chunk in result
        )
        assert total_events_in_chunks == len(sample_events)

    def test_build_chunks_single_event_year_produces_one_chunk(self):
        """A year with only one event produces a 1:1 chunk."""
        events = [
            {
                "year": "AD 14",
                "year_normalized": 14,
                "era": "1st century AD",
                "date": "19 August",
                "event": "Only event in this year.",
                "linked_entity_ids": {"Augustus": "Q1397"},
            }
        ]
        
        result = chunker.build_chunks(events)
        
        assert len(result) == 1
        chunk = result[0]
        assert chunk["year_normalized"] == 14
        assert len(chunk.get("events", [])) == 1

    def test_build_chunks_sorted_by_year_normalized(self, sample_events):
        """Chunks are sorted by year_normalized (ascending)."""
        result = chunker.build_chunks(sample_events)
        
        years = [chunk.get("year_normalized") for chunk in result]
        assert years == sorted(years)

    def test_build_chunks_merges_linked_entity_ids(self):
        """When events merge, linked_entity_ids are unioned (no duplicates)."""
        events = [
            {
                "year": "753 BC",
                "year_normalized": -752,
                "era": "8th century BC",
                "date": "",
                "event": "First event.",
                "linked_entity_ids": {"A": "Q1", "B": "Q2"},
            },
            {
                "year": "753 BC",
                "year_normalized": -752,
                "era": "8th century BC",
                "date": "",
                "event": "Second event.",
                "linked_entity_ids": {"B": "Q2", "C": "Q3"},
            },
        ]
        
        result = chunker.build_chunks(events)
        
        chunk = result[0]
        merged_ids = chunk.get("linked_entity_ids", {})
        # Union should have A, B, C (no duplicate B)
        assert merged_ids.get("A") == "Q1"
        assert merged_ids.get("B") == "Q2"
        assert merged_ids.get("C") == "Q3"
        assert len(merged_ids) == 3

    def test_build_chunks_includes_metadata(self, sample_events):
        """Each chunk includes source metadata (SOURCE_NAME, SOURCE_URL)."""
        result = chunker.build_chunks(sample_events)
        
        assert len(result) > 0
        chunk = result[0]
        assert "source_name" in chunk or chunk.get("source") == chunker.SOURCE_NAME
        assert "source_url" in chunk or chunker.SOURCE_URL in str(chunk)

    def test_build_chunks_year_field_preserved(self, sample_events):
        """Chunk preserves the year display string (e.g. '753 BC')."""
        result = chunker.build_chunks(sample_events)
        
        assert len(result) > 0
        chunk = result[0]
        assert "year" in chunk
        assert isinstance(chunk["year"], str)

    def test_build_chunks_era_field_preserved(self, sample_events):
        """Chunk includes era from the merged events."""
        result = chunker.build_chunks(sample_events)
        
        assert len(result) > 0
        chunk = result[0]
        assert "era" in chunk


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
class TestRun:
    @patch("build_chunks.build_chunks")
    def test_run_reads_input_and_writes_chunks(self, mock_build, tmp_path):
        """run() reads events, builds chunks, writes to output."""
        events = [
            {
                "year": "AD 14",
                "year_normalized": 14,
                "era": "1st century AD",
                "date": "19 August",
                "event": "Test event.",
                "linked_entity_ids": {"Augustus": "Q1397"},
            }
        ]
        mock_build.return_value = [
            {
                "year_normalized": 14,
                "year": "AD 14",
                "era": "1st century AD",
                "events": events,
                "linked_entity_ids": {"Augustus": "Q1397"},
                "source_name": chunker.SOURCE_NAME,
                "source_url": chunker.SOURCE_URL,
            }
        ]

        input_file = tmp_path / "events.json"
        output_file = tmp_path / "chunks.json"
        input_file.write_text(json.dumps(events))

        result = chunker.run(
            input_path=str(input_file),
            output=str(output_file),
            dry_run=False,
        )

        assert result == str(output_file)
        assert output_file.exists()
        assert mock_build.called

    @patch("build_chunks.build_chunks")
    def test_run_dry_run_writes_no_files(self, mock_build, tmp_path):
        """dry_run=True prevents file writes but still builds chunks."""
        events = [
            {
                "year": "AD 14",
                "year_normalized": 14,
                "era": "1st century AD",
                "date": "19 August",
                "event": "Test event.",
                "linked_entity_ids": {"Augustus": "Q1397"},
            }
        ]
        mock_build.return_value = []

        input_file = tmp_path / "events.json"
        output_file = tmp_path / "chunks.json"
        input_file.write_text(json.dumps(events))

        result = chunker.run(
            input_path=str(input_file),
            output=str(output_file),
            dry_run=True,
        )

        assert result is None
        assert not output_file.exists()
        assert mock_build.called

    @patch("build_chunks.build_chunks")
    def test_run_returns_output_path_on_success(self, mock_build, tmp_path):
        """run() returns the output path as a string on success."""
        mock_build.return_value = []
        input_file = tmp_path / "events.json"
        output_file = tmp_path / "chunks.json"
        input_file.write_text(json.dumps([]))

        result = chunker.run(
            input_path=str(input_file),
            output=str(output_file),
            dry_run=False,
        )

        assert result == str(output_file)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_arg_parser()
# ---------------------------------------------------------------------------
class TestBuildArgParser:
    def test_parser_has_input_flag(self):
        parser = chunker.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "input")
        assert args.input == chunker.DEFAULT_INPUT

    def test_parser_has_output_flag(self):
        parser = chunker.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "output")
        assert args.output == chunker.DEFAULT_OUTPUT

    def test_parser_has_dry_run_flag(self):
        parser = chunker.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "dry_run")
        assert args.dry_run is False

    def test_parser_dry_run_flag_sets_true(self):
        parser = chunker.build_arg_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parser_has_log_level_flag(self):
        parser = chunker.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "log_level")

    def test_parser_custom_input_output_paths(self):
        parser = chunker.build_arg_parser()
        args = parser.parse_args([
            "--input", "/tmp/custom_input.json",
            "--output", "/tmp/custom_output.json",
        ])
        assert args.input == "/tmp/custom_input.json"
        assert args.output == "/tmp/custom_output.json"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
class TestMain:
    @patch("build_chunks.run")
    def test_main_invokes_run_with_parsed_args(self, mock_run, monkeypatch):
        monkeypatch.setattr("sys.argv", ["build_chunks.py"])
        chunker.main()
        assert mock_run.called

    @patch("build_chunks.run")
    def test_main_propagates_dry_run_flag(self, mock_run, monkeypatch):
        monkeypatch.setattr("sys.argv", ["build_chunks.py", "--dry-run"])
        chunker.main()
        assert mock_run.called
        _, kwargs = mock_run.call_args
        if kwargs:
            assert kwargs.get("dry_run") is True


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_sample_events_fixture_has_required_structure(self, sample_events):
        """Verify fixture has all fields from resolve_wikidata output."""
        for event in sample_events:
            assert "year" in event
            assert "year_normalized" in event
            assert "era" in event
            assert "date" in event
            assert "event" in event
            assert "wiki_links" in event
            assert "linked_entity_ids" in event

    def test_fixture_all_years_distinct(self, year_normalized_counts):
        """Fixture has no same-year events (all distinct years)."""
        counts = list(year_normalized_counts.values())
        assert all(c == 1 for c in counts), (
            "Fixture should have all distinct years for basic testing; "
            "same-year merges tested separately with synthetic input"
        )

    def test_chunk_count_equals_event_count_when_no_merges(
        self, sample_events
    ):
        """With all distinct years, chunk count = event count."""
        result = chunker.build_chunks(sample_events)
        assert len(result) == len(sample_events)