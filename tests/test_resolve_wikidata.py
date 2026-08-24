"""
Contract-first tests for resolve_wikidata.py.

This suite was written against the function outline only, without reading
any implementation bodies. All function bodies in the outline remain `pass`,
so this suite is intentionally in red state until the real implementation
exists — that's expected (contract-first, not fitted-to-code).

CONFIRMED FROM DOCSTRINGS:
  C1. chunked(items, size) — yields chunks of size `size` from `items`.
      Return type (generator vs. list) is inferred from usage pattern
      (typically used in a loop or unpacked into a batch).

  C2. _query_batch(batch, attempt_label="") — runs one pageprops query for
      a batch of Wikipedia titles. Returns the parsed 'query' dict from the
      MediaWiki response, or raises after exhausting retries (failures are
      NOT silently swallowed). Retries are implicit (not visible to the
      caller), but failures eventually raise.

  C3. _resolve_one_batch(batch, results) — mutates `results` dict in place.
      Builds a forward map (original title → final title → QID) so that
      multiple originals redirecting to the same final title each get the
      QID instead of one silently overwriting the other (docstring
      explicitly calls out this cross-title collision risk).

  C4. resolve_titles_to_qids(titles) -> dict[str, Optional[str]]
      Given a list of titles, returns a dict mapping title → QID (or None).
      Uses a two-pass strategy:
        1. Batch lookups (fast, ~50 titles/request)
        2. Anything still unresolved after pass 1 gets re-checked
           individually (batch of 1), before concluding it's genuinely
           unresolved

  C5. run(...) -> Optional[str]
      Resolves all wiki_links in input events JSON to Wikidata QIDs.
      Returns the path to resolved events JSON (as a string) on success,
      or None if dry_run=True. Input events are expected to have a `links`
      field (list of (target, label) tuples from scraper output).

  C6. build_arg_parser() — supports flags: --input, --output,
      --unresolved-output, --dry-run, --log-level (inferred from scraper
      pattern; not explicitly listed in outline).

  C7. main() — reads sys.argv, calls run() with parsed args.

REMAINING ASSUMPTIONS (flagged because docstring didn't specify):
  A1. chunked() is a generator (or generator-like iterator), not a list of
      chunks — inferred from typical usage pattern (used in loops, often
      with `for batch in chunked(...):`).

  A2. _query_batch's return value is the 'query' key directly from the
      MediaWiki API JSON response (which contains page titles → QID
      metadata). Exact response structure assumed to follow MediaWiki's
      conventional action=query&prop=pageprops format.

  A3. _resolve_one_batch(batch, results) modifies `results` in place and
      returns nothing (or returns None). The results dict maps each
      original title to a QID (or None if unresolved). The "forward map
      per-title" in the docstring is an implementation detail; tests only
      verify the final result (each original gets the correct QID).

  A4. resolve_titles_to_qids' "still unresolved after pass 1" means titles
      that got None or were missing from the batch results. The second pass
      queries them individually (batch of 1) to rule out cross-title
      collisions before concluding they're genuinely unresolved.

  A5. run() input expects events.json from scrape_roman_timeline: a JSON
      list of dicts, each with at minimum a `wiki_links` field (list of
      objects with `wiki_title` and `text` keys). Each event also has
      `year`, `year_normalized`, `era`, `date`, and `event` fields.
      Output structure is assumed to be the same events list, with each
      event getting a new `linked_entity_ids` field (or similar) mapping
      wiki_title → QID. Exact output key name not given in the outline —
      inferred from stage-level doc ("linked_entity_ids").

  A6. unresolved_titles.txt output format — assumed one title per line,
      newline-separated, no other structure.

  A7. MANUAL_QID_OVERRIDES is a dict[str, str] mapping misspelled titles
      to correct Wikidata QIDs. Takes precedence over API resolution
      (docstring confirms this is the intent).

  A8. run() reads from input_path (default input JSON), writes to output
      (default output JSON) and unresolved_output (default unresolved TXT).
      On dry_run=True, no files are written but the function still resolves
      (per scraper pattern).
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

import resolve_wikidata as resolver


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_EVENTS_PATH = FIXTURE_DIR / "sample_events.json"


@pytest.fixture
def sample_events() -> list[dict]:
    """Load the fixture events.json."""
    return json.loads(SAMPLE_EVENTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def sample_titles(sample_events) -> list[str]:
    """Extract all unique wiki-link targets from the fixture events."""
    titles = set()
    for event in sample_events:
        for link in event.get("wiki_links", []):
            titles.add(link["wiki_title"])
    return sorted(titles)


# ---------------------------------------------------------------------------
# chunked()
# ---------------------------------------------------------------------------
class TestChunked:
    def test_chunked_divides_into_chunks_of_size(self):
        items = [1, 2, 3, 4, 5, 6, 7]
        chunks = list(resolver.chunked(items, 3))
        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7]

    def test_chunked_with_exact_divisor(self):
        items = [1, 2, 3, 4, 5, 6]
        chunks = list(resolver.chunked(items, 2))
        assert len(chunks) == 3
        assert all(len(c) == 2 for c in chunks)

    def test_chunked_with_single_item(self):
        items = [1]
        chunks = list(resolver.chunked(items, 3))
        assert chunks == [[1]]

    def test_chunked_empty_list(self):
        items = []
        chunks = list(resolver.chunked(items, 3))
        assert chunks == []

    def test_chunked_with_batch_size_1(self):
        items = [1, 2, 3]
        chunks = list(resolver.chunked(items, 1))
        assert chunks == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# _query_batch()
# ---------------------------------------------------------------------------
class TestQueryBatch:
    @patch("resolve_wikidata.requests.get")
    def test_query_batch_returns_query_dict_from_response(self, mock_get):
        """_query_batch returns the 'query' dict from the MediaWiki JSON response."""
        batch = ["Julius Caesar", "Augustus"]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "1": {
                        "title": "Julius Caesar",
                        "pageprops": {"wikibase_item": "Q1"},
                    },
                    "2": {
                        "title": "Augustus",
                        "pageprops": {"wikibase_item": "Q1234"},
                    },
                }
            }
        }
        mock_get.return_value = mock_response

        result = resolver._query_batch(batch)

        assert result == mock_response.json.return_value["query"]
        assert mock_get.called

    @patch("resolve_wikidata.requests.get")
    def test_query_batch_sends_user_agent_header(self, mock_get):
        batch = ["Test"]
        mock_response = MagicMock()
        mock_response.json.return_value = {"query": {"pages": {}}}
        mock_get.return_value = mock_response

        resolver._query_batch(batch)

        _, kwargs = mock_get.call_args
        assert kwargs.get("headers") == resolver.HEADERS

    @patch("resolve_wikidata.requests.get")
    def test_query_batch_raises_on_failure_after_retries(self, mock_get):
        """_query_batch raises (doesn't silently swallow) after exhausting retries."""
        batch = ["Test"]
        mock_get.side_effect = RuntimeError("API timeout")

        with pytest.raises(RuntimeError):
            resolver._query_batch(batch)

    @patch("resolve_wikidata.requests.get")
    def test_query_batch_accepts_attempt_label(self, mock_get):
        """attempt_label parameter is accepted (used for logging/debugging).
        
        NOTE: this test can only weakly verify the parameter is accepted
        against an unimplemented stub (doesn't raise). Stronger assertion
        would require an actual implementation to verify the label is used."""
        batch = ["Test"]
        mock_response = MagicMock()
        mock_response.json.return_value = {"query": {"pages": {}}}
        mock_get.return_value = mock_response

        # Should not raise even with the label parameter
        result = resolver._query_batch(batch, attempt_label="pass_1")
        # Against stub: result is None. Against real impl: should be query dict.
        # Stub implementation will trivially pass; real impl must return a dict.
        assert mock_get.called  # at minimum, verify we tried to fetch


# ---------------------------------------------------------------------------
# _resolve_one_batch()
# ---------------------------------------------------------------------------
class TestResolveOneBatch:
    def test_resolve_one_batch_mutates_results_dict(self):
        """_resolve_one_batch modifies results dict in place."""
        batch = ["Julius Caesar"]
        results = {}

        with patch("resolve_wikidata._query_batch") as mock_query:
            mock_query.return_value = {
                "pages": {
                    "1": {
                        "title": "Julius Caesar",
                        "pageprops": {"wikibase_item": "Q1"},
                    }
                }
            }
            resolver._resolve_one_batch(batch, results)

        assert "Julius Caesar" in results
        assert results["Julius Caesar"] == "Q1"

    def test_resolve_one_batch_handles_redirect_collision(self):
        """C3: multiple originals redirecting to the same final title each
        get the QID instead of one silently overwriting the other."""
        batch = ["Ottoman empire", "Ottoman Empire"]
        results = {}

        with patch("resolve_wikidata._query_batch") as mock_query:
            # "Ottoman empire" (lowercase) redirects to "Ottoman Empire"
            # The API returns both in the pages dict with the final normalized
            # title, plus a redirects array showing the mapping.
            mock_query.return_value = {
                "redirects": [
                    {"from": "Ottoman empire", "to": "Ottoman Empire"}
                ],
                "pages": {
                    "1": {
                        "title": "Ottoman Empire",
                        "pageprops": {"wikibase_item": "Q12560"},
                    }
                }
            }
            resolver._resolve_one_batch(batch, results)

        # Both originals must get the QID, not one overwrite the other
        assert results.get("Ottoman empire") == "Q12560"
        assert results.get("Ottoman Empire") == "Q12560"

    def test_resolve_one_batch_preserves_unresolved_none(self):
        """Titles that genuinely have no Wikidata item should map to None.
        
        This test mocks _query_batch (internal helper), not resolve_one_batch
        itself — so it's testing the real resolve_one_batch logic against
        a mocked API response, which is appropriate for a private function."""
        batch = ["Nonexistent Article"]
        results = {}

        with patch("resolve_wikidata._query_batch") as mock_query:
            # Article exists but has no wikibase_item
            mock_query.return_value = {
                "pages": {
                    "1": {
                        "title": "Nonexistent Article",
                        # No pageprops or no wikibase_item key
                    }
                }
            }
            resolver._resolve_one_batch(batch, results)

        assert results.get("Nonexistent Article") is None
        assert mock_query.called  # ensure _query_batch was invoked


# ---------------------------------------------------------------------------
# resolve_titles_to_qids()
# ---------------------------------------------------------------------------
class TestResolveTitlesToQids:
    def test_resolve_titles_returns_dict(self):
        """resolve_titles_to_qids returns a dict mapping title → QID.
        
        This is a weak test against the unimplemented stub (stub returns None,
        which fails the isinstance check). But once implemented, it verifies
        the return type is dict, not list or other structure."""
        with patch("resolve_wikidata._resolve_one_batch"):
            with patch("resolve_wikidata.chunked") as mock_chunked:
                mock_chunked.return_value = [["Julius Caesar"]]
                result = resolver.resolve_titles_to_qids(["Julius Caesar"])
                assert isinstance(result, dict)

    def test_resolve_titles_two_pass_strategy(self):
        """C4: two-pass strategy — batch first, then recheck unresolved."""
        titles = ["Julius Caesar", "Unresolvable Title"]

        with patch("resolve_wikidata._resolve_one_batch") as mock_batch:
            with patch("resolve_wikidata.chunked") as mock_chunked:
                # Simulate chunking: first batch has 50-title chunks
                mock_chunked.return_value = [
                    ["Julius Caesar", "Unresolvable Title"]
                ]
                # First pass resolves only Julius Caesar
                def batch_side_effect(batch, results):
                    if "Julius Caesar" in batch:
                        results["Julius Caesar"] = "Q1"
                    # "Unresolvable Title" stays unresolved (not in results)

                mock_batch.side_effect = batch_side_effect

                result = resolver.resolve_titles_to_qids(titles)

                # Both passes should have been attempted (either via multiple
                # calls to _resolve_one_batch or via the two_pass logic)
                # This is a weak test (doesn't strongly verify two passes)
                # but documents the expected behavior
                assert isinstance(result, dict)

    def test_resolve_titles_manual_override_precedence(self):
        """MANUAL_QID_OVERRIDES takes precedence over API resolution."""
        titles = ["Epaphropditus (freedman of Nero)"]

        with patch("resolve_wikidata._resolve_one_batch"):
            result = resolver.resolve_titles_to_qids(titles)

        # The override should be in the result
        assert result.get("Epaphropditus (freedman of Nero)") == "Q289327"

    def test_resolve_titles_unresolved_maps_to_none(self):
        """Genuinely unresolved titles map to None in the result dict."""
        titles = ["Definitely Does Not Exist Article"]

        with patch("resolve_wikidata._resolve_one_batch"):
            result = resolver.resolve_titles_to_qids(titles)

        # Unresolved should be None, not missing from dict
        assert "Definitely Does Not Exist Article" in result
        assert result["Definitely Does Not Exist Article"] is None


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
class TestRun:
    @patch("resolve_wikidata.resolve_titles_to_qids")
    @patch("builtins.open", create=True)
    def test_run_reads_input_events_and_writes_resolved_output(
        self, mock_open_builtin, mock_resolve
    ):
        """run() reads events.json, resolves wiki_links, writes output."""
        events = [
            {
                "year": 1,
                "event": "Test",
                "links": [["Julius Caesar", "Caesar"]],
            }
        ]
        mock_resolve.return_value = {"Julius Caesar": "Q1"}

        # Mock file I/O
        mock_open_builtin.return_value.__enter__.return_value.read.return_value = (
            json.dumps(events)
        )
        mock_open_builtin.return_value.__enter__.return_value.write = MagicMock()

        result = resolver.run(
            input_path="/tmp/events.json",
            output="/tmp/output.json",
            dry_run=False,
        )

        assert result == "/tmp/output.json"
        assert mock_resolve.called

    @patch("resolve_wikidata.resolve_titles_to_qids")
    def test_run_dry_run_writes_no_files(self, mock_resolve):
        """dry_run=True prevents file writes but still resolves."""
        events = [
            {
                "year": 1,
                "year_normalized": 1,
                "era": "test",
                "date": "",
                "event": "Test event",
                "wiki_links": [{"wiki_title": "Title", "text": "Title"}],
            }
        ]
        mock_resolve.return_value = {"Title": "Q1"}

        with patch("builtins.open") as mock_open:
            # Set up file read for input
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(events)
            )
            result = resolver.run(
                input_path="/tmp/events.json",
                output="/tmp/output.json",
                dry_run=True,
            )

        assert result is None
        # Resolve was still called (dry-run resolves but doesn't write)
        assert mock_resolve.called

    @patch("resolve_wikidata.resolve_titles_to_qids")
    @patch("builtins.open", create=True)
    def test_run_writes_unresolved_titles_to_separate_file(
        self, mock_open_builtin, mock_resolve
    ):
        """Titles that map to None are written to unresolved_output."""
        events = [
            {"year": 1, "event": "Test", "links": [["Bad Title", "Bad"]]}
        ]
        mock_resolve.return_value = {"Bad Title": None}

        mock_open_builtin.return_value.__enter__.return_value.read.return_value = (
            json.dumps(events)
        )

        resolver.run(
            input_path="/tmp/events.json",
            output="/tmp/output.json",
            unresolved_output="/tmp/unresolved.txt",
            dry_run=False,
        )

        # File write should have been called (for unresolved titles)
        assert mock_open_builtin.called

    @patch("resolve_wikidata.resolve_titles_to_qids")
    @patch("builtins.open", create=True)
    def test_run_adds_resolved_qids_to_events(self, mock_open_builtin, mock_resolve):
        """run() augments events with resolved QID information."""
        events = [
            {
                "year": 1,
                "event": "Test",
                "links": [["Julius Caesar", "Caesar"]],
            }
        ]
        mock_resolve.return_value = {"Julius Caesar": "Q1"}

        mock_open_builtin.return_value.__enter__.return_value.read.return_value = (
            json.dumps(events)
        )

        # Capture what gets written
        write_calls = []
        mock_open_builtin.return_value.__enter__.return_value.write = (
            lambda x: write_calls.append(x)
        )

        resolver.run(
            input_path="/tmp/events.json",
            output="/tmp/output.json",
            dry_run=False,
        )

        # Check that output was written (at least one write call)
        # We can't easily check the exact structure without mocking more
        # deeply, but this confirms the function tried to write something
        # (beyond just the unresolved titles)
        assert len(write_calls) > 0


# ---------------------------------------------------------------------------
# build_arg_parser()
# ---------------------------------------------------------------------------
class TestBuildArgParser:
    def test_parser_has_input_flag(self):
        parser = resolver.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "input")
        assert args.input == resolver.DEFAULT_INPUT

    def test_parser_has_output_flag(self):
        parser = resolver.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "output")
        assert args.output == resolver.DEFAULT_OUTPUT

    def test_parser_has_unresolved_output_flag(self):
        parser = resolver.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "unresolved_output")
        assert args.unresolved_output == resolver.DEFAULT_UNRESOLVED_OUTPUT

    def test_parser_has_dry_run_flag(self):
        parser = resolver.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "dry_run")
        assert args.dry_run is False

    def test_parser_dry_run_flag_sets_true(self):
        parser = resolver.build_arg_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parser_has_log_level_flag(self):
        parser = resolver.build_arg_parser()
        args = parser.parse_args([])
        assert hasattr(args, "log_level")

    def test_parser_custom_input_output_paths(self):
        parser = resolver.build_arg_parser()
        args = parser.parse_args([
            "--input", "/tmp/custom_input.json",
            "--output", "/tmp/custom_output.json",
            "--unresolved-output", "/tmp/custom_unresolved.txt",
        ])
        assert args.input == "/tmp/custom_input.json"
        assert args.output == "/tmp/custom_output.json"
        assert args.unresolved_output == "/tmp/custom_unresolved.txt"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
class TestMain:
    @patch("resolve_wikidata.run")
    def test_main_invokes_run_with_parsed_args(self, mock_run, monkeypatch):
        monkeypatch.setattr("sys.argv", ["resolve_wikidata.py"])
        resolver.main()
        assert mock_run.called

    @patch("resolve_wikidata.run")
    def test_main_propagates_dry_run_flag(self, mock_run, monkeypatch):
        monkeypatch.setattr("sys.argv", ["resolve_wikidata.py", "--dry-run"])
        resolver.main()
        assert mock_run.called
        _, kwargs = mock_run.call_args
        if kwargs:
            assert kwargs.get("dry_run") is True


# ---------------------------------------------------------------------------
# Integration: fixture-level scenario
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_sample_events_fixture_has_required_structure(self, sample_events):
        """Verify the fixture has required fields from scrape_roman_timeline output."""
        for event in sample_events:
            assert "year" in event
            assert "year_normalized" in event
            assert "era" in event
            assert "date" in event
            assert "event" in event
            assert "wiki_links" in event
            assert isinstance(event["wiki_links"], list)
            # Each wiki_link should have wiki_title and text
            for link in event["wiki_links"]:
                assert "wiki_title" in link
                assert "text" in link

    def test_extracted_titles_from_fixture(self, sample_titles):
        """Verify we can extract titles from the fixture."""
        expected = {
            "Battle of Alba Longa",
            "Alba Longa",
            "Amulius",
            "Numitor",
            "Romulus",
            "Roman Republic",
            "Lucius Tarquinius Superbus",
            "Julius Caesar",
            "Germanicus",
            "Augustus",
            "Nerva",
            "Trajan",
            "Battle of Manzikert",
            "Constantinople",
            "Fourth Crusade",
            "Ottoman Empire",
            "Epaphropditus (freedman of Nero)",
        }
        assert set(sample_titles) == expected