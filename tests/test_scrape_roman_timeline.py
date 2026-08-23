"""
Contract-first tests for scrape_roman_timeline.py.

REVISION NOTE: this replaces an earlier version written against a bare
function outline with no docstrings. The full docstrings below changed
several contracts materially — most importantly normalize_year's return
type/semantics. Those corrections are called out explicitly (marked
"CORRECTED") so it's clear what changed and why, rather than silently
overwriting prior assumptions.

All bodies are still `pass` in the source shared so far, so this suite is
intentionally in red state — that's expected (contract-first, not
fitted-to-code).

REVISION 2 NOTE: the fixture used to be invented from imagination rather
than checked against the real article, which produced two real errors,
now fixed:
  (a) the table is actually THREE columns (Year | Date | Event) — the
      fixture had collapsed Year and Date into one "Date" cell.
  (b) forward-fill happens because a continuation row's year cell is
      genuinely ABSENT (fewer cells in that row) — not because of an
      explicit rowspan="N" HTML-table attribute, which is what the
      fixture used before. This was actually already stated plainly in
      parse_timeline's own docstring ("a row missing its year cell
      inherits the previous row's year") — misreading that, not just an
      invented fixture, caused the earlier version to test the wrong
      mechanism entirely. Confirmed against the real rendered article
      (fetched via web_search/web_fetch — raw wikitext itself isn't
      reachable from this sandbox's network allowlist, so the real
      column/blank-cell/wikilinked-year patterns were inferred from the
      rendered page, then reconstructed as fixture wikitext).

CONFIRMED FROM DOCSTRINGS (no longer assumptions):
  C1. normalize_year(year_str) -> Optional[int]
      Uses ASTRONOMICAL year numbering (no BC/AD discontinuity):
        "1 BC"   -> 0
        "2 BC"   -> -1
        "100 BC" -> -99
        "AD 1"   -> 1
        "AD 98"  -> 98
        "1071"   -> 1071
      Returns None (not a raised exception) if unparseable/empty.
      CORRECTED from an earlier assumed (int, era) tuple return with a
      simple sign flip (753 BC -> -753) and ValueError on failure — both
      wrong. This ripples into every parse_timeline year assertion below,
      since "753 BC" now normalizes to -752, not -753.

  C2. expand_dr_template converts {{dr|y|y|N|0|ysa}}:
        N < 0  -> "<abs(N)> BC"   e.g. -754 -> "754 BC"
        N >= 0 -> "AD <N>"        e.g.  98  -> "AD 98"
      Exact string format is now given directly in the docstring, so tests
      use exact equality rather than the earlier loose "contains" checks.

  C3. strip_cell_attributes strips a genuine leading "attrs | value" once,
      but only when the text before the first "|" contains NO "[[", "{{",
      or "<" (which would indicate real content/citations, not attributes).
      Cells starting with "[[" are never touched. Only a single split ever
      happens — content after the first pipe is not re-processed even if
      it contains further pipes.

  C4. extract_links has no awareness of <ref> tags itself — the caller
      (parse_timeline) is responsible for stripping citation footnotes
      BEFORE calling extract_links, so links embedded inside citations
      (e.g. a publisher link in a {{cite}} template) aren't treated as
      event-relevant entities.

  C5. split_cells treats "||" and "!!" as interchangeable separators (the
      article uses "!!" in header rows, "||" in data rows).

  C6. parse_timeline handles, per its docstring, a THREE-column table
      (Year | Date | Event) — confirmed against the real article, along
      with:
        - multiple tables, each with its own repeated header row
        - year cells as {{dr|...}} (BC-era sections, confirmed real) OR
          as plain [[AD 14]] / [[1071]]-style wikilinks (AD-era and later
          sections, confirmed real)
        - forward-fill via a genuinely MISSING year cell on continuation
          rows (CORRECTED — not a rowspan="N" attribute); the Date cell
          is independently often blank on the same rows, but blank ≠
          missing — Date is just an empty string, not an absent column
        - multi-line cell content (continuation lines with no leading "|")
        - a "|-" separator with a cell packed onto the same line, e.g.
          "|-| [[1204]] || || event text"
        - stray non-table text glued onto a "|-" line (vandalism), which
          is discarded

  C7. run(...) returns the events JSON output path as a string on success,
      and None when dry_run=True. (New — not confirmed before.)

REMAINING ASSUMPTIONS (still guessed, not given):
  A6. parse_timeline's exact dict keys are assumed to be "year" (the
      normalize_year int), "date" (original raw display text, often ""),
      "event" (cleaned prose), and "links" (list of (target, label)
      tuples from extract_links, applied post-ref-stripping per C4). The
      docstring confirms the *behaviors* but not the literal key names —
      easy to rename once confirmed.

  A6b. When a row has fewer than 3 cells, the missing cell is assumed to
       always be Year specifically (the leftmost column) — i.e. a 2-cell
       row is (Date, Event), never (Year, Event) with Date dropped. This
       matches the real article's pattern (Year is the rowspan-style
       column) but isn't literally stated in the docstring.

  A4b. strip_cell_attributes is assumed to require an "=" character in the
       candidate prefix to treat it as a genuine attribute list (all given
       examples are key="value" style) — not explicitly stated, so a
       prefix with a pipe but no "=" is assumed to be left untouched.

  A7. fetch_wikitext's exact MediaWiki response-parsing path is still
      assumed (conventional action=query&prop=revisions shape) — no new
      detail was given for this function beyond its one-line docstring.

  A9/A10. build_arg_parser()/main() flag names and calling convention are
      still assumed — no new detail given for these either.

  A11. _detect_project_root()'s docstring confirms the two supported
      layouts (src/ sibling vs. flat) and the src/-layout fallback, but
      the precedence when checking is assumed to be a simple existence
      check per layout, tested independently rather than for every
      possible combination (e.g. both existing at once is not tested).
"""
import csv
import json
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import scrape_roman_timeline as scraper


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scrape_sample.wikitext"


@pytest.fixture
def sample_wikitext() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _detect_project_root (A11)
# ---------------------------------------------------------------------------
class TestDetectProjectRoot:
    def test_detects_src_layout(self, tmp_path, monkeypatch):
        # <root>/src/scrape_roman_timeline.py, data/ sibling of src/
        root = tmp_path
        src_dir = root / "src"
        src_dir.mkdir()
        (root / "data").mkdir()
        monkeypatch.setattr(scraper, "__file__", str(src_dir / "scrape_roman_timeline.py"))
        assert scraper._detect_project_root() == root

    def test_detects_flat_layout(self, tmp_path, monkeypatch):
        # <root>/scrape_roman_timeline.py, data/ sibling of the script itself
        root = tmp_path
        (root / "data").mkdir()
        monkeypatch.setattr(scraper, "__file__", str(root / "scrape_roman_timeline.py"))
        assert scraper._detect_project_root() == root

    def test_falls_back_to_src_layout_when_neither_data_dir_exists(
        self, tmp_path, monkeypatch
    ):
        # Documented fallback: assume src/ layout scaffold on first run
        # before data/ has been created anywhere.
        root = tmp_path
        src_dir = root / "src"
        src_dir.mkdir()
        monkeypatch.setattr(scraper, "__file__", str(src_dir / "scrape_roman_timeline.py"))
        assert scraper._detect_project_root() == root


# ---------------------------------------------------------------------------
# normalize_year — astronomical numbering (C1, CORRECTED)
# ---------------------------------------------------------------------------
class TestNormalizeYear:
    def test_one_bc_is_astronomical_zero(self):
        assert scraper.normalize_year("1 BC") == 0

    def test_two_bc_is_astronomical_negative_one(self):
        assert scraper.normalize_year("2 BC") == -1

    def test_hundred_bc_is_astronomical_negative_ninety_nine(self):
        assert scraper.normalize_year("100 BC") == -99

    def test_ad_one_is_one(self):
        assert scraper.normalize_year("AD 1") == 1

    def test_ad_ninety_eight_is_ninety_eight(self):
        assert scraper.normalize_year("AD 98") == 98

    def test_bare_year_defaults_to_ad_no_adjustment(self):
        assert scraper.normalize_year("1071") == 1071

    def test_case_insensitive_and_whitespace_tolerant(self):
        assert scraper.normalize_year("509   bc") == -508

    def test_empty_string_returns_none(self):
        # NOTE: this test can't by itself distinguish "correctly returns
        # None for invalid input" from "always returns None" (e.g. an
        # unimplemented stub) — it only has teeth paired with the
        # non-None-expecting tests above, which do fail against a stub.
        assert scraper.normalize_year("") is None

    def test_unparseable_string_returns_none(self):
        # Same caveat as above — CORRECTED from an earlier assumed
        # ValueError, paired with the non-None tests for real coverage.
        assert scraper.normalize_year("not a year") is None

    def test_bare_zero_is_left_as_zero(self):
        # Bare "0" (no BC marker) is treated as a plain AD-style value with
        # no astronomical adjustment applied (adjustment only applies to
        # BC-marked input).
        assert scraper.normalize_year("0") == 0


# ---------------------------------------------------------------------------
# DR_TEMPLATE_RE / expand_dr_template (C2)
# ---------------------------------------------------------------------------
class TestDrTemplateExpansion:
    def test_dr_template_regex_captures_signed_year(self):
        match = scraper.DR_TEMPLATE_RE.search("{{dr|y|y|-754|0|ysa}}")
        assert match is not None
        assert match.group(1) == "-754"

    def test_dr_template_regex_does_not_match_missing_args(self):
        assert scraper.DR_TEMPLATE_RE.search("{{dr|y|y}}") is None

    def test_expand_negative_year_matches_docstring_example_exactly(self):
        # Exact example straight from the docstring.
        result = scraper.expand_dr_template("{{dr|y|y|-754|0|ysa}}")
        assert result == "754 BC"

    def test_expand_positive_year_matches_docstring_example_exactly(self):
        result = scraper.expand_dr_template("{{dr|y|y|98|0|ysa}}")
        assert result == "AD 98"

    def test_expand_within_surrounding_text(self):
        text = "Founded in {{dr|y|y|-716|0|ysa}} allegedly."
        result = scraper.expand_dr_template(text)
        assert result == "Founded in 716 BC allegedly."

    def test_expand_leaves_text_without_template_untouched(self):
        text = "509 BC: the Republic begins."
        assert scraper.expand_dr_template(text) == text

    def test_expanded_negative_output_round_trips_through_normalize_year(self):
        expanded = scraper.expand_dr_template("{{dr|y|y|-716|0|ysa}}")
        assert scraper.normalize_year(expanded) == -715  # astronomical: 1 - 716

    def test_expanded_positive_output_round_trips_through_normalize_year(self):
        expanded = scraper.expand_dr_template("{{dr|y|y|98|0|ysa}}")
        assert scraper.normalize_year(expanded) == 98


# ---------------------------------------------------------------------------
# LINK_RE / extract_links
# ---------------------------------------------------------------------------
class TestExtractLinks:
    def test_link_without_label(self):
        assert scraper.extract_links("[[Julius Caesar]]") == [
            ("Julius Caesar", "Julius Caesar")
        ]

    def test_link_with_label(self):
        assert scraper.extract_links("[[Roman Republic|the Republic]]") == [
            ("Roman Republic", "the Republic")
        ]

    def test_multiple_links_in_order(self):
        text = "[[Augustus]] was succeeded by [[Tiberius]]."
        assert scraper.extract_links(text) == [
            ("Augustus", "Augustus"),
            ("Tiberius", "Tiberius"),
        ]

    def test_no_links_returns_empty_list(self):
        assert scraper.extract_links("Plain text, no links here.") == []

    def test_anchor_links_are_not_matched(self):
        text = "[[Roman Republic#Fall|the fall of the Republic]]"
        assert scraper.extract_links(text) == []

    def test_does_not_itself_filter_citation_content(self):
        # C4: extract_links has no ref-awareness of its own — pins that the
        # caller (parse_timeline) must strip refs first. A future "fix"
        # inside extract_links itself would silently break this division
        # of responsibility without this test catching it.
        text = '<ref>{{cite book|publisher=[[Oxford University Press]]}}</ref>'
        assert scraper.extract_links(text) == [
            ("Oxford University Press", "Oxford University Press")
        ]


# ---------------------------------------------------------------------------
# split_cells — now includes "!!" as an interchangeable separator (C5)
# ---------------------------------------------------------------------------
class TestSplitCells:
    def test_splits_two_cells_on_double_pipe(self):
        assert scraper.split_cells("753 BC || Founding of Rome") == [
            "753 BC",
            "Founding of Rome",
        ]

    def test_splits_two_cells_on_double_bang(self):
        # Header rows use "!!" interchangeably with "||".
        assert scraper.split_cells("Date !! Event") == ["Date", "Event"]

    def test_splits_three_cells(self):
        result = scraper.split_cells("509 BC || Republic begins || citation")
        assert len(result) == 3

    def test_single_cell_no_separator(self):
        assert scraper.split_cells("just one cell") == ["just one cell"]

    def test_pipe_inside_wikilink_is_not_a_cell_separator(self):
        content = "509 BC || The Republic begins, see [[Roman Republic|link]]"
        result = scraper.split_cells(content)
        assert len(result) == 2
        assert result[1] == "The Republic begins, see [[Roman Republic|link]]"


# ---------------------------------------------------------------------------
# strip_cell_attributes (C3, refined)
# ---------------------------------------------------------------------------
class TestStripCellAttributes:
    def test_strips_leading_style_attribute(self):
        assert scraper.strip_cell_attributes(
            'style="text-align:center" | 494 BC'
        ) == "494 BC"

    def test_strips_leading_rowspan_and_valign_attributes(self):
        assert scraper.strip_cell_attributes(
            'rowspan="2" valign="top" | 753 BC'
        ) == "753 BC"

    def test_cell_without_attributes_is_unchanged(self):
        assert scraper.strip_cell_attributes("509 BC") == "509 BC"

    def test_citation_template_pipes_do_not_truncate_event_text(self):
        # Regression test for the exact previously-fixed bug, using the
        # docstring's own named-ref example shape.
        cell = (
            "The Republic is established"
            '<ref name="livy">{{cite web|url=http://example.com/livy'
            "|title=Ab Urbe Condita|author=Livy}}</ref>"
        )
        assert scraper.strip_cell_attributes(cell) == cell  # fully unchanged

    def test_cell_starting_with_wikilink_is_never_stripped(self):
        cell = "[[Roman Republic|the Republic]] founded amid crisis"
        assert scraper.strip_cell_attributes(cell) == cell

    def test_align_attribute_variant_is_also_stripped(self):
        assert scraper.strip_cell_attributes('align="center"| 44 BC') == "44 BC"

    def test_strips_only_the_first_attribute_boundary(self):
        # C3: only a single split ever happens — content after the first
        # pipe must not be re-processed even though it contains another
        # pipe of its own.
        cell = 'rowspan="2" valign="top" | 753 BC, later reconquered | again'
        result = scraper.strip_cell_attributes(cell)
        assert result == "753 BC, later reconquered | again"

    def test_prefix_without_equals_sign_is_not_treated_as_attribute(self):
        # A4b (still an assumption, flagged): genuine attributes look like
        # key="value"; a pipe-containing prefix with no "=" at all is
        # assumed not to be misidentified as one.
        cell = "maybe not an attribute | 753 BC"
        assert scraper.strip_cell_attributes(cell) == cell


# ---------------------------------------------------------------------------
# strip_wiki_markup
# ---------------------------------------------------------------------------
class TestStripWikiMarkup:
    def test_removes_ref_tags_and_content(self):
        text = 'The Republic begins<ref name="livy">{{cite web|url=x|title=y}}</ref>.'
        result = scraper.strip_wiki_markup(text)
        assert "<ref" not in result
        assert "cite web" not in result
        assert "The Republic begins" in result

    def test_removes_bold_and_italic_markup(self):
        text = "'''Rome''' was founded in ''legend''."
        result = scraper.strip_wiki_markup(text)
        assert "'''" not in result
        assert "''" not in result
        assert "Rome" in result
        assert "legend" in result

    def test_preserves_link_display_text(self):
        text = "Death of [[Julius Caesar]] on the Ides of March"
        result = scraper.strip_wiki_markup(text)
        assert "Julius Caesar" in result
        assert "[[" not in result and "]]" not in result


# ---------------------------------------------------------------------------
# parse_timeline — integration (C6; corrected 3-column fixture)
# ---------------------------------------------------------------------------
class TestParseTimeline:
    # NOTE: every expected `year` below is the astronomical value per C1,
    # not a simple BC sign flip. 44 BC -> -43 (= 1 - 44), not -44.

    def test_distinct_bc_years_each_get_their_own_event(self, sample_wikitext):
        # Table 1: every row has its OWN year cell (no forward-fill here) —
        # confirms sequential distinct-dr-template years aren't merged.
        events = scraper.parse_timeline(sample_wikitext)
        for expected_year in (-753, -752, -751, -508, -493):
            matches = [e for e in events if e["year"] == expected_year]
            assert len(matches) == 1, f"expected exactly one event for year {expected_year}"

    def test_forward_fill_applies_only_within_its_own_span(self, sample_wikitext):
        # -44 BC group: first row carries the year cell; the row right
        # after it has NO year cell at all (only Date+Event) and must
        # inherit -43 via forward-fill.
        events = scraper.parse_timeline(sample_wikitext)
        year_44 = [e for e in events if e["year"] == -43]
        assert len(year_44) == 2
        assert any("Assassination of Julius Caesar" in e["event"] for e in year_44)
        assert any("Second Triumvirate" in e["event"] for e in year_44)

    def test_forward_fill_does_not_bleed_into_next_distinct_year(self, sample_wikitext):
        # The AD14 group (2 rows) immediately follows the -44 BC group (2
        # rows) — two back-to-back forward-fill groups, catching
        # off-by-one span-boundary bugs between them.
        events = scraper.parse_timeline(sample_wikitext)
        year_14 = [e for e in events if e["year"] == 14]
        assert len(year_14) == 2
        assert any("Augustus" in e["event"] and "died" in e["event"] for e in year_14)
        assert any("Germanicus" in e["event"] for e in year_14)

    def test_missing_year_cell_maps_remaining_cells_to_date_and_event(
        self, sample_wikitext
    ):
        # A6b: a 2-cell continuation row is assumed to map to (Date, Event)
        # — Year is the cell that's missing, not Date.
        events = scraper.parse_timeline(sample_wikitext)
        germanicus = [e for e in events if "Germanicus" in e["event"]]
        assert len(germanicus) == 1
        assert germanicus[0]["year"] == 14
        assert germanicus[0]["date"] == ""

    def test_section_headers_never_produce_events(self, sample_wikitext):
        events = scraper.parse_timeline(sample_wikitext)
        for e in events:
            assert e["event"].strip() not in {"Kingdom", "Republic", "Medieval"}

    def test_table_header_row_never_produces_an_event(self, sample_wikitext):
        events = scraper.parse_timeline(sample_wikitext)
        for e in events:
            assert e["event"].strip().lower() not in scraper.HEADER_LABELS
            assert e["date"].strip().lower() not in scraper.HEADER_LABELS

    def test_citation_does_not_truncate_event_text(self, sample_wikitext):
        events = scraper.parse_timeline(sample_wikitext)
        republic_event = [e for e in events if "expulsion of" in e["event"]]
        assert len(republic_event) == 1
        assert "Tarquin the Proud" in republic_event[0]["event"]

    def test_formatting_attribute_on_year_cell_is_stripped_before_parsing(
        self, sample_wikitext
    ):
        # The style="..." attribute here is on the YEAR cell (wrapping the
        # dr-template), not the Date cell — must be stripped before the
        # dr-template is expanded/parsed, not leak into any output field.
        events = scraper.parse_timeline(sample_wikitext)
        year_494 = [e for e in events if e["year"] == -493]
        assert len(year_494) == 1
        assert "style=" not in year_494[0]["date"]
        assert "style=" not in year_494[0]["event"]

    def test_malformed_row_does_not_crash_and_is_excluded_or_flagged(
        self, sample_wikitext
    ):
        events = scraper.parse_timeline(sample_wikitext)  # must not raise
        for e in events:
            if "unclosed citation template" in e["event"]:
                assert isinstance(e["year"], int)

    def test_ad_wikilinked_year_parses_correctly(self, sample_wikitext):
        # C6: AD-era years are wikilinks, e.g. [[AD 14]] — confirmed
        # against the real article, not bare "AD 14" text.
        events = scraper.parse_timeline(sample_wikitext)
        died = [
            e for e in events if "Death of" in e["event"] or (
                "Augustus" in e["event"] and "died" in e["event"]
            )
        ]
        assert len(died) == 1
        assert died[0]["year"] == 14

        nerva = [e for e in events if "Nerva" in e["event"] and "Trajan" in e["event"]]
        assert len(nerva) == 1
        assert nerva[0]["year"] == 98
        assert nerva[0]["date"] == "27 January"

    def test_bare_bracket_year_link_is_parsed(self, sample_wikitext):
        # C6: later/medieval-section years are plain [[1071]]-style
        # wikilinks with no AD/BC marker at all (implicitly AD).
        events = scraper.parse_timeline(sample_wikitext)
        manzikert = [e for e in events if "Manzikert" in e["event"]]
        assert len(manzikert) == 1
        assert manzikert[0]["year"] == 1071

    def test_multiline_continuation_cell_is_joined(self, sample_wikitext):
        # C6: continuation lines with no leading "|" belong to the
        # previous cell and must be joined into the same event text.
        events = scraper.parse_timeline(sample_wikitext)
        manzikert = [e for e in events if "Manzikert" in e["event"]][0]
        assert "Anatolia continues here on a second physical line" in manzikert["event"]

    def test_packed_row_separator_and_cell_on_same_line(self, sample_wikitext):
        # C6: "|-| [[1204]] || || event text" packs the row separator and
        # all three cells (Year, blank Date, Event) onto one physical line.
        events = scraper.parse_timeline(sample_wikitext)
        sack = [e for e in events if "Sack of Constantinople" in e["event"]]
        assert len(sack) == 1
        assert sack[0]["year"] == 1204
        assert sack[0]["date"] == ""

    def test_vandalism_glued_to_row_separator_is_discarded(self, sample_wikitext):
        # C6: stray non-table text glued onto a "|-" line is discarded and
        # must not corrupt parsing of the row that follows it.
        events = scraper.parse_timeline(sample_wikitext)
        for e in events:
            assert "vandal editor scribbled" not in e["event"]
        fall = [
            e for e in events if "Fall of" in e["event"] and "Constantinople" in e["event"]
        ]
        assert len(fall) == 1
        assert fall[0]["year"] == 1453

    def test_links_inside_citations_are_excluded_from_event_links(self, sample_wikitext):
        # C4, integration-level: the publisher link buried inside the
        # <ref>{{cite book|...}}</ref> footnote must not appear in the
        # event's extracted entity links, even though real prose links in
        # the same cell (Roman Republic, Tarquin the Proud) must.
        events = scraper.parse_timeline(sample_wikitext)
        republic_event = [e for e in events if "expulsion of" in e["event"]][0]
        link_targets = [target for target, _ in republic_event.get("links", [])]
        assert "Oxford University Press" not in link_targets
        assert "Roman Republic" in link_targets
        assert "Lucius Tarquinius Superbus" in link_targets

    def test_links_field_present_for_a_wikilinked_year_event(self, sample_wikitext):
        events = scraper.parse_timeline(sample_wikitext)
        died = [e for e in events if "Augustus" in e["event"] and "died" in e["event"]][0]
        link_targets = [target for target, _ in died.get("links", [])]
        assert "Augustus" in link_targets

    def test_total_event_count_matches_fixture_row_count(self, sample_wikitext):
        # 13 unambiguous rows: 5 in table 1 (each with its own dr-template
        # year cell, plus the uncertain malformed row not counted here),
        # 5 in table 2 (2 forward-fill pairs + 1 standalone), 3 in table 3
        # (excluding the discarded vandalism line). Lower-bound check
        # rather than a hardcoded total, since the malformed row's fate
        # (kept-and-flagged vs. skipped) isn't specified.
        events = scraper.parse_timeline(sample_wikitext)
        assert len(events) >= 13


# ---------------------------------------------------------------------------
# fetch_wikitext (A7) — still highest uncertainty, no new detail given
# ---------------------------------------------------------------------------
class TestFetchWikitext:
    @patch("scrape_roman_timeline.requests.get")
    def test_sends_configured_user_agent_header(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "1": {"revisions": [{"slots": {"main": {"*": "sample wikitext"}}}]}
                }
            }
        }
        mock_get.return_value = mock_response

        scraper.fetch_wikitext("Timeline of Roman history")

        assert mock_get.called
        _, kwargs = mock_get.call_args
        assert kwargs.get("headers") == scraper.HEADERS

    @patch("scrape_roman_timeline.requests.get")
    def test_requests_the_given_page_title(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "1": {"revisions": [{"slots": {"main": {"*": "sample wikitext"}}}]}
                }
            }
        }
        mock_get.return_value = mock_response

        scraper.fetch_wikitext("Timeline of Roman history")

        _, kwargs = mock_get.call_args
        assert "Timeline of Roman history" in str(kwargs.get("params", {}))

    @patch("scrape_roman_timeline.requests.get")
    def test_returns_wikitext_content_from_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "1": {"revisions": [{"slots": {"main": {"*": "== Kingdom ==\nsample"}}}]}
                }
            }
        }
        mock_get.return_value = mock_response

        result = scraper.fetch_wikitext("Timeline of Roman history")
        assert result == "== Kingdom ==\nsample"


# ---------------------------------------------------------------------------
# run() (C7 — return value now confirmed)
# ---------------------------------------------------------------------------
class TestRun:
    @patch("scrape_roman_timeline.fetch_wikitext")
    def test_writes_json_csv_and_raw_output(self, mock_fetch, tmp_path, sample_wikitext):
        mock_fetch.return_value = sample_wikitext
        out_json = tmp_path / "events.json"
        out_csv = tmp_path / "events.csv"
        out_raw = tmp_path / "roman_timeline.wikitext"

        scraper.run(
            output=str(out_json),
            output_csv=str(out_csv),
            raw_output=str(out_raw),
            dry_run=False,
        )

        assert out_json.exists()
        assert out_csv.exists()
        assert out_raw.exists()

        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0

        with out_csv.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) > 1

        assert out_raw.read_text(encoding="utf-8") == sample_wikitext

    @patch("scrape_roman_timeline.fetch_wikitext")
    def test_run_returns_output_json_path_on_success(
        self, mock_fetch, tmp_path, sample_wikitext
    ):
        mock_fetch.return_value = sample_wikitext
        out_json = tmp_path / "events.json"
        out_csv = tmp_path / "events.csv"
        out_raw = tmp_path / "roman_timeline.wikitext"

        result = scraper.run(
            output=str(out_json),
            output_csv=str(out_csv),
            raw_output=str(out_raw),
            dry_run=False,
        )
        assert result == str(out_json)

    @patch("scrape_roman_timeline.fetch_wikitext")
    def test_dry_run_writes_no_files_but_still_fetches(
        self, mock_fetch, tmp_path, sample_wikitext
    ):
        mock_fetch.return_value = sample_wikitext
        out_json = tmp_path / "events.json"
        out_csv = tmp_path / "events.csv"
        out_raw = tmp_path / "roman_timeline.wikitext"

        scraper.run(
            output=str(out_json),
            output_csv=str(out_csv),
            raw_output=str(out_raw),
            dry_run=True,
        )

        # Asserts fetch_wikitext WAS called — an unimplemented/no-op run()
        # would also trivially produce "no files exist", so this guards
        # against a vacuous pass.
        assert mock_fetch.called
        assert not out_json.exists()
        assert not out_csv.exists()
        assert not out_raw.exists()

    @patch("scrape_roman_timeline.fetch_wikitext")
    def test_run_returns_none_on_dry_run(self, mock_fetch, tmp_path, sample_wikitext):
        # NOTE: also individually weak against a no-op stub, same as the
        # normalize_year None-checks above — paired with
        # test_run_returns_output_json_path_on_success for real coverage.
        mock_fetch.return_value = sample_wikitext
        result = scraper.run(
            output=str(tmp_path / "events.json"),
            output_csv=str(tmp_path / "events.csv"),
            raw_output=str(tmp_path / "roman_timeline.wikitext"),
            dry_run=True,
        )
        assert result is None

    @patch("scrape_roman_timeline.fetch_wikitext")
    def test_json_and_csv_have_matching_row_counts(
        self, mock_fetch, tmp_path, sample_wikitext
    ):
        mock_fetch.return_value = sample_wikitext
        out_json = tmp_path / "events.json"
        out_csv = tmp_path / "events.csv"
        out_raw = tmp_path / "roman_timeline.wikitext"

        scraper.run(
            output=str(out_json),
            output_csv=str(out_csv),
            raw_output=str(out_raw),
            dry_run=False,
        )

        data = json.loads(out_json.read_text(encoding="utf-8"))
        with out_csv.open(encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == len(data)


# ---------------------------------------------------------------------------
# build_arg_parser (A9, unchanged — no new detail given)
# ---------------------------------------------------------------------------
class TestBuildArgParser:
    def test_parses_with_no_args_using_defaults(self):
        parser = scraper.build_arg_parser()
        args = parser.parse_args([])
        assert args.output == scraper.DEFAULT_OUTPUT
        assert args.output_csv == scraper.DEFAULT_OUTPUT_CSV
        assert args.raw_output == scraper.DEFAULT_RAW_OUTPUT
        assert args.page_title == scraper.PAGE_TITLE
        assert args.dry_run is False

    def test_dry_run_flag_sets_true(self):
        parser = scraper.build_arg_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_custom_output_paths_are_honored(self):
        parser = scraper.build_arg_parser()
        args = parser.parse_args(
            ["--output", "/tmp/custom.json", "--output-csv", "/tmp/custom.csv"]
        )
        assert args.output == "/tmp/custom.json"
        assert args.output_csv == "/tmp/custom.csv"

    def test_custom_page_title_is_honored(self):
        parser = scraper.build_arg_parser()
        args = parser.parse_args(["--page-title", "Some Other Page"])
        assert args.page_title == "Some Other Page"

    def test_log_level_flag_is_accepted(self):
        # Mentioned in the module's usage docstring, previously untested.
        parser = scraper.build_arg_parser()
        args = parser.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# main() (A10, unchanged — no new detail given)
# ---------------------------------------------------------------------------
class TestMain:
    @patch("scrape_roman_timeline.run")
    def test_main_invokes_run_with_parsed_defaults(self, mock_run, monkeypatch):
        monkeypatch.setattr("sys.argv", ["scrape_roman_timeline.py"])
        scraper.main()

        assert mock_run.called
        call_args, call_kwargs = mock_run.call_args
        if call_kwargs:
            assert call_kwargs.get("dry_run", False) is False
        else:
            assert mock_run.call_count == 1

    @patch("scrape_roman_timeline.run")
    def test_main_propagates_dry_run_flag(self, mock_run, monkeypatch):
        monkeypatch.setattr("sys.argv", ["scrape_roman_timeline.py", "--dry-run"])
        scraper.main()

        assert mock_run.called
        _, call_kwargs = mock_run.call_args
        if call_kwargs:
            assert call_kwargs.get("dry_run") is True