"""
Phase 2a: prepare_embedding_input.py test fixtures.

Provides minimal, purpose-built chunk fixtures matching the real chunks.json schema:
- Nested events[] array per chunk
- linked_entity_ids as a dict {QID: QID} at chunk level, plus a list per event
- date field populated on ~36% of events, empty string otherwise
- Real multi-event chunk from actual dataset plus synthetic edge cases
"""

import pytest


@pytest.fixture
def real_multi_event_chunk():
    """
    Real multi-event chunk from chunks.json (year_normalized=-751, era="8th and 7th centuries BC").
    Both events have empty date fields, representing the common case (~64% of events).
    """
    return {
        "year_normalized": -751,
        "year": "752 BC",
        "era": "8th and 7th centuries BC",
        "dates": [],
        "events": [
            {
                "year": -751,
                "year_display": "752 BC",
                "year_normalized": -751,
                "era": "8th and 7th centuries BC",
                "date": "",
                "event": "Romulus, first king of Rome, celebrates the first Roman triumph after his victory over the Caeninenses, following the Rape of the Sabine Women. He celebrates a further triumph later in the year over the Antemnates.",
                "links": [
                    ["Romulus", "Romulus"],
                    ["ancient Rome", "Rome"],
                    ["Roman triumph", "Roman triumph"],
                    ["Caenina (Town)", "Caeninenses"],
                    ["Rape of the Sabine Women", "Rape of the Sabine Women"],
                    ["Antemnae", "Antemnates"],
                ],
                "linked_entity_ids": [
                    "Q1048568",
                    "Q1747689",
                    "Q2186",
                    "Q2260505",
                    "Q571799",
                    "Q657438",
                ],
            },
            {
                "year": -751,
                "year_display": "752 BC",
                "year_normalized": -751,
                "era": "8th and 7th centuries BC",
                "date": "",
                "event": "Rome's first colonies were established.",
                "links": [["Colonia (Roman)", "colonies"]],
                "linked_entity_ids": ["Q756780"],
            },
        ],
        "linked_entity_ids": {
            "Q1048568": "Q1048568",
            "Q1747689": "Q1747689",
            "Q2186": "Q2186",
            "Q2260505": "Q2260505",
            "Q571799": "Q571799",
            "Q657438": "Q657438",
            "Q756780": "Q756780",
        },
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }


@pytest.fixture
def single_event_chunk_with_date():
    """
    Single-event chunk with a populated date field (the ~36% case).
    Tests that date-conditional formatting works and single-event chunks
    follow the same prefix rule as multi-event chunks.
    """
    return {
        "year_normalized": -753,
        "year": "753 BC",
        "era": "8th and 7th centuries BC",
        "dates": [],
        "events": [
            {
                "year": -753,
                "year_display": "753 BC",
                "year_normalized": -753,
                "era": "8th and 7th centuries BC",
                "date": "21 April",
                "event": "Rome was founded. According to Roman legend, Romulus was the founder and first King of Rome.",
                "links": [
                    ["Romulus", "Romulus"],
                    ["King of Rome", "King"],
                    ["Rome", "Rome"],
                ],
                "linked_entity_ids": ["Q1405", "Q1747689", "Q220"],
            }
        ],
        "linked_entity_ids": {
            "Q1405": "Q1405",
            "Q1747689": "Q1747689",
            "Q220": "Q220",
        },
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }


@pytest.fixture
def multi_event_chunk_mixed_dates():
    """
    Multi-event chunk where one event has a date and one doesn't.
    Tests that date-conditional formatting applies independently per event,
    and that linked_entity_ids union is correct across events with different date presence.
    """
    return {
        "year_normalized": 79,
        "year": "79 AD",
        "era": "Imperial",
        "dates": ["August 24"],
        "events": [
            {
                "year": 79,
                "year_display": "79 AD",
                "year_normalized": 79,
                "era": "Imperial",
                "date": "August 24",
                "event": "Eruption of Mount Vesuvius destroys Pompeii and Herculaneum.",
                "links": [
                    ["Mount Vesuvius", "Mount Vesuvius"],
                    ["Pompeii", "Pompeii"],
                    ["Herculaneum", "Herculaneum"],
                ],
                "linked_entity_ids": ["Q4406", "Q48438", "Q160460"],
            },
            {
                "year": 79,
                "year_display": "79 AD",
                "year_normalized": 79,
                "era": "Imperial",
                "date": "",
                "event": "Emperor Titus ascends to the throne after the death of Vespasian.",
                "links": [
                    ["Titus", "Titus"],
                    ["Roman emperor", "Emperor"],
                    ["Vespasian", "Vespasian"],
                ],
                "linked_entity_ids": ["Q1405", "Q12140", "Q1401"],
            },
        ],
        "linked_entity_ids": {
            "Q4406": "Q4406",
            "Q48438": "Q48438",
            "Q160460": "Q160460",
            "Q1405": "Q1405",
            "Q12140": "Q12140",
            "Q1401": "Q1401",
        },
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }


@pytest.fixture
def multi_event_chunk_overlapping_entities():
    """
    Multi-event chunk where one QID appears in both events' linked_entity_ids.
    Tests that the union is correctly deduped (QID appears once in output, not twice).
    """
    return {
        "year_normalized": 410,
        "year": "410 AD",
        "era": "Imperial",
        "dates": ["August 24"],
        "events": [
            {
                "year": 410,
                "year_display": "410 AD",
                "year_normalized": 410,
                "era": "Imperial",
                "date": "August 24",
                "event": "Visigoths under Alaric sack Rome.",
                "links": [
                    ["Visigoths", "Visigoths"],
                    ["Alaric I", "Alaric"],
                    ["Rome", "Rome"],
                ],
                "linked_entity_ids": ["Q103705", "Q106405", "Q220"],
            },
            {
                "year": 410,
                "year_display": "410 AD",
                "year_normalized": 410,
                "era": "Imperial",
                "date": "",
                "event": "The Roman Empire is plunged into chaos as Rome falls to barbarian invasion.",
                "links": [
                    ["Roman Empire", "Roman Empire"],
                    ["Rome", "Rome"],
                    ["Barbarian", "barbarian"],
                ],
                "linked_entity_ids": ["Q12544", "Q220", "Q105762"],
            },
        ],
        "linked_entity_ids": {
            "Q103705": "Q103705",
            "Q106405": "Q106405",
            "Q220": "Q220",
            "Q12544": "Q12544",
            "Q105762": "Q105762",
        },
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }


@pytest.fixture
def chunks_batch_for_uniqueness_check():
    """
    A small batch of chunks with distinct (year_normalized, era) pairs.
    Used to verify the transform doesn't accidentally produce duplicate IDs.
    """
    return [
        {
            "year_normalized": -753,
            "year": "753 BC",
            "era": "8th and 7th centuries BC",
            "dates": [],
            "events": [
                {
                    "year": -753,
                    "year_display": "753 BC",
                    "year_normalized": -753,
                    "era": "8th and 7th centuries BC",
                    "date": "21 April",
                    "event": "Rome was founded.",
                    "links": [],
                    "linked_entity_ids": ["Q220"],
                }
            ],
            "linked_entity_ids": {"Q220": "Q220"},
            "source": "Wikipedia: Timeline of Roman history",
            "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
        },
        {
            "year_normalized": -751,
            "year": "752 BC",
            "era": "8th and 7th centuries BC",
            "dates": [],
            "events": [
                {
                    "year": -751,
                    "year_display": "752 BC",
                    "year_normalized": -751,
                    "era": "8th and 7th centuries BC",
                    "date": "",
                    "event": "Romulus celebrates the first Roman triumph.",
                    "links": [],
                    "linked_entity_ids": ["Q1048568"],
                }
            ],
            "linked_entity_ids": {"Q1048568": "Q1048568"},
            "source": "Wikipedia: Timeline of Roman history",
            "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
        },
        {
            "year_normalized": 27,
            "year": "27 BC",
            "era": "Imperial",
            "dates": [],
            "events": [
                {
                    "year": 27,
                    "year_display": "27 BC",
                    "year_normalized": 27,
                    "era": "Imperial",
                    "date": "",
                    "event": "Augustus becomes emperor.",
                    "links": [],
                    "linked_entity_ids": ["Q1405"],
                }
            ],
            "linked_entity_ids": {"Q1405": "Q1405"},
            "source": "Wikipedia: Timeline of Roman history",
            "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
        },
    ]


@pytest.fixture
def chunk_with_empty_events():
    """
    Edge case: chunk with empty events[] array.
    Not a live case in current data (0 of 412 chunks), but defensive coverage.
    """
    return {
        "year_normalized": -999,
        "year": "1000 BC",
        "era": "Pre-republic",
        "dates": [],
        "events": [],
        "linked_entity_ids": {},
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }


# ============================================================================
# Expected output fixtures (for comparison against actual transform output)
# ============================================================================


@pytest.fixture
def expected_multi_event_chunk_output():
    """
    Expected flat schema output from transforming real_multi_event_chunk.
    Two events, both without dates, same year, different linked_entity_ids.
    """
    return {
        "id": "3fa85f64-5717-5770-b6fe-dfe6b8d29f63",  # uuid5 of "-751|8th and 7th centuries BC"
        "chunk_key": "y_-751__8th_and_7th_centuries_BC",
        "year": "752 BC",
        "year_normalized": -751,
        "era": "8th and 7th centuries BC",
        "text": (
            "752 BC: Romulus, first king of Rome, celebrates the first Roman triumph after his victory over the Caeninenses, following the Rape of the Sabine Women. He celebrates a further triumph later in the year over the Antemnates.\n"
            "752 BC: Rome's first colonies were established."
        ),
        "embedding_text": (
            "752 BC: Romulus, first king of Rome, celebrates the first Roman triumph after his victory over the Caeninenses, following the Rape of the Sabine Women. He celebrates a further triumph later in the year over the Antemnates.\n"
            "752 BC: Rome's first colonies were established."
        ),
        "linked_entity_ids": [
            "Q1048568",
            "Q1747689",
            "Q2186",
            "Q2260505",
            "Q571799",
            "Q657438",
            "Q756780",
        ],
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }


@pytest.fixture
def expected_single_event_with_date_output():
    """
    Expected flat schema output from transforming single_event_chunk_with_date.
    Single event with date field, formatted with conditional date inclusion.
    """
    return {
        "id": "uuid5(-753, '8th and 7th centuries BC')",
        "chunk_key": "y_-753__8th_and_7th_centuries_BC",
        "year": "753 BC",
        "year_normalized": -753,
        "era": "8th and 7th centuries BC",
        "text": "753 BC (21 April): Rome was founded. According to Roman legend, Romulus was the founder and first King of Rome.",
        "embedding_text": "753 BC (21 April): Rome was founded. According to Roman legend, Romulus was the founder and first King of Rome.",
        "linked_entity_ids": ["Q1405", "Q1747689", "Q220"],
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }


@pytest.fixture
def expected_multi_event_mixed_dates_output():
    """
    Expected flat schema output from transforming multi_event_chunk_mixed_dates.
    Two events, one with date, one without — each formatted independently.
    """
    return {
        "id": "uuid5(79, 'Imperial')",
        "chunk_key": "y_79__Imperial",
        "year": "79 AD",
        "year_normalized": 79,
        "era": "Imperial",
        "text": (
            "79 AD (August 24): Eruption of Mount Vesuvius destroys Pompeii and Herculaneum.\n"
            "79 AD: Emperor Titus ascends to the throne after the death of Vespasian."
        ),
        "embedding_text": (
            "79 AD (August 24): Eruption of Mount Vesuvius destroys Pompeii and Herculaneum.\n"
            "79 AD: Emperor Titus ascends to the throne after the death of Vespasian."
        ),
        "linked_entity_ids": [
            "Q1401",
            "Q12140",
            "Q1405",
            "Q160460",
            "Q4406",
            "Q48438",
        ],  # sorted and deduped
        "source": "Wikipedia: Timeline of Roman history",
        "source_url": "https://en.wikipedia.org/wiki/Timeline_of_Roman_history",
    }
