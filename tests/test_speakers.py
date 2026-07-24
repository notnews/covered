"""Tests for the speaker-turn parser (measure a)."""

from covered import speakers


def _names(turns: list[speakers.Turn]) -> list[str]:
    return [t.speaker_raw for t in turns]


def test_basic_two_speakers_with_roles() -> None:
    text = (
        "WOLF BLITZER, CNN ANCHOR: Welcome to our coverage. "
        "We begin with breaking news tonight. "
        "JOHN KING, CNN CHIEF NATIONAL CORRESPONDENT: Thanks, Wolf. The story is developing."
    )
    turns = speakers.parse_turns(text, era_id="era3")
    assert _names(turns) == ["WOLF BLITZER", "JOHN KING"]
    assert turns[0].role_raw == "CNN ANCHOR"
    assert turns[0].staff_flag == "staff"
    assert turns[1].role_raw == "CNN CHIEF NATIONAL CORRESPONDENT"
    assert turns[1].staff_flag == "staff"
    assert turns[0].utterance.startswith("Welcome to our coverage.")
    assert turns[1].utterance.endswith("The story is developing.")


def test_offsets_map_back_to_source() -> None:
    text = "WOLF BLITZER, CNN ANCHOR: First. JANE DOE, ANALYST: Second point here."
    turns = speakers.parse_turns(text, era_id="era3")
    for t in turns:
        assert text[t.char_start : t.char_end] == t.utterance


def test_continuation_bare_surname_resolves_to_full_name() -> None:
    text = (
        "ANDERSON COOPER, CNN HOST: Good evening. We start with the storm. "
        "SENATOR JANE DOE, (D) NEW YORK: Thank you, Anderson. "
        "COOPER: Senator, what is your response to the criticism?"
    )
    turns = speakers.parse_turns(text, era_id="era3")
    assert _names(turns) == ["ANDERSON COOPER", "SENATOR JANE DOE", "COOPER"]
    # bare "COOPER" inherits the canonical name + staff flag from the roster
    assert turns[2].name_norm == "anderson cooper"
    assert turns[2].staff_flag == "staff"
    # the senator is a guest (external role)
    assert turns[1].staff_flag == "guest"


def test_unidentified_speakers_flagged_nonperson() -> None:
    text = (
        "WOLF BLITZER, CNN ANCHOR: We go live to the scene. "
        "UNIDENTIFIED MALE: I heard a loud bang. "
        "UNIDENTIFIED FEMALE: It was terrifying to watch."
    )
    turns = speakers.parse_turns(text, era_id="era3")
    assert _names(turns) == ["WOLF BLITZER", "UNIDENTIFIED MALE", "UNIDENTIFIED FEMALE"]
    assert turns[1].staff_flag == "nonperson"
    assert turns[2].staff_flag == "nonperson"


def test_stage_directions_are_not_turns() -> None:
    text = (
        "WOLF BLITZER, CNN ANCHOR: Here is the tape. "
        "(BEGIN VIDEO CLIP) "
        "JOHN SMITH, EYEWITNESS: It was loud and sudden. "
        "(END VIDEO CLIP) "
        "BLITZER: Powerful testimony there."
    )
    turns = speakers.parse_turns(text, era_id="era3")
    assert _names(turns) == ["WOLF BLITZER", "JOHN SMITH", "BLITZER"]
    assert turns[1].staff_flag == "guest"


def test_title_correspondent_is_staff() -> None:
    text = "DR. SANJAY GUPTA, CNN CHIEF MEDICAL CORRESPONDENT: The data shows a clear trend."
    turns = speakers.parse_turns(text, era_id="era3")
    assert len(turns) == 1
    assert turns[0].staff_flag == "staff"


def test_era1_newline_delimited_turns() -> None:
    # Old-format text joined turns with newlines and utterances need not end in
    # sentence punctuation; the parser must still split on the newline boundary.
    text = "BERNARD SHAW, CNN ANCHOR: tonight on the program\nJUDY WOODRUFF, CNN ANCHOR: a developing story"
    turns = speakers.parse_turns(text, era_id="era1")
    assert _names(turns) == ["BERNARD SHAW", "JUDY WOODRUFF"]


def test_clip_vs_live_tagging() -> None:
    text = (
        "WOLF BLITZER, CNN ANCHOR: Here is the president. "
        "(BEGIN VIDEO CLIP) "
        "DONALD TRUMP, PRESIDENT OF THE UNITED STATES: We will win bigly. "
        "(END VIDEO CLIP) "
        "BLITZER: That was the president speaking earlier."
    )
    turns = speakers.parse_turns(text, era_id="era3")
    mode = {t.speaker_raw: t.source_mode for t in turns}
    assert mode["WOLF BLITZER"] == "live"
    assert (
        mode["DONALD TRUMP"] == "clip"
    )  # inside (BEGIN VIDEO CLIP)...(END VIDEO CLIP)
    assert turns[-1].source_mode == "live"  # back to anchor after END


def test_clip_handles_videotape_and_audio_markers() -> None:
    text = (
        "ANNOUNCER LIVE: setup. "
        "(BEGIN AUDIO CLIP) JANE DOE, WITNESS: I heard it. (END AUDIO CLIP) "
        "(BEGIN VIDEOTAPE) JOHN ROE, ANALYST: My view. (END VIDEOTAPE)"
    )
    turns = speakers.parse_turns(text, era_id="era3")
    by = {t.speaker_raw: t.source_mode for t in turns}
    assert by["JANE DOE"] == "clip"
    assert by["JOHN ROE"] == "clip"


def test_empty_text_returns_no_turns() -> None:
    assert speakers.parse_turns("", era_id="era3") == []
    assert speakers.parse_turns("   \n  ", era_id="era3") == []
