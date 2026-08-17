# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Iterable

Timestamp = dict[str, float | int | str | list[str]]


def build_timestamps(text: str, token_timestamps: Iterable[dict]) -> dict[str, list[Timestamp]]:
    char = _chars_from_tokens(text, token_timestamps)
    word = _words_from_chars(text, char)
    segment = _segments_from_words(text, word)
    return {"char": char, "word": word, "segment": segment}


def offset_timestamps(timestamp: dict[str, list[Timestamp]], seconds: float) -> dict[str, list[Timestamp]]:
    if seconds == 0:
        return timestamp

    shifted: dict[str, list[Timestamp]] = {}
    for level, stamps in timestamp.items():
        shifted[level] = []
        for stamp in stamps:
            next_stamp = dict(stamp)
            start = next_stamp["start"]
            end = next_stamp["end"]
            assert not isinstance(start, list)
            assert not isinstance(end, list)
            next_stamp["start"] = round(float(start) + seconds, 3)
            next_stamp["end"] = round(float(end) + seconds, 3)
            shifted[level].append(next_stamp)
    return shifted


def merge_timestamps(
    parts: list[dict[str, list[Timestamp]]],
) -> dict[str, list[Timestamp]]:
    merged = {"char": [], "word": [], "segment": []}
    for part in parts:
        for level in merged:
            merged[level].extend(part.get(level, []))
    return merged


def _chars_from_tokens(text: str, token_timestamps: Iterable[dict]) -> list[Timestamp]:
    stamps: list[Timestamp] = []

    for token_stamp in token_timestamps:
        token = str(token_stamp.get("token", ""))
        clean_token = _clean_token(token)
        if not token:
            continue

        token_text = clean_token if clean_token else token
        if not token_text:
            continue

        start_offset = int(token_stamp.get("start_offset", 0))
        end_offset = int(token_stamp.get("end_offset", start_offset))
        stamps.append(
            {
                "char": [token_text],
                "token": token,
                "token_id": int(token_stamp["token_id"]) if "token_id" in token_stamp else -1,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "start": round(float(token_stamp["start"]), 3),
                "end": round(float(token_stamp["end"]), 3),
            }
        )

    return stamps


def _words_from_chars(text: str, char_timestamps: list[Timestamp]) -> list[Timestamp]:
    words: list[Timestamp] = []
    text_positions = _token_timestamps_by_text_position(text, char_timestamps)

    for match in re.finditer(r"\S+", text):
        covered = [stamp for position, stamp in text_positions if match.start() <= position < match.end()]
        if not covered:
            continue
        words.append(
            {
                "word": match.group(0),
                "start_offset": covered[0]["start_offset"],
                "end_offset": covered[-1]["end_offset"],
                "start": covered[0]["start"],
                "end": covered[-1]["end"],
            }
        )

    return words


def _segments_from_words(text: str, word_timestamps: list[Timestamp]) -> list[Timestamp]:
    if not word_timestamps:
        return []

    segments: list[Timestamp] = []
    current_words: list[Timestamp] = []

    for stamp in word_timestamps:
        current_words.append(stamp)
        if str(stamp["word"]).endswith((".", "!", "?")):
            segments.append(_segment(current_words))
            current_words = []

    if current_words:
        segments.append(_segment(current_words))

    if len(segments) == 1:
        segments[0]["segment"] = text

    return segments


def _segment(words: list[Timestamp]) -> Timestamp:
    return {
        "segment": " ".join(str(word["word"]) for word in words),
        "start_offset": words[0]["start_offset"],
        "end_offset": words[-1]["end_offset"],
        "start": words[0]["start"],
        "end": words[-1]["end"],
    }


def _clean_token(token: str) -> str:
    return token.replace("▁", " ").replace("<unk>", "")


def _advance_to_char(text: str, start: int, expected: str) -> int:
    expected = expected.casefold()
    for index in range(start, len(text)):
        if text[index].isspace():
            continue
        if text[index].casefold() == expected:
            return index
    return len(text)


def _token_timestamps_by_text_position(
    text: str,
    char_timestamps: list[Timestamp],
) -> list[tuple[int, Timestamp]]:
    by_position: list[tuple[int, Timestamp]] = []
    cursor = 0
    for stamp in char_timestamps:
        token_chars = _token_chars(stamp)
        token_start: int | None = None
        for char in token_chars:
            cursor = _advance_to_char(text, cursor, char)
            if cursor >= len(text):
                break
            token_start = cursor if token_start is None else token_start
            cursor += 1
        if token_start is not None:
            by_position.append((token_start, stamp))
    return by_position


def _token_chars(stamp: Timestamp) -> list[str]:
    raw_char = stamp["char"]
    text = "".join(raw_char) if isinstance(raw_char, list) else str(raw_char)
    return [char for char in text if not char.isspace()]
