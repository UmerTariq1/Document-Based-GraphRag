"""Common text-processing helpers shared by ingest and query pipelines."""
from typing import List
import re

from spacy.lang.en.stop_words import STOP_WORDS

from utils.ml_models import get_spacy_model


def clean_phrase(phrase: str) -> str:
    """Remove stop-words from a phrase and normalise whitespace."""
    tokens = [t for t in phrase.split() if t.lower() not in STOP_WORDS]
    return " ".join(tokens)


def fix_hyphenated_linebreaks(text: str) -> str:
    """Undo hyphen-newline line breaks that split words across lines.

    Example: "inter-\nnet" -> "internet"
    """
    # Removes hyphen + newline (optionally surrounded by whitespace)
    text = re.sub(r"-\s*\n\s*", "", text)
    # Replace remaining newlines with a single space for easier NLP parsing
    text = re.sub(r"\n+", " ", text)
    return text


def extract_keywords(text: str) -> List[str]:
    """Return a deduplicated list of key noun / verb phrases from *text*.

    The logic mirrors the previous implementation in both ingest.py and
    query.py but lives in a single place to avoid code duplication.
    """
    nlp = get_spacy_model()

    text = fix_hyphenated_linebreaks(text)
    doc = nlp(text)
    phrases = set()

    # 1. Cleaned noun chunks
    for chunk in doc.noun_chunks:
        phrase = clean_phrase(chunk.text.strip()).lower()
        if 1 <= len(phrase.split()) <= 5:
            phrases.add(phrase)

    # 2. Cleaned verb + object phrases
    for token in doc:
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in {"dobj", "attr", "pobj"}:
                    phrase = clean_phrase(f"{token.text} {child.text}")
                    if 1 <= len(phrase.split()) <= 5:
                        phrases.add(phrase)

    # Filter out phrases that are solely stop words and return
    keywords = [p.lower() for p in phrases if not all(w in STOP_WORDS for w in p.split())]
    return sorted(set(keywords))


def calculate_keyword_matches(
    main_keywords: List[str],
    related_keywords: List[str],
    excluded_keywords: List[str] | None = None,
):
    """Return (matching_keywords, count) after optional exclusion list."""
    main_set = {kw.lower() for kw in main_keywords}
    related_set = {kw.lower() for kw in related_keywords}
    if excluded_keywords:
        excl_set = {kw.lower() for kw in excluded_keywords}
        main_set -= excl_set
    matching = list(main_set & related_set)
    return matching, len(matching) 