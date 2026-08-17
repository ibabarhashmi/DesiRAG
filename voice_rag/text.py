"""Text utilities shared by retrieval and guardrails (no external tokenizer).

Unicode-aware tokenization keeps Devanagari (and other Indic) words intact:
a token is a run of letters + combining marks + digits, so matras (dependent
vowel signs, category Mn) never split a word the way regex ``\\w+`` does.
A small Hindi+English stopword list keeps BM25 focused on content words.
"""

import unicodedata

HINDI_STOP = (
    "में और का की हैं है था थी थे से के लिए पर यह ये वह वो जो को ने ही भी नहीं "
    "तो कि कर करने औऱ अक्सर एक दो अपनी अपने उनकी उनके इस उस जिन इन हैंं ").split()
EN_STOP = (
    "a an the of to in and or is are was were be been being for on with as by "
    "at from into about after before between over under what which who whom "
    "this that these those it its it's they them their he she his her we our "
    "you your i me my not no do does did will would can could should shall "
    "may might must there where when why how all any both each few more most "
    "other some such only own same too very just also than then so if then "
    "etc ie e.g vs").split()
STOPWORDS = frozenset(HINDI_STOP + EN_STOP)


def _iter_token_chars(text: str):
    """Yield character category groups; keep letters/marks/digits/underscore."""
    buf = []
    for ch in text:
        if ch == "_" or unicodedata.category(ch)[0] in "LMN":
            buf.append(ch)
        else:
            if buf:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def tokenize(text: str, stopwords: bool = True) -> list[str]:
    toks = [t.lower() for t in _iter_token_chars(text or "")]
    if stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def bigram_terms(text: str, stopwords: bool = True) -> list[str]:
    """Character bigrams per token (padded), flattened.

    Maps Hindi word-boundary variants onto the same terms: ``ताज महल`` and
    ``ताजमहल`` share the same bigram multiset for their common characters, so
    a lexical (BM25) matcher built on these terms tolerates compound and spaced
    spellings that an exact token matcher silently misses."""
    terms: list[str] = []
    for tok in tokenize(text, stopwords=stopwords):
        t = "~" + tok + "~"
        terms.extend(t[i:i + 2] for i in range(len(t) - 1))
    return terms


def token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())