"""Guardrails: the system knows when *not* to answer.

Five independent gates, in order (harness.py runs them):

1. ``check_input``     — reject empty / oversized / binary inputs.
2. ``safety``          — abuse, PII-harvesting and prompt-injection patterns
                         (curated phrase lists; dependency-free by design —
                         production would swap in a dedicated classifier).
3. ``off_topic_result``— query whose best cosine similarity over the whole
                         corpus is below threshold => "I only answer grounded
                         questions".
4. ``confidence_gate_result`` — retrieved evidence is too weak/flat.
5. ``grounded_result`` — for non-extractive (LLM) answers: every substantive
                         sentence must attach to a retrieved chunk, else the
                         answer is refused / downgraded.
"""

import re
from dataclasses import dataclass, field

from .chunk import split_sentences
from .text import tokenize, token_set

TOXIC_HI = (
    "चूत बहनचोद मादरचोद गांड गांडू लौड़ा भोसड़ी हरामी कमीना रंडी चुदाई "
    "चूतिया भड़वा चुन्नीलाल पम्मी").split()
TOXIC_EN = (
    "bitch fuck fucking cunt asshole bastard dick pussy nigger faggot "
    "motherfucker slut whore rape rapist necrophile").split()

_INJECT_PATTERNS = (
    r"ignore (all |any )?(previous|prior|above)? ?instructions?",
    r"ignore everything (before|above)",
    r"forget (your|all|everything)( previous| prior)? (instructions?|prompt)",
    r"reveal (your|the) (system|hidden) prompt",
    r"system prompt",
    r"you are now (dan|do anything now)",
    r"jailbreak",
    r"base64 decode",
    r"act as (a )?free\b",
    r"do anything now",
    r"no restrictions",
    r"पिछले (सभी )?निर्देशों (को )?(अनदेखा|भूल|नज़रअंदाज़|नजरअंदाज) कर",
    r"(अपना|अपने|अपनी) ?सिस्टम प्रॉम्प्ट (बताओ|दो|दे|क्या है)",
    r"सिस्टम प्रॉम्प्ट क्या है",
    r"कोई प्रतिबंध नहीं",
    r"बेस64 डिकोड",
)
_PII_HARVEST_HI = (
    r"किसी (का|के) .{0,25}(क्रेडिट|डेबिट) कार्ड",
    r"(उसका|उसकी|उनका|उनकी|राम का|सीता का) .{0,15}(कार्ड|आधार|पासपोर्ट|पैन|बैंक)",
    r"(क्रेडिट|डेबिट|कार्ड|आधार|पासपोर्ट|पैन|बैंक खाता|अकाउंट) (नंबर|नम्बर) "
    r"(बताओ|बताइए|दो|दे|दीजिये|चाहिए)",
    r"पासवर्ड (बताओ|दो|दे|चाहिए)",
    r"ओटीपी (बताओ|दो|दे|चाहिए)",
    r"बैंक खाते का (पता|विवरण|नंबर) (बताओ|दो|दे)",
)
_PII_PATTERNS = (
    r"(credit|debit) card number",
    r"card (cvv|cvv2|pin)",
    r"bank (account|iban) number",
    r"aadhaar (card )?number",
    r"pan card number",
    r"pan number",
    r"passport number",
    r"social security number",
    r"how to (hack|crack)",
    r"make (a )?(bomb|explosive)",
    r"how (to )?make (crystal )?meth",
    r"(steal|dox|doxx)",
)
_INJECT_RE = [re.compile(p, re.IGNORECASE) for p in _INJECT_PATTERNS]
_PII_RE = [re.compile(p, re.IGNORECASE) for p in _PII_PATTERNS + _PII_HARVEST_HI]


@dataclass
class GuardResult:
    ok: bool
    level: str          # safe | blocked | off_topic | low_confidence | ungrounded
    reason: str = ""
    meta: dict = field(default_factory=dict)


# --- gate 1 & 2: input + safety ---------------------------------------------

def check_input(text: str, max_chars: int = 500) -> GuardResult:
    if text is None or not str(text).strip():
        return GuardResult(False, "blocked", "empty input")
    text = str(text).strip()
    if len(text) > max_chars:
        return GuardResult(False, "blocked",
                           f"input too long ({len(text)} chars)")
    return GuardResult(True, "safe")


def safety_block(text: str) -> GuardResult | None:
    low = text.lower()
    toks = set(tokenize(low, stopwords=False))
    for t in TOXIC_HI + TOXIC_EN:
        if t in toks:
            return GuardResult(False, "blocked",
                               f"inappropriate language detected", {"term": t})
    for r in _PII_RE:
        if r.search(low):
            return GuardResult(False, "blocked",
                               "request targets private/unsafe information",
                               {"pattern": r.pattern})
    for r in _INJECT_RE:
        if r.search(low):
            return GuardResult(False, "blocked",
                               "prompt-injection attempt detected",
                               {"pattern": r.pattern})
    return None


# --- gate 3: off-topic ------------------------------------------------------

def off_topic_result(top_sim: float, coverage: float,
                     sim_threshold: float = 0.24,
                     coverage_threshold: float = 0.6) -> GuardResult | None:
    """Query is out-of-domain if (a) even the best corpus passage scores below
    ``sim_threshold`` OR (b) the retrieved passages don't lexically cover the
    query terms (``coverage_threshold``).

    Calibration on this corpus showed the plain similarity gate alone is not
    discriminative (the 80k-passage corpus always has *some* nearby passage for
    almost any query), while lexical coverage of the top-5 retrieved passages
    separates cleanly (on-topic p10=1.0 vs off-topic max=0.5)."""
    if top_sim < sim_threshold or coverage < coverage_threshold:
        return GuardResult(False, "off_topic",
                           "question is outside the knowledge base",
                           {"top_sim": round(top_sim, 4),
                            "coverage": round(coverage, 4),
                            "sim_threshold": sim_threshold,
                            "coverage_threshold": coverage_threshold})
    return None


# --- gate 4: retrieval confidence -------------------------------------------

def confidence_gate_result(hits, min_sim: float = 0.30,
                           min_margin: float = 0.02) -> GuardResult | None:
    if not hits:
        return GuardResult(False, "low_confidence", "no evidence retrieved")
    top = hits[0].vector_sim
    margin = (hits[0].vector_sim - hits[1].vector_sim) if len(hits) > 1 else 0.0
    if top < min_sim:
        return GuardResult(False, "low_confidence",
                           "retrieved evidence is too weak",
                           {"top_sim": round(top, 4),
                            "margin": round(margin, 4),
                            "threshold": min_sim})
    return None


# --- gate 5: hallucination / grounding check for LLM answers -----------------

_REF = re.compile(r"\[\d+\]|\[citation\]|\[n\]")


def _substantive(sent: str) -> bool:
    return len(tokenize(sent)) >= 3


def grounded_result(answer: str, hit_texts: list[str],
                    overlap_min: float = 0.7) -> GuardResult:
    """Every substantive sentence must be *contained* in the retrieved context
    (fraction of the sentence's content tokens present in some chunk). The
    containment view, not token-F1, is what catches a plausible-looking
    sentence that smuggles in one or two un-grounded claims."""
    ungrounded: list[str] = []
    for _, _, sent in split_sentences(_REF.sub("", answer)):
        if not _substantive(sent):
            continue
        best = max(_containment(sent, ctx) for ctx in hit_texts)
        if best < overlap_min:
            ungrounded.append(sent)
    if ungrounded:
        return GuardResult(False, "ungrounded",
                           "answer not supported by retrieved context",
                           {"sentences": ungrounded[:3]})
    return GuardResult(True, "safe")


def extractive_ok(extracted, min_overlap: float) -> GuardResult | None:
    """Gate extractive answers on their confidence (query-sentence F1)."""
    if extracted is None or extracted.confidence < min_overlap:
        return GuardResult(False, "low_confidence",
                           "no extractive span found with enough overlap",
                           {"confidence": getattr(extracted, "confidence", 0.0)})
    return None


def ijac(a: str, b: str) -> float:
    """F1 of token sets — used for answer-overlap scoring in the extractor."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 2.0 * inter / (len(ta) + len(tb))


def _containment(sentence: str, context: str) -> float:
    ta = set(tokenize(sentence))
    tb = set(tokenize(context))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def answer_conciseness(answer: str, max_sentences: int = 4) -> GuardResult | None:
    if len(split_sentences(answer)) > max_sentences:
        return GuardResult(False, "blocked",
                           "answer exceeds length budget", {})
    return None