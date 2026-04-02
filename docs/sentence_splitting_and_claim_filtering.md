# Sentence Splitting & Claim Filtering

How LLM-generated answers are broken into sentences, mapped to tokens, and filtered to identify only factual claims worth scoring for hallucination risk.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Sentence Splitting with pysbd](#2-sentence-splitting-with-pysbd)
3. [Token-to-Sentence Alignment](#3-token-to-sentence-alignment)
4. [Claim Filtering](#4-claim-filtering)
5. [How Claim Filtering Affects Scoring](#5-how-claim-filtering-affects-scoring)

---

## 1. Overview

When the SLT mode scores a generated answer, it needs to evaluate risk at the **sentence level**, not just for the answer as a whole. This requires three steps:

```
Generated answer text
    |
    v
[Sentence Splitting]  ──  pysbd library
    |                      Splits text into individual sentences
    v
[Token-to-Sentence Alignment]  ──  character-span midpoint mapping
    |                               Maps each token to its parent sentence
    v
[Claim Filtering]  ──  ClaimFilter (regex-based)
    |                   Identifies which sentences contain factual claims
    v
Only claim sentences are scored (B1 logit confidence + probe risk)
Non-claims are tagged level="none" and excluded from all aggregation
```

**Source files:**
- Sentence splitting: `backend/engine.py:176-179`
- Token alignment: `backend/engine.py:181-213`
- Claim filtering: `backend/claim_filter.py`
- Integration: `backend/engine.py:215-299` (`score_sentences`)

---

## 2. Sentence Splitting with pysbd

### What is pysbd?

**pysbd** (Pragmatic Sentence Boundary Disambiguation) is a Python library for sentence segmentation. It is rule-based (not ML-based), using a set of hand-crafted rules to handle abbreviations, decimal numbers, ellipsis, URLs, and other edge cases that trip up naive period-based splitting.

**Why pysbd over alternatives?**
- Zero overhead (no model loading, no GPU)
- Handles common LLM output patterns (numbered lists, abbreviations like "Dr.", "U.S.", decimal numbers)
- Deterministic — same input always gives same output

### Initialization

**Source:** `backend/engine.py:91`

```python
self._sentence_segmenter = pysbd.Segmenter(language='en', clean=False)
```

| Parameter | Value | Meaning |
|---|---|---|
| `language` | `'en'` | English language rules |
| `clean` | `False` | Do NOT remove newlines or extra whitespace — preserve original text for character-span alignment |

`clean=False` is important: if pysbd cleaned the text, the character positions would shift and break the token-to-sentence alignment step.

### Usage

**Source:** `backend/engine.py:176-179`

```python
def split_into_sentences(self, text):
    sentences = self._sentence_segmenter.segment(text)
    return [s.strip() for s in sentences if s.strip()]
```

**Post-processing:** Each sentence is stripped of leading/trailing whitespace, and empty strings are removed.

### Examples of pysbd Behavior

| Input | Output | Why |
|---|---|---|
| `"Paris is the capital of France. It has 2.1 million people."` | `["Paris is the capital of France.", "It has 2.1 million people."]` | Correctly splits on sentence period |
| `"Dr. Smith went to Washington. He arrived at 3 p.m."` | `["Dr. Smith went to Washington.", "He arrived at 3 p.m."]` | Does NOT split on "Dr." or "p.m." |
| `"The U.S. has 50 states. Each has a capital."` | `["The U.S. has 50 states.", "Each has a capital."]` | Handles "U.S." abbreviation |
| `"1. First point. 2. Second point."` | `["1. First point.", "2. Second point."]` | Handles numbered lists |

### When Sentence Splitting Is Skipped

**Source:** `backend/engine.py:225-226`

```python
sentences = self.split_into_sentences(answer_text)
if len(sentences) < 2:
    return []    # no sentence-level scoring for single-sentence answers
```

If the answer is a single sentence (common for short TriviaQA-style answers), sentence-level scoring is skipped entirely and an empty list is returned. The overall SLT probe score is used without per-sentence breakdown.

---

## 3. Token-to-Sentence Alignment

After splitting into sentences, each token must be mapped to its parent sentence. This is needed to:
1. Group per-token logits by sentence (for B1 confidence scoring)
2. Find the last token of each sentence (for per-sentence probe hidden state extraction)

### Algorithm

**Source:** `backend/engine.py:181-213`

**Step 1 — Get token character spans:**
```python
encoding = self.tokenizer(answer_text, return_offsets_mapping=True, add_special_tokens=False)
offset_mapping = encoding['offset_mapping']   # [(char_start, char_end), ...] per token
```

**Step 2 — Build sentence character spans:**
```python
sent_spans = []
pos = 0
for sent in sentences:
    start = answer_text.find(sent, pos)    # sequential search from last position
    if start == -1:
        start = pos                         # fallback if not found
    end = start + len(sent)
    sent_spans.append((start, end))
    pos = end
```

This uses `str.find()` with a moving `pos` cursor to handle repeated substrings correctly. Each sentence gets a `(char_start, char_end)` span.

**Step 3 — Map tokens to sentences via midpoint:**
```python
for tok_pos in range(n):
    char_start, char_end = offset_mapping[tok_pos]
    char_mid = (char_start + char_end) // 2        # midpoint of token's character span

    sent_idx = len(sentences) - 1                   # default: last sentence
    for si, (ss, se) in enumerate(sent_spans):
        if ss <= char_mid < se:                     # midpoint falls within this sentence
            sent_idx = si
            break

    token_sentence_idx.append(sent_idx)
```

**Why midpoint?** A token might span a sentence boundary (e.g., a space token between sentences). Using the character midpoint assigns the token to whichever sentence contains most of its characters.

### Visual Example

```
Answer: "Paris is the capital. It is in France."

Sentence spans:
  S0: "Paris is the capital."  chars [0, 21)
  S1: "It is in France."       chars [22, 38)

Token:     "Paris"  " is"  " the"  " capital"  "."  " It"  " is"  " in"  " France"  "."
Offsets:   [0,5)   [5,8)  [8,12)  [12,20)    [20,21) [22,24) [24,27) [27,30) [30,36) [36,37)
Midpoint:  2       6      10      16          20      23      25      28      33      36
Sentence:  S0      S0     S0      S0          S0      S1      S1      S1      S1      S1

Result: token_sentence_idx = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```

---

## 4. Claim Filtering

### Purpose

Not every sentence in an LLM response contains a factual claim worth evaluating for hallucination. Filler phrases, meta-commentary, hedging, and questions should be excluded from confidence scoring because:
- They don't make verifiable assertions
- Scoring them would dilute the signal from actual factual claims
- Their logit patterns don't reflect factual confidence

### Design Choice: Regex Over ML

**Source:** `backend/claim_filter.py:1-8`

```python
"""
Lightweight claim filter using regex patterns.
Inspired by semantic-entropy-probes/backend/claim_filter.py but without
the DeBERTa NLI dependency (~1GB model). Regex-only for zero overhead.
"""
```

The original SEP (Semantic Entropy Probes) project used a DeBERTa NLI model (~1GB) for claim detection. This project replaces it with regex patterns for:
- Zero additional memory usage
- No extra model loading time
- Instant evaluation (compiled regex vs. neural inference)
- Sufficient accuracy for the non-claim categories that LLMs commonly produce

### The `is_claim()` Method

**Source:** `backend/claim_filter.py:44-74`

The method returns `True` if a sentence likely contains a factual claim, `False` otherwise. It uses a **conservative default**: sentences are treated as claims unless explicitly matched by a non-claim rule.

```python
def is_claim(self, sentence: str) -> bool:
```

### Filter Pipeline (in order)

The filters are applied in sequence. If any filter matches, the sentence is classified as a non-claim and `False` is returned immediately.

#### Filter 1: Empty Text

```python
text = sentence.strip()
if not text:
    return False
```

Empty or whitespace-only strings are not claims.

#### Filter 2: Too Short (< 3 words)

```python
words = text.split()
if len(words) < 3 and not any(c.isdigit() for c in text):
    return False
```

Sentences with fewer than 3 words are likely fragments, not claims. **Exception:** Short text containing digits may still be a factual statement (e.g., "42." or "1776.").

| Input | Result | Reason |
|---|---|---|
| `"Yes."` | Non-claim | 1 word, no digits |
| `"Thank you."` | Non-claim | 2 words, no digits |
| `"In 1776."` | **Claim** | 2 words but contains digits |
| `"The answer is Paris."` | Proceeds to next filter | 4 words |

#### Filter 3: Filler Phrase Exact Match

```python
FILLER_PHRASES = {
    "sure", "okay", "thank you", "got it", "of course",
    "certainly", "absolutely", "no problem", "you're welcome",
    "great question", "good question",
}

normalized = text.lower().rstrip(".!,")
if normalized in FILLER_PHRASES:
    return False
```

The sentence is lowercased and trailing punctuation (`.`, `!`, `,`) is stripped before matching against the set.

| Input | Normalized | Result |
|---|---|---|
| `"Sure."` | `"sure"` | Non-claim |
| `"Of course!"` | `"of course"` | Non-claim |
| `"Great question."` | `"great question"` | Non-claim |
| `"Sure, the answer is Paris."` | `"sure, the answer is paris"` | Proceeds (not in set) |

#### Filter 4: Regex Non-Claim Patterns

```python
NON_CLAIM_PATTERNS = [...]
_compiled = [re.compile(p, re.IGNORECASE) for p in NON_CLAIM_PATTERNS]

for pattern in self._compiled:
    if pattern.match(text):    # match = anchored at start of string
        return False
```

All patterns use `re.IGNORECASE` and are checked with `.match()` (anchored at the start of the string, not searching the whole string).

**Pattern 1 — Meta/transitional statements:**
```python
r"^(here\s+(are|is|'s)|let\s+me|i'll|i\s+will|in\s+summary|to\s+summarize|to\s+conclude)"
```

| Matches | Does Not Match |
|---|---|
| "Here are the main points..." | "The results are here." |
| "Let me explain..." | "Please let the process run." |
| "I'll break this down..." | "They'll arrive tomorrow." |
| "In summary, the data shows..." | "The summary was published." |
| "To summarize the findings..." | |
| "To conclude, we found..." | |

**Pattern 2 — Headings/labels (short text ending with colon):**
```python
r"^.{1,60}:\s*$"
```

Matches any line up to 60 characters that ends with a colon followed by optional whitespace. These are typically section headers or labels in structured answers.

| Matches | Does Not Match |
|---|---|
| `"Key Points:"` | `"The ratio is 3:1 in favor."` (doesn't end at colon) |
| `"Step 1:"` | `"Note: this is important."` (has content after colon) |
| `"Overview:"` | A line longer than 60 characters |

**Pattern 3 — Questions:**
```python
r"^(what|who|where|when|why|how|is|are|do|does|can|could|would|should)\b.*\?\s*$"
```

Matches sentences starting with a question word and ending with a question mark.

| Matches | Does Not Match |
|---|---|
| `"What do you think?"` | `"What is known is that Paris is the capital."` (no `?`) |
| `"How does this work?"` | `"I wonder how this works."` (doesn't start with question word) |
| `"Is this correct?"` | |

**Pattern 4 — Hedging/opinion:**
```python
r"^(i\s+think|i\s+believe|in\s+my\s+opinion|it\s+seems|perhaps|maybe)"
```

| Matches | Does Not Match |
|---|---|
| `"I think the answer is Paris."` | `"Scientists think this is true."` |
| `"In my opinion, this is correct."` | `"The opinion was widely shared."` |
| `"Perhaps it was built in 1889."` | `"The perhaps surprising result..."` (not at start) |
| `"Maybe the answer is 42."` | |

**Pattern 5 — Advisory/disclaimer:**
```python
r"^(please\s+note|keep\s+in\s+mind|note\s+that|disclaimer|important)"
```

| Matches | Does Not Match |
|---|---|
| `"Please note that results may vary."` | `"The note was written in 1823."` |
| `"Keep in mind this is approximate."` | `"She kept the record in mind."` |
| `"Important: always verify sources."` | `"This is an important discovery."` (not at start) |

**Pattern 6 — Greeting/sign-off:**
```python
r"^(hi|hello|hey|dear|hope\s+this\s+helps|glad\s+to|happy\s+to)\b"
```

| Matches | Does Not Match |
|---|---|
| `"Hi! The capital of France is Paris."` | `"The high point was in 2020."` (`\b` prevents "hi" matching "high") |
| `"Hope this helps!"` | `"Historians hope to find more."` |
| `"Happy to help with that."` | |

**Pattern 7 — Offer to help more:**
```python
r"^(if\s+you\s+(have|need|want)|feel\s+free|don't\s+hesitate|let\s+me\s+know)\b"
```

| Matches | Does Not Match |
|---|---|
| `"If you need more details, ask."` | `"The results show if you compare..."` |
| `"Feel free to ask questions."` | |
| `"Don't hesitate to reach out."` | |
| `"Let me know if you need help."` | |

**Pattern 8 — Enumeration intros:**
```python
r"^(the\s+following|some\s+(examples?|key|important|notable))\b"
```

| Matches | Does Not Match |
|---|---|
| `"The following points are key..."` | `"Following the war, peace came."` (no "the") |
| `"Some examples include..."` | `"There are some results."` |
| `"Some key factors are..."` | |

#### Filter 5: Markdown Bold Headers

```python
if re.match(r"^\*\*[^*]+\*\*:?\s*$", text):
    return False
```

Matches bold markdown text that serves as a header, optionally followed by a colon.

| Matches | Does Not Match |
|---|---|
| `"**Overview:**"` | `"**Paris** is the capital of France."` (has content after bold) |
| `"**Key Points**"` | `"The **important** thing is..."` (not just a header) |

#### Filter 6: Numbered List Prefix Only

```python
if re.match(r"^\d+[.)]\s*$", text):
    return False
```

Matches bare numbered list markers with no content.

| Matches | Does Not Match |
|---|---|
| `"1."` | `"1. Paris is the capital."` (has content) |
| `"2)"` | `"There are 3 reasons."` |

#### Default: Treat as Claim

```python
return True  # Default: treat as claim (conservative)
```

If no filter matched, the sentence is conservatively treated as a factual claim. This ensures we don't accidentally skip sentences that should be scored.

---

## 5. How Claim Filtering Affects Scoring

### In B1 Logit Confidence Scoring

**Source:** `backend/engine.py:284-287`

```python
if not is_claim:
    level = "none"
    confidence = None
```

Non-claim sentences still have their logit statistics computed (`mean_chosen_logit`, `mean_logit_margin`) but their confidence is set to `None` and level to `"none"`. They are included in the output for transparency but excluded from aggregation.

### In Per-Sentence Probe Scoring

**Source:** `backend/engine.py:497-501, 543-548`

Non-claim sentences are excluded at two points:

**1. Hidden state extraction** — only claim sentence positions are passed for extraction:
```python
valid_positions = [
    p for si, p in enumerate(sent_end_positions)
    if p is not None and sentence_scores[si].get("is_claim", True)
]
```

**2. Probe scoring loop** — non-claims are skipped:
```python
if not sentence_scores[si].get("is_claim", True):
    sentence_scores[si]["energy_risk"] = None
    sentence_scores[si]["entropy_risk"] = None
    sentence_scores[si]["probe_risk"] = None
    continue
```

### In Risk Aggregation

**Source:** `backend/engine.py:609-610, 626-627`

Only claim sentences contribute to the aggregate risk score:
```python
per_sent_risks = [s["probe_risk"] for s in sentence_scores
                  if s.get("probe_risk") is not None and s.get("is_claim", True)]
```

### In Sentence Average Confidence

**Source:** `backend/engine.py:657-659`

```python
valid_confs = [s["confidence"] for s in sentence_scores
               if s["confidence"] is not None and s.get("is_claim", True)]
sentence_avg_confidence = mean(valid_confs) if valid_confs else None
```

### Summary of Claim vs Non-Claim Treatment

| Aspect | Claim Sentence | Non-Claim Sentence |
|---|---|---|
| **B1 confidence** | Computed (sigmoid + margin boost) | `None` |
| **B1 level** | "high" / "medium" / "low" | "none" |
| **Hidden state extracted** | Yes (at sentence-end token) | No (skipped) |
| **Probe energy_risk** | Computed | `None` |
| **Probe entropy_risk** | Computed | `None` |
| **Probe probe_risk** | Computed (AUROC-weighted) | `None` |
| **Included in aggregation** | Yes | No |
| **Included in sentence_avg_confidence** | Yes | No |
| **Returned in sentence_scores** | Yes (with all fields) | Yes (with None fields) |

### Example Output

For an answer: `"Sure! Paris is the capital of France. Let me know if you need more info."`

```python
sentence_scores = [
    {
        "text": "Sure!",
        "confidence": None,
        "level": "none",          # filler phrase
        "is_claim": False,
        "energy_risk": None,
        "entropy_risk": None,
        "probe_risk": None,
    },
    {
        "text": "Paris is the capital of France.",
        "confidence": 0.8721,     # B1 logit confidence
        "level": "high",          # may be overridden by probes
        "is_claim": True,
        "energy_risk": 0.1234,    # per-sentence energy probe
        "entropy_risk": 0.0987,   # per-sentence entropy probe
        "probe_risk": 0.1107,     # 0.515 * 0.0987 + 0.485 * 0.1234
    },
    {
        "text": "Let me know if you need more info.",
        "confidence": None,
        "level": "none",          # offer-to-help pattern
        "is_claim": False,
        "energy_risk": None,
        "entropy_risk": None,
        "probe_risk": None,
    },
]

# Only the middle sentence contributes to aggregation
# per_sent_risks = [0.1107]
# sentence_avg_confidence = 0.8721
```
