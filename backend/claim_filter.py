"""
Lightweight claim filter using regex patterns.
Identifies whether a sentence contains a factual claim worth scoring
for hallucination risk, or is filler/transitional text that should be skipped.

Inspired by semantic-entropy-probes/backend/claim_filter.py but without
the DeBERTa NLI dependency (~1GB model). Regex-only for zero overhead.
"""

import re


class ClaimFilter:
    """Regex-based claim filter. No ML model needed."""

    FILLER_PHRASES = {
        "sure", "okay", "thank you", "got it", "of course",
        "certainly", "absolutely", "no problem", "you're welcome",
        "great question", "good question",
    }

    NON_CLAIM_PATTERNS = [
        # Meta/transitional: "Here are...", "Let me explain...", "In summary..."
        r"^(here\s+(are|is|'s)|let\s+me|i'll|i\s+will|in\s+summary|to\s+summarize|to\s+conclude)",
        # Headings/labels: short text ending with ":"
        r"^.{1,60}:\s*$",
        # Questions
        r"^(what|who|where|when|why|how|is|are|do|does|can|could|would|should)\b.*\?\s*$",
        # Hedging/opinion: "I think...", "In my opinion..."
        r"^(i\s+think|i\s+believe|in\s+my\s+opinion|it\s+seems|perhaps|maybe)",
        # Advisory: "Please note...", "Keep in mind..."
        r"^(please\s+note|keep\s+in\s+mind|note\s+that|disclaimer|important)",
        # Greeting/sign-off
        r"^(hi|hello|hey|dear|hope\s+this\s+helps|glad\s+to|happy\s+to)\b",
        # Offer to help more
        r"^(if\s+you\s+(have|need|want)|feel\s+free|don't\s+hesitate|let\s+me\s+know)\b",
        # Enumeration intros: "The following are...", "Some examples include..."
        r"^(the\s+following|some\s+(examples?|key|important|notable))\b",
    ]

    # Compiled patterns for performance
    _compiled = [re.compile(p, re.IGNORECASE) for p in NON_CLAIM_PATTERNS]

    def is_claim(self, sentence: str) -> bool:
        """Returns True if sentence likely contains a factual claim."""
        text = sentence.strip()

        if not text:
            return False

        # Too short to be a claim (unless contains numbers — could be a fact)
        words = text.split()
        if len(words) < 3 and not any(c.isdigit() for c in text):
            return False

        # Exact filler match
        normalized = text.lower().rstrip(".!,")
        if normalized in self.FILLER_PHRASES:
            return False

        # Regex patterns
        for pattern in self._compiled:
            if pattern.match(text):
                return False

        # Bold markdown headers: "**Something:**" or "**Something**"
        if re.match(r"^\*\*[^*]+\*\*:?\s*$", text):
            return False

        # Numbered list prefix only: "1." or "1)" with no substantial content
        if re.match(r"^\d+[.)]\s*$", text):
            return False

        return True  # Default: treat as claim (conservative)
