# Scoring Formulas and Thresholds

Reference for all hallucination risk scoring formulas used in the SemanticEnergy system.

---

## 1. Per-Sentence Logit Confidence (Fallback)

Used when SLT probe scores are not available (e.g., Full SE mode).

**Sigmoid on mean chosen-token logits:**

```
confidence = 1 / (1 + exp(-(mean_logit - 33.0) / 3.0))
```

- `LOGIT_SIGMOID_CENTER = 33.0` (logit value mapping to 50% confidence)
- `LOGIT_SIGMOID_SCALE = 3.0` (steepness)

**Margin boost** (when top1 - top2 logit margin > 3.0):

```
margin_boost = min(0.1, (mean_margin - 3.0) * 0.02)
confidence = min(1.0, confidence + margin_boost)
```

**Level thresholds (logit-based, used as fallback):**

| Confidence | Level | UI Badge |
|---|---|---|
| >= 0.60 | high | OK |
| >= 0.30 | medium | WARN |
| < 0.30 | low | RISK |

Non-claim sentences get level `"none"` and are not scored.

---

## 2. Per-Sentence Probe Risk (SLT Mode — Primary)

Each claim sentence gets dual-probe scoring from hidden states at the sentence-end token position within the shared forward pass.

**Energy probe** (layers `best_energy_slt_range`): outputs confidence, inverted for risk.

```
sent_energy_risk = 1.0 - energy_probe.predict_proba(hidden_state)[0, 1]
```

**Entropy probe** (layers `best_entropy_slt_range`): outputs risk directly.

```
sent_entropy_risk = entropy_probe.predict_proba(hidden_state)[0, 1]
```

**Combined probe risk** (energy weighted higher):

```
W_ENERGY  = 0.70
W_ENTROPY = 0.30

probe_risk = W_ENERGY * sent_energy_risk + W_ENTROPY * sent_entropy_risk
```

If only one probe produces a result, that probe's risk is used directly.

**Level thresholds (probe-based, overrides logit level in both directions):**

| probe_risk | Level | UI Badge |
|---|---|---|
| >= 0.65 | low | RISK (red) |
| >= 0.35 | medium | WARN (yellow) |
| < 0.35 | high | OK (green) |

---

## 3. Overall SLT Aggregate (Badge for Entire Answer)

The overall hallucination risk badge uses `combined_risk`, which blends the overall SLT probe scores with per-sentence probe risks.

**Overall SLT combined:**

```
slt_combined = 0.70 * energy_risk + 0.30 * entropy_risk
```

### Short Answers (token count <= 100)

**0 claim sentences:**

```
combined_risk = slt_combined
```

**1 claim sentence:**

```
combined_risk = 0.5 * slt_combined + 0.5 * per_sent_probe_risk
```

**2+ claim sentences:**

```
combined_risk = 0.5 * slt_combined + 0.5 * mean(per_sent_probe_risks)
```

### Long Answers (token count > 100)

```
slt_weight  = 0.15
max_weight  = 0.25 / (1 + log(n))        # n = number of claim sentences
mean_weight = 1.0 - slt_weight - max_weight

combined_risk = slt_weight  * slt_combined
              + max_weight  * max(per_sent_probe_risks)
              + mean_weight * mean(per_sent_probe_risks)
```

**Fallback** (no per-sentence probe data): `combined_risk = slt_combined`

### Overall Level Thresholds

| combined_risk | Level | Badge Color |
|---|---|---|
| < 0.35 | high | Green |
| < 0.65 | medium | Yellow |
| >= 0.65 | low | Red |

The badge displays confidence as `(1 - combined_risk) * 100%`.

---

## 4. Full SE Mode (Cluster Energy)

Uses semantic energy clustering, not SLT probes.

**Main confidence** = normalized energy of the dominant answer cluster.

| main_confidence | Level | Badge Color |
|---|---|---|
| > 0.80 | high | Green |
| > 0.50 | medium | Yellow |
| <= 0.50 | low | Red |

Sentence-level scoring in Full SE mode uses logit confidence only (Section 1 above).

---

## 5. TBG Mode (Pre-Generation)

Uses the same energy/entropy probes as SLT but on the prompt-only forward pass (before generation).

```
combined_risk = (energy_risk + entropy_risk) / 2.0
```

Same overall level thresholds as SLT (Section 3).

---

## 6. Sentence-Averaged Confidence

Displayed in the Details panel. Computed from logit confidence of claim sentences only:

```
sentence_avg_confidence = mean([s.confidence for s in claim_sentences if s.confidence is not None])
```
