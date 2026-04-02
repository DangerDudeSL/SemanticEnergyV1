# Risk Aggregation Calculations Reference

This document comprehensively describes every calculation performed after obtaining probe bundle results, from raw hidden states through final aggregated risk scores.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Stage 1: Hidden State Extraction](#stage-1-hidden-state-extraction)
3. [Stage 2: Semantic Energy Pipeline (Multi-Sample)](#stage-2-semantic-energy-pipeline-multi-sample)
4. [Stage 3: Sentence-Level Logit Confidence (B1 Baseline)](#stage-3-sentence-level-logit-confidence-b1-baseline)
5. [Stage 4: TBG Probe Scoring (Pre-Generation)](#stage-4-tbg-probe-scoring-pre-generation)
6. [Stage 5: SLT Probe Scoring (Post-Generation)](#stage-5-slt-probe-scoring-post-generation)
7. [Stage 6: Token-Length Conditional Aggregation](#stage-6-token-length-conditional-aggregation)
8. [Constants and Thresholds Reference](#constants-and-thresholds-reference)
9. [Calculations Quick-Reference Table](#calculations-quick-reference-table)

---

## Pipeline Overview

There are three independent scoring paths, each exposed through a separate API endpoint:

| Endpoint | Mode | Model Calls | What It Measures |
|---|---|---|---|
| `/chat` | Semantic Energy (multi-sample clustering) | N generations + N*(N-1)/2 equivalence checks | Cluster-level energy distribution |
| `/score_fast_tbg` | TBG probes (pre-generation) | 1 forward pass (no generation) | Prompt-only risk prediction |
| `/score_fast_slt` | SLT probes (post-generation) | 1 generation + 1 forward pass | Post-generation per-sentence + overall risk |

All three share the B1 sentence-level logit confidence scoring as a sub-component (except TBG which has no generated answer).

---

## Stage 1: Hidden State Extraction

**Source:** `backend/engine.py:351-401` (`_extract_hidden_states`)

A single forward pass on `(prompt + answer)` with `output_hidden_states=True` extracts:

| Hidden State | Position | Shape | Description |
|---|---|---|---|
| `tbg_hidden` | `prompt_len - 1` (last prompt token) | `(num_layers, hidden_dim)` | Token Before Generation - captures the model's "readiness" state before generating |
| `slt_hidden` | `full_len - 2` (second-to-last generated token) | `(num_layers, hidden_dim)` | Second-to-Last Token - captures the model's state near the end of generation |
| `sent_hiddens` | Last token of each claim sentence | `list[(num_layers, hidden_dim)]` | Per-sentence hidden states at sentence-boundary positions |

**Layer slicing:** Each probe uses a contiguous window of layers (e.g., layers 17-21) selected during training for peak AUROC. The selected window is flattened into a single feature vector: `hidden[l0:l1, :].reshape(1, -1)` producing shape `(1, window_size * hidden_dim)`.

---

## Stage 2: Semantic Energy Pipeline (Multi-Sample)

**Source:** `backend/engine.py:15-48` (helper functions), `backend/app.py:133-232` (`/chat` endpoint)

This is the full multi-sample approach. Generate N responses, cluster them by semantic equivalence, then compute energy scores per cluster.

### Step 2.1: Token Probability Product

**Function:** `cal_probs(probs_list)` at `engine.py:30-31`

```
probs[i] = prod(probs_list[i])  =  p_t1 * p_t2 * ... * p_tK
```

- **Input:** `probs_list` = list of per-sample token probability lists
- **Output:** Single probability value per sample (product of all token probabilities)
- **Range:** [0, 1]

### Step 2.2: Boltzmann Logit Aggregation

**Function:** `cal_boltzmann_logits(logits_list)` at `engine.py:39-40`

```
logits[i] = -mean(logits_list[i])  =  -(1/K) * sum(logit_t1, logit_t2, ..., logit_tK)
```

- **Input:** `logits_list` = list of per-sample raw logit lists
- **Output:** Negated mean logit per sample
- **Range:** (-inf, 0)

**Alternative (unused by default):** `cal_fermi_dirac_logits` applies a Fermi-Dirac distribution transform before averaging: `E / (exp((E - mu) / kT) + 1)`. Activated by passing `fermi_mu` parameter.

### Step 2.3: Cluster-Level Aggregation

**Function:** `cal_cluster_ce(probs, logits, clusters)` at `engine.py:19-28`

```
normalized_probs = sum_normalize(probs)           # each p_i / sum(all p_i)

For each cluster c:
    probs_se[c]  = sum(normalized_probs[i] for i in cluster_c)
    logits_se[c] = -sum(logits[i] for i in cluster_c)
```

- **Probability aggregation:** Sums the normalized probability mass assigned to each cluster
- **Logit aggregation:** Sums the negated logit values for all samples within each cluster
- **Output:** Two parallel lists indexed by cluster ID

### Step 2.4: Energy Score Normalization

**Function:** `sum_normalize(lst)` at `engine.py:15-17`

```
cluster_energies[c] = logits_se[c] / sum(logits_se)
```

- **Output:** Normalized energy per cluster, values in [0, 1], all summing to 1.0
- **Interpretation:** The fraction of total "energy" assigned to each semantic cluster

### Step 2.5: Main Cluster Confidence

**Source:** `app.py:198-205`

```
main_cluster_idx = index of cluster containing sample 0
energy_score_raw = cluster_energies[main_cluster_idx]
```

- **Range:** [0, 1]
- **Orientation:** Higher = more confident (more energy concentrated in the main answer's cluster)

### Step 2.6: Confidence Level Bucketing (`/chat`)

```
> 0.80  ->  "high"
> 0.50  ->  "medium"
<= 0.50 ->  "low"
```

---

## Stage 3: Sentence-Level Logit Confidence (B1 Baseline)

**Source:** `backend/engine.py:215-299` (`score_sentences`)

No additional model calls. Purely post-processes per-token logit data from the generation step.

### Step 3.1: Token-to-Sentence Alignment

**Source:** `engine.py:181-213` (`align_tokens_to_sentences`)

- Split answer into sentences using `pysbd`
- For each token, compute character-span midpoint and map it to the containing sentence
- Output: `token_sentence_idx[i]` = which sentence token `i` belongs to

### Step 3.2: Group Metrics by Sentence

**Source:** `engine.py:233-244`

For each sentence `s`:
- `sent_logits[s]` = list of chosen-token logits for all tokens in sentence `s`
- `sent_margins[s]` = list of (top1_logit - top2_logit) margins for all tokens in sentence `s`

### Step 3.3: Sigmoid Logit-to-Confidence Mapping

**Source:** `engine.py:266-270`

```
mean_logit = mean(sent_logits[s])

confidence = 1 / (1 + exp(-(mean_logit - CENTER) / SCALE))
```

Where:
- `LOGIT_SIGMOID_CENTER = 33.0` (calibrated for Llama 3.1 8B 4-bit; logit value mapping to 50% confidence)
- `LOGIT_SIGMOID_SCALE = 3.0` (controls sigmoid steepness)

**Interpretation:** Typical raw chosen-token logits for this model range 25-45. The sigmoid maps this to a [0, 1] confidence score with 33.0 as the midpoint.

### Step 3.4: Margin-Based Confidence Boost

**Source:** `engine.py:272-275`

```
if mean_margin > 3.0:
    margin_boost = min(0.1, (mean_margin - 3.0) * 0.02)
    confidence = min(1.0, confidence + margin_boost)
```

- **Trigger:** Only when the average gap between the model's top-1 and top-2 logit choices exceeds 3.0
- **Effect:** Up to +0.1 boost, linearly scaled at rate 0.02 per unit of margin above 3.0
- **Cap:** Confidence cannot exceed 1.0

### Step 3.5: Claim Filtering

**Source:** `engine.py:256, 284-287`, `backend/claim_filter.py`

Non-claim sentences (filler, meta-commentary, hedging, questions) are detected via regex patterns:
- Their `level` is set to `"none"`
- Their `confidence` is set to `None`
- They are excluded from all downstream aggregation

### Step 3.6: Confidence Level Bucketing (B1)

```
>= 0.6  ->  "high"
>= 0.3  ->  "medium"
< 0.3   ->  "low"
```

### Step 3.7: Per-Sentence Output

```python
{
    "text": str,                    # sentence text
    "confidence": float | None,     # [0, 1] or None for non-claims
    "level": "high"|"medium"|"low"|"none",
    "num_tokens": int,
    "mean_chosen_logit": float,
    "mean_logit_margin": float | None,
    "is_claim": bool,
}
```

---

## Stage 4: TBG Probe Scoring (Pre-Generation)

**Source:** `backend/engine.py:403-459` (`score_with_tbg_probe`)

Single forward pass on prompt only (no generation). Fastest scoring mode.

### Step 4.1: Energy Probe

```
X = tbg_hidden[l0:l1, :].reshape(1, -1)            # Select layer window, flatten
X_scaled = energy_scaler.transform(X)               # StandardScaler normalization
energy_confidence = energy_probe.predict_proba(X_scaled)[0, 1]  # P(high_confidence)
energy_risk = 1.0 - energy_confidence               # Invert: risk = 1 - confidence
```

- **Layer range:** `probe_bundle["best_energy_tbg_range"]` (e.g., layers 28-32)
- **Probe type:** LogisticRegression trained on binarized energy teacher labels
- **Inversion:** The probe predicts P(confident), so risk = 1 - P(confident)

### Step 4.2: Entropy Probe

```
X = tbg_hidden[l0h:l1h, :].reshape(1, -1)
X_scaled = entropy_scaler.transform(X)
entropy_risk = entropy_probe.predict_proba(X_scaled)[0, 1]   # P(high_uncertainty)
```

- **Layer range:** `probe_bundle["best_entropy_tbg_range"]` (e.g., layers 21-25)
- **No inversion:** The probe directly predicts P(uncertain), which IS the risk

### Step 4.3: Combined Risk

```
combined_risk = (energy_risk + entropy_risk) / 2.0
```

Simple average of both probe risks.

### Step 4.4: Confidence Level (TBG)

```
< 0.35  ->  "high"    (low risk = high confidence)
< 0.65  ->  "medium"
>= 0.65 ->  "low"     (high risk = low confidence)
```

---

## Stage 5: SLT Probe Scoring (Post-Generation)

**Source:** `backend/engine.py:461-671` (`score_with_slt_probe`)

Generates one answer, then performs a second forward pass to extract hidden states. Most comprehensive scoring mode.

### Step 5.1: Overall SLT Probe Scores

**Source:** `engine.py:522-532`

Identical to TBG probes but using SLT (second-to-last token) hidden states and SLT-trained probes:

```
energy_risk = 1.0 - slt_energy_probe.predict_proba(scaled_X)[0, 1]
entropy_risk = slt_entropy_probe.predict_proba(scaled_X)[0, 1]
```

### Step 5.2: Per-Sentence Dual-Probe Scoring

**Source:** `engine.py:534-600`

For each claim sentence, extract hidden states at the sentence-end token position and run both probes:

```
sent_energy_risk  = 1.0 - energy_probe.predict_proba(scaled_X_e)[0, 1]
sent_entropy_risk = entropy_probe.predict_proba(scaled_X_h)[0, 1]
```

**AUROC-weighted combination:**

```
W_ENTROPY = 0.515    # derived from validation AUROC 0.773
W_ENERGY  = 0.485    # derived from validation AUROC 0.727

probe_risk = W_ENTROPY * sent_entropy_risk + W_ENERGY * sent_energy_risk
```

**Rationale:** Entropy probes showed higher AUROC (0.773 vs 0.727) during training validation, so they get proportionally more weight. The weights are normalized: 0.773/(0.773+0.727) = 0.515.

**Per-sentence level override:**

```
probe_risk >= 0.65  ->  override level to "low"
probe_risk >= 0.35  ->  override level to "medium"
probe_risk < 0.35   ->  keep existing logit-based level (don't downgrade)
```

Note: probes can only raise the risk level, never lower it below the B1 baseline.

---

## Stage 6: Token-Length Conditional Aggregation

**Source:** `engine.py:602-655`

The final combined_risk is computed differently depending on answer length, because the SLT token's representational quality degrades for long answers.

### Short Answers (<=100 tokens)

**Source:** `engine.py:606-623`

```
slt_combined = (energy_risk + entropy_risk) / 2.0
per_sent_risks = [s.probe_risk for s in claim_sentences]

if len(per_sent_risks) >= 2:
    mean_sent_risk = mean(per_sent_risks)
    combined_risk = 0.5 * slt_combined + 0.5 * mean_sent_risk
else:
    combined_risk = slt_combined
```

**Logic:**
- **0-1 claims:** SLT probe alone is sufficient (covers the whole short answer)
- **2+ claims:** SLT only represents the state at the end, so blend 50-50 with per-sentence average to account for all claims

### Long Answers (>100 tokens)

**Source:** `engine.py:624-648`

```
n = len(per_sent_risks)
max_sent_risk = max(per_sent_risks)
mean_sent_risk = mean(per_sent_risks)

slt_weight  = 0.15
max_weight  = 0.25 / (1.0 + ln(max(n, 1)))
mean_weight = 1.0 - slt_weight - max_weight

combined_risk = slt_weight  * entropy_risk      # SLT anchor (entropy only)
              + max_weight  * max_sent_risk      # Worst-case sentence
              + mean_weight * mean_sent_risk     # Average across sentences
```

**Design decisions:**
- **SLT weight = 0.15:** SLT is unreliable for long answers, so it gets minimal weight and only the entropy component is used (not energy)
- **Max weight decreases logarithmically:** `0.25 / (1 + ln(n))` ensures that with many claims, the worst-case sentence doesn't dominate. Example: 1 claim -> max_weight=0.25, 5 claims -> max_weight=0.11, 20 claims -> max_weight=0.06
- **Mean weight gets the remainder:** Always the largest component, providing stable average risk

**Fallback** (no per-sentence probe data):
```
combined_risk = (energy_risk + entropy_risk) / 2.0
```

### Final Confidence Level (SLT)

```
< 0.35  ->  "high"
< 0.65  ->  "medium"
>= 0.65 ->  "low"
```

### Sentence Average Confidence

**Source:** `engine.py:657-660`

```
valid_confs = [s.confidence for s in claim_sentences where confidence is not None]
sentence_avg_confidence = mean(valid_confs)
```

A simple average of B1 logit-confidence scores across claim sentences. Reported alongside the probe-based scores for comparison.

---

## Constants and Thresholds Reference

### Sigmoid Calibration (Llama 3.1 8B 4-bit)

| Constant | Value | Purpose |
|---|---|---|
| `LOGIT_SIGMOID_CENTER` | 33.0 | Logit value mapping to 50% confidence |
| `LOGIT_SIGMOID_SCALE` | 3.0 | Sigmoid steepness |

### Margin Boost

| Constant | Value | Purpose |
|---|---|---|
| Margin threshold | 3.0 | Minimum top1-top2 gap to trigger boost |
| Boost rate | 0.02 per unit | Linear scaling above threshold |
| Max boost | 0.1 | Hard cap on boost amount |

### AUROC Probe Weights

| Probe | Validation AUROC | Weight |
|---|---|---|
| Entropy | 0.773 | 0.515 |
| Energy | 0.727 | 0.485 |

### Token Length Threshold

| Constant | Value | Purpose |
|---|---|---|
| `TOKEN_THRESHOLD` | 100 | Boundary between short/long aggregation strategies |

### Confidence Level Thresholds

| Context | High | Medium | Low |
|---|---|---|---|
| `/chat` (energy score) | > 0.80 | > 0.50 | <= 0.50 |
| B1 sentence-level | >= 0.60 | >= 0.30 | < 0.30 |
| Probe risk (TBG & SLT) | < 0.35 | < 0.65 | >= 0.65 |

### Long Answer Aggregation Weights

| Component | Weight | Notes |
|---|---|---|
| SLT (entropy only) | 0.15 | Fixed, minimal anchor |
| Max sentence risk | 0.25 / (1 + ln(n)) | Decreases logarithmically with claim count |
| Mean sentence risk | 1.0 - slt - max | Gets the remainder, always the largest |

---

## Calculations Quick-Reference Table

| # | Calculation | Location | Formula | Input | Output | Range |
|---|---|---|---|---|---|---|
| 1 | Token probability product | `engine.py:30` | `prod(token_probs)` | Per-token probs | Per-sample prob | [0,1] |
| 2 | Boltzmann logit mean | `engine.py:39` | `-mean(token_logits)` | Per-token logits | Per-sample score | (-inf,0) |
| 3 | Cluster probability sum | `engine.py:24` | `sum(norm_probs[i] for i in cluster)` | Normalized probs | Per-cluster prob | [0,1] |
| 4 | Cluster logit sum | `engine.py:26` | `-sum(logits[i] for i in cluster)` | Per-sample logits | Per-cluster logit | [0,inf) |
| 5 | Energy normalization | `engine.py:15` | `x / sum(all_x)` | Cluster logits | Normalized energies | [0,1], sum=1 |
| 6 | Main cluster confidence | `app.py:205` | `cluster_energies[main_idx]` | Normalized energies | Confidence score | [0,1] |
| 7 | Sigmoid confidence | `engine.py:270` | `1/(1+exp(-(x-33)/3))` | Mean sentence logit | Confidence | [0,1] |
| 8 | Margin boost | `engine.py:274` | `min(0.1, (m-3)*0.02)` | Mean margin | Boost amount | [0,0.1] |
| 9 | Energy risk (probe) | `engine.py:437,527,566` | `1 - predict_proba[0,1]` | Hidden states | Risk score | [0,1] |
| 10 | Entropy risk (probe) | `engine.py:443,532,574` | `predict_proba[0,1]` | Hidden states | Risk score | [0,1] |
| 11 | AUROC-weighted risk | `engine.py:584` | `0.515*e + 0.485*h` | Both probe risks | Combined risk | [0,1] |
| 12 | TBG combined risk | `engine.py:445` | `(e + h) / 2` | TBG probe risks | Combined risk | [0,1] |
| 13 | Short answer blend | `engine.py:616` | `0.5*slt + 0.5*mean(sent)` | SLT + sentence risks | Combined risk | [0,1] |
| 14 | Long answer adaptive | `engine.py:639-641` | `0.15*e + (0.25/ln(n))*max + rest*mean` | SLT + sentence risks | Combined risk | [0,1] |
| 15 | Sentence avg confidence | `engine.py:658-660` | `mean(claim_confidences)` | B1 confidences | Average | [0,1] |

---

## Data Flow Diagram

```
User Prompt
    |
    v
[Generate N=5 Responses] ──────────────────────────── /chat path
    |                                                     |
    ├── Per-sample: token_ids, logits, probs, top2_logits |
    |                                                     |
    ├──> [Score Sentences (B1)]                           |
    |    ├── Sigmoid(mean_logit) per sentence              |
    |    ├── Margin boost if gap > 3.0                     |
    |    └── Claim filtering                               |
    |                                                     |
    ├──> [Semantic Clustering]                             |
    |    ├── LLM pairwise equivalence checks               |
    |    └── Greedy clustering                             |
    |                                                     |
    └──> [cal_flow -> sum_normalize]                       |
         ├── cal_probs: product per sample                 |
         ├── cal_boltzmann_logits: neg-mean per sample     |
         ├── cal_cluster_ce: aggregate to clusters         |
         └── energy_score_raw = main cluster energy ───────┘


User Prompt
    |
    v
[Forward Pass Only] ──────────────────────────── /score_fast_tbg path
    |
    └──> tbg_hidden at last prompt token
         ├── Energy probe -> energy_risk = 1 - P(confident)
         ├── Entropy probe -> entropy_risk = P(uncertain)
         └── combined = (e + h) / 2


User Prompt
    |
    v
[Generate 1 Response] ─────────────────────────── /score_fast_slt path
    |
    ├── B1 sentence scoring
    |
    ├── [Forward Pass on prompt+answer]
    |   ├── slt_hidden at second-to-last token
    |   └── sent_hiddens at each sentence-end token
    |
    ├── Overall SLT probes: energy_risk, entropy_risk
    |
    ├── Per-sentence dual probes:
    |   └── probe_risk = 0.515 * entropy + 0.485 * energy
    |
    └── Token-length conditional aggregation:
        ├── Short (<=100): 0.5 * slt + 0.5 * mean(sent)
        └── Long (>100):   0.15*slt + adaptive*max + rest*mean
```
