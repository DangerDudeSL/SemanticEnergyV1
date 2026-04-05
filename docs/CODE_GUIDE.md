# Code Guide — SemanticEnergy Developer Walkthrough

A comprehensive reference for understanding, modifying, and extending the SemanticEnergy codebase.

---

## 1. Project Overview

SemanticEnergy is a hallucination detection system for large language models. It calculates confidence scores in real-time by analysing how certain the model is about its own output, using three complementary methods:

| Mode | What it does | Speed | Key idea |
|---|---|---|---|
| **Full SE** | Generate 5 answers, cluster by meaning, compute energy flow | ~60–120s | If the model gives the same answer 5 times, it's probably right |
| **Fast SLT** | Generate 1 answer, read the model's internal state after writing | ~5–15s | The model's hidden state reveals if it was "guessing" |
| **Fast TBG** | Read the model's internal state before writing anything | ~0.5–2s | The model already "knows" if it can answer the question |

All three modes also run a **B1 sentence baseline** — per-sentence logit confidence scoring that highlights which specific sentences are least confident.

**Stack:** Python 3.12 + FastAPI backend + vanilla HTML/CSS/JS frontend. No build step.

---

## 2. Architecture

```
                        +---------------------------+
                        |        Frontend           |
                        |  index.html / script.js   |
                        |  styles.css               |
                        +-------------|-------------+
                                      | HTTP (JSON)
                                      v
                        +---------------------------+
                        |    FastAPI Server          |
                        |    backend/app.py          |
                        |                           |
                        |  /status          (GET)   |
                        |  /switch_model    (POST)  |
                        |  /chat            (POST)  |  <-- Full SE
                        |  /score_fast_tbg  (POST)  |  <-- TBG probe
                        |  /score_fast_slt  (POST)  |  <-- SLT probe
                        +-------------|-------------+
                                      |
                        +-------------|-------------+
                        |   SemanticEngine          |
                        |   backend/engine.py       |
                        |                           |
                        |   - Model loading (CUDA)  |
                        |   - Response generation   |
                        |   - Semantic clustering   |
                        |   - Hidden state probes   |
                        |   - Sentence scoring      |
                        +-------------|-------------+
                                      |
                  +-------------------|-------------------+
                  |                   |                   |
        +---------+------+  +--------+-------+  +-------+--------+
        | ClaimFilter    |  | Probe Bundles  |  | HF Transformer |
        | claim_filter.py|  | models/*.pkl   |  | (Llama/Qwen)   |
        +----------------+  +----------------+  +----------------+
```

---

## 3. Backend Reference

### 3.1 engine.py — Helper Functions (Lines 14–48)

These are pure math functions used by the Full SE pipeline. They operate on the logits and probabilities collected during multi-sample generation.

#### `sum_normalize(lst)` — Line 15
Normalizes a list so values sum to 1. Used to convert raw cluster energies into a probability distribution.
```
Input:  [3.0, 1.0, 1.0]
Output: [0.6, 0.2, 0.2]
```

#### `cal_cluster_ce(probs, logits, clusters)` — Line 19
Aggregates per-response probabilities and logits into per-cluster values. For each cluster, sums the normalized probabilities of member responses, and sums the negated logits.
- **Input:** `probs` (list of floats per response), `logits` (list of floats per response), `clusters` (list of index lists)
- **Output:** `(probs_se, logits_se)` — one value per cluster

#### `cal_probs(probs_list)` — Line 30
Computes the product of per-token probabilities for each response. This gives the joint probability of the entire generated sequence.
- **Input:** `[[p1, p2, ...], [p1, p2, ...], ...]` — per-token probs for each of N responses
- **Output:** `[prod_1, prod_2, ...]` — one float per response

#### `fermi_dirac(E, mu=0.0, kT=1.0)` — Line 33
The Fermi-Dirac distribution function: `E / (exp((E - mu) / kT) + 1)`. Used as a bounded energy function that prevents extreme logit values from dominating.

#### `cal_fermi_dirac_logits(logits_list, mu=0.0)` — Line 36
Applies `fermi_dirac()` to each token logit in each response, then takes the negated mean per response.

#### `cal_boltzmann_logits(logits_list)` — Line 39
Simpler alternative: just negates the mean of per-token logits for each response. This is the default used when `fermi_mu=None`.

#### `cal_flow(probs_list, logits_list, clusters, fermi_mu=None)` — Line 42
**Main SE calculation.** Chains the above functions:
1. `cal_probs(probs_list)` — joint probability per response
2. `cal_boltzmann_logits(logits_list)` or `cal_fermi_dirac_logits(logits_list, mu)` — energy per response
3. `cal_cluster_ce(probs, logits, clusters)` — aggregate into clusters
- **Output:** `(probs_se, logits_se)` — per-cluster probability mass and energy

---

### 3.2 engine.py — SemanticEngine Class (Lines 51–676)

#### `__init__(model_id, use_8bit=True)` — Line 52
Loads a quantized LLM onto CUDA. Steps:
1. Verify CUDA is available (raises `RuntimeError` if not)
2. Configure `BitsAndBytesConfig` for 8-bit (or 4-bit) quantization
3. Load tokenizer via `AutoTokenizer.from_pretrained(model_id)`; set `pad_token = eos_token` if missing
4. Load model via `AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="cuda:0")`
5. Verify VRAM usage > 0.1 GB (catches failed quantization)
6. Initialize pysbd sentence segmenter and ClaimFilter

**Sets:** `self.model`, `self.tokenizer`, `self._sentence_segmenter`, `self._claim_filter`

#### `_safe_apply_chat_template(tokenizer, messages, **kwargs)` — Line 94 (static)
Wrapper for `tokenizer.apply_chat_template()` that handles Qwen3's `enable_thinking=False` parameter. Falls back gracefully for models that don't support it.

#### `generate_responses(question, num_samples=5)` — Line 102
Generates N diverse answers with full token-level data. For each sample:
1. Apply chat template to `[system_prompt, user_question]`
2. Run `model.generate()` with `do_sample=True, temperature=0.7, max_new_tokens=512`
3. For each generated token, record:
   - `logits[token_id]` — the chosen token's logit value
   - `F.softmax(logits)[token_id]` — the chosen token's probability
   - `top2_logits` — the two highest raw logit values (from `outputs.logits`, not the filtered `outputs.scores`)

**Returns:** List of dicts, each with keys: `answer`, `logits`, `probs`, `token_ids`, `top2_logits`

**Key detail:** `outputs.logits[step]` gives unfiltered logits; `outputs.scores[step]` is post-filtered by top-k/top-p sampling and may contain `-inf`.

#### `split_into_sentences(text)` — Line 176
Splits text into sentences using the pysbd library. Returns a list of non-empty stripped strings.

#### `align_tokens_to_sentences(answer_text, token_ids, sentences)` — Line 181
Maps each token to its parent sentence. Algorithm:
1. Re-tokenize `answer_text` with `return_offsets_mapping=True` to get character ranges per token
2. Build sentence character spans by sequential `str.find()` search
3. For each token, compute its character midpoint, find which sentence span contains it
- **Output:** `list[int]` — sentence index for each token (0-based)

#### `score_sentences(answer_text, token_ids, logits, top2_logits)` — Line 215
**B1 sentence-level logit confidence baseline.** No additional model calls — purely post-processing of existing per-token data.

For each sentence:
1. Group token logits and margins by sentence (using `align_tokens_to_sentences`)
2. Compute mean logit, mean margin (top1 - top2)
3. **Sigmoid mapping:** `confidence = 1 / (1 + exp(-(mean_logit - 33.0) / 3.0))`
   - Center=33.0 maps to 50% confidence (calibrated for Llama 3.1 8B 8-bit)
   - Scale=3.0 controls steepness
4. **Margin boost:** if mean margin > 3.0, add up to 0.1 confidence
5. Check `ClaimFilter.is_claim(sentence)` — non-claims get `level="none"`, no scoring

**Level thresholds:** >= 0.6 = "high", >= 0.3 = "medium", < 0.3 = "low"

**Returns:** List of dicts per sentence: `text`, `confidence`, `level`, `num_tokens`, `mean_chosen_logit`, `mean_logit_margin`, `is_claim`

#### `semantic_analyse(question, answer_a, answer_b)` — Line 301
Uses the LLM as a semantic equivalence judge. Prompts the model with both answers and asks for "Final Decision: Yes" or "Final Decision: No". Uses `do_sample=False` for deterministic output.
- **Returns:** `True` if "Yes" in response

#### `find_semantic_clusters(question, answer_list)` — Line 334
Groups N answers into semantically equivalent clusters using greedy single-linkage:
1. For each unvisited answer i, start a new cluster [i]
2. Compare i against all remaining unvisited j via `semantic_analyse()`
3. If equivalent, add j to i's cluster
4. **Complexity:** O(n^2) pairwise comparisons, each requiring a full LLM forward pass. For 5 samples, up to 10 comparisons.
- **Returns:** `[[0, 2, 4], [1], [3]]` — list of clusters (each a list of response indices)

#### `_extract_hidden_states(question, answer_text, extra_positions=None)` — Line 351
Runs a **separate forward pass** on `prompt + answer` to extract hidden states at key positions. This is separate from generation because storing hidden states during `model.generate()` would use ~1.3 GB per call.

**Important:** Uses user-only template (no system prompt) to match probe training data.

1. Build full text = `chat_template(user_question) + answer_text`
2. Run `model(**full_inputs, output_hidden_states=True)`
3. Stack hidden states: shape `(num_layers, seq_len, hidden_dim)` — e.g., (33, 150, 4096)
4. Extract:
   - **TBG hidden:** at `prompt_len - 1` (last prompt token)
   - **SLT hidden:** at `full_len - 2` (second-to-last answer token)
   - **Extra hiddens:** at caller-specified answer-relative positions (for per-sentence probes)
- **Returns:** `(tbg_hidden, slt_hidden, extra_hiddens)` as numpy arrays of shape `(num_layers, hidden_dim)`

#### `score_with_tbg_probe(question, probe_bundle)` — Line 403
**TBG mode: pre-generation risk estimation.** Single forward pass on prompt only, no generation.

1. Forward pass on prompt with `output_hidden_states=True`
2. Extract TBG hidden state at `prompt_len - 1`
3. Slice to best layer range: `X = tbg_hidden[l0:l1, :].reshape(1, -1)` (e.g., layers 28–32 = 4 layers x 4096 dims = 16384 features)
4. Scale with `probe_bundle["tbg_energy_scaler"].transform(X)`
5. Predict: `energy_confidence = probe.predict_proba(X)[0, 1]`
6. `energy_risk = 1.0 - energy_confidence` (invert: probe predicts correctness)
7. Repeat for entropy probe (different layer range)
8. `combined_risk = (energy_risk + entropy_risk) / 2.0`
9. Map: < 0.35 = "high", < 0.65 = "medium", >= 0.65 = "low"

**Returns:** `{mode, energy_risk, entropy_risk, combined_risk, confidence_level}`

#### `score_with_slt_probe(question, probe_bundle)` — Line 461
**SLT mode: post-generation confidence.** The most complex function. Generates one answer, then runs per-sentence and overall probe scoring.

**Steps:**
1. Generate 1 response via `generate_responses(question, num_samples=1)`
2. Run B1 sentence scoring (`score_sentences`)
3. Find last-token positions for each claim sentence
4. Run `_extract_hidden_states()` with sentence-end positions
5. **Overall SLT scoring:** energy probe + entropy probe at second-to-last token
6. **Per-sentence dual-probe scoring** (claim sentences only):
   - Energy weight = 0.70, Entropy weight = 0.30
   - `probe_risk = 0.70 * energy_risk + 0.30 * entropy_risk`
   - Override sentence level from probe risk
7. **Length-adaptive aggregation:**
   - Short answers (<=100 tokens): blend 50% SLT overall + 50% mean per-sentence risks
   - Long answers (>100 tokens): `slt_weight=0.15`, `max_weight = 0.25 / (1 + log(n))`, `mean_weight = remainder`
   - This ensures long multi-sentence responses are scored by their riskiest claims, not just the SLT signal at the end

**Returns:** `{mode, answer, energy_risk, entropy_risk, combined_risk, confidence_level, sentence_scores, sentence_avg_confidence}`

---

### 3.3 app.py — FastAPI Server (297 lines)

#### Global State (Lines 22–28)
- `engine` — SemanticEngine instance (None until startup completes)
- `current_model_id` — active model string
- `probe_bundle` — dict of trained probes/scalers (from pickle)
- `loading_model_id` — set during model loading for status polling
- `_model_lock` — threading.Lock serializing GPU access

#### `load_probe_bundle(model_id)` — Line 50
Loads a probe pickle from `backend/models/`. Applies `_patch_sklearn_compat()` to fix sklearn version gaps (some versions removed `multi_class` from LogisticRegression but still reference it in `predict_proba`).

#### `startup_event()` — Line 65
Runs on FastAPI startup. Loads default model (Llama 3.1 8B) and its probe bundle.

#### Endpoints

| Endpoint | Type | Purpose |
|---|---|---|
| `GET /status` | async | Readiness check — returns `{ready, model_id}` |
| `POST /switch_model` | sync | Hot-swap the loaded model, release old VRAM, load new |
| `POST /chat` | sync | Full SE pipeline — 5 generations + clustering + energy |
| `POST /score_fast_tbg` | sync | TBG probe — prompt-only forward pass |
| `POST /score_fast_slt` | sync | SLT probe — 1 generation + forward pass |

The heavy endpoints are `def` (not `async def`) so FastAPI runs them in a thread pool, preventing the event loop from blocking. The `_model_lock` ensures only one GPU operation runs at a time.

`/chat` also caps `num_samples` to max 10 server-side.

---

### 3.4 claim_filter.py — ClaimFilter (74 lines)

Regex-based lightweight filter that identifies whether a sentence contains a factual claim worth scoring, or is filler/transitional text.

#### Pattern Categories

| Category | Examples | Effect |
|---|---|---|
| Filler phrases | "sure", "thank you", "of course" | Exact match → skip |
| Meta/transitional | "Here are...", "Let me explain..." | Regex → skip |
| Questions | "What is...?", "Why does...?" | Regex → skip |
| Hedging/opinion | "I think...", "It seems..." | Regex → skip |
| Advisory | "Please note...", "Disclaimer" | Regex → skip |
| Greetings | "Hi", "Hope this helps" | Regex → skip |
| Markdown headers | `**Something:**` | Regex → skip |

#### `is_claim(sentence)` — Line 44
Returns `True` if the sentence likely contains a factual claim. Algorithm:
1. Empty → False
2. Less than 3 words and no digits → False
3. Exact match in filler phrases → False
4. Matches any NON_CLAIM_PATTERN regex → False
5. Markdown header or numbered prefix → False
6. **Default: True** (conservative — score unless filtered)

---

## 4. Frontend Reference

### 4.1 script.js — Key Functions (853 lines)

#### API Layer (Lines 1–18)
- `NGROK_DOMAIN` — static ngrok URL for cloud deployment
- `getBaseUrl()` — returns `localhost:8000` for local, `NGROK_DOMAIN` for cloud
- `apiFetch(url, options)` — fetch wrapper that adds `ngrok-skip-browser-warning` header

#### State (Lines 29–31)
- `selectedMode` — `"full"` | `"slt"` | `"tbg"`
- `selectedModelId` — e.g., `"meta-llama/Llama-3.1-8B-Instruct"`

#### Score History (Lines 33–134)
- `scoreHistory` — array of `{q, risk, conf, mode, ts}` entries, persisted in `sessionStorage`
- `renderScoreChart()` — draws an SVG line chart with color-coded confidence zones (green/yellow/red)

#### Model Readiness Polling (Lines 139–178)
- `pollBackendStatus()` — polls `GET /status` every 2 seconds until `ready: true`
- Shows loading overlay during model initialization and model switching

#### Preferences (Lines 180–200)
- `savePrefs()` / `loadPrefs()` — persist mode and model selection to `localStorage`

#### Message Persistence (Lines 202–232)
- `persistMessages()` — saves all chat message HTML to `sessionStorage`
- `restoreMessages()` — restores on page reload

#### Message Rendering (Lines 353–690)
- `addMessage(text, isUser)` — user messages use `textContent` (XSS-safe), AI messages HTML-escape then apply bold markdown
- `addMessageWithSentenceScores(text, sentenceScores)` — renders answer with per-sentence confidence highlighting (color-coded spans, tooltips)
- `appendConfidenceBadge(messageEl, {score, level, clusters, mode})` — adds the colored confidence badge
- `appendMetricsPanel(messageEl, metricsData)` — collapsible details panel with energy/entropy/cluster metrics
- `appendSentenceAnalysis(messageEl, sentenceScores)` — collapsible table showing per-sentence scores

#### `sendMessage()` — Line 692
Main dispatch function. Branches on `selectedMode`:

- **Full SE:** POST `/chat` with `{prompt, num_samples: 5, model_id}` → render answer + badge + metrics + sentence analysis
- **Fast SLT:** POST `/score_fast_slt` with `{prompt}` → render answer with probe-based sentence highlights
- **Fast TBG:** Two-phase — POST `/score_fast_tbg` (Phase 1: pre-generation risk) then POST `/score_fast_slt` (Phase 2: answer + sentence analysis). Badge shows TBG risk; sentences show SLT probe scores.

### 4.2 styles.css — Design Tokens (1091 lines)

Key CSS variables (`:root`):

| Variable | Value | Purpose |
|---|---|---|
| `--bg-color` | `#f5f5f7` | Page background |
| `--msg-user-bg` | `#0A84FF` | User message bubble (iOS blue) |
| `--conf-high` | `rgba(48,209,88,0.12)` | High confidence background (green) |
| `--conf-med` | `rgba(255,159,10,0.12)` | Medium confidence background (orange) |
| `--conf-low` | `rgba(255,69,58,0.12)` | Low confidence background (red) |

Sentence highlighting classes: `.sentence-medium` (orange tint + underline), `.sentence-low` (red tint + underline), `.sentence-non-claim` (dimmed opacity 0.55).

---

## 5. Data Flow Diagrams

### 5.1 Full SE Mode

```
User types question, clicks Send (mode = "full")
  |
  v
script.js sendMessage() [line 715]
  |  POST /chat  {prompt, num_samples: 5, model_id}
  v
app.py chat_endpoint() [line 134]
  |
  |-- engine.generate_responses(question, 5) [engine.py:102]
  |     |-- 5x model.generate(do_sample=True, temp=0.7)
  |     |-- Per token: record logit, prob, top2_logits
  |     +-> Returns [{answer, logits, probs, token_ids, top2_logits}, ...]
  |
  |-- engine.score_sentences(main_answer, ...) [engine.py:215]
  |     |-- pysbd split -> token alignment -> sigmoid confidence per sentence
  |     +-> Returns [{text, confidence, level, is_claim}, ...]
  |
  |-- engine.find_semantic_clusters(question, answers) [engine.py:334]
  |     |-- Pairwise semantic_analyse() via LLM  (up to 10 calls)
  |     +-> Returns [[0, 2, 4], [1], [3]]
  |
  |-- cal_flow(probs_list, logits_list, clusters) [engine.py:42]
  |     |-- cal_probs -> cal_boltzmann_logits -> cal_cluster_ce
  |     +-> Returns (probs_se, logits_se)
  |
  |-- sum_normalize(logits_se) -> cluster_energies
  |-- main_confidence = cluster_energies[main_cluster_index]
  |
  +-> Response: {answer, confidence_score, confidence_level, clusters_found,
                  sentence_scores, debug_data: {all_answers, energies_per_cluster}}
  |
  v
script.js renders:
  |-- addMessageWithSentenceScores()  (highlighted answer text)
  |-- appendConfidenceBadge()         (e.g. "High Confidence (87.0%) - 1 Semantic Cluster")
  |-- appendTimer()                   (e.g. "Generated in 64.2s")
  |-- appendMetricsPanel()            (clusters, per-cluster energies)
  +-- appendSentenceAnalysis()        (per-sentence confidence table)
```

### 5.2 Fast SLT Mode

```
User types question, clicks Send (mode = "slt")
  |
  v
script.js sendMessage() [line 753]
  |  POST /score_fast_slt  {prompt}
  v
app.py score_fast_slt() [line 247]
  |
  +-- engine.score_with_slt_probe(question, probe_bundle) [engine.py:461]
        |
        |-- generate_responses(question, num_samples=1) [engine.py:102]
        |     +-> 1 answer with logits/probs/top2_logits
        |
        |-- score_sentences(answer, ...) [engine.py:215]
        |     +-> B1 logit confidence per sentence
        |
        |-- align_tokens_to_sentences -> find last token per sentence
        |
        |-- _extract_hidden_states(question, answer, sent_end_positions) [engine.py:351]
        |     |-- Single forward pass with output_hidden_states=True
        |     |-- Extract TBG hidden @ prompt_len - 1
        |     |-- Extract SLT hidden @ full_len - 2
        |     +-- Extract sentence-end hiddens at each claim's last token
        |
        |-- Overall SLT probe scoring:
        |     |-- energy_risk = 1.0 - energy_probe.predict_proba(slt_hidden[l0:l1])
        |     +-- entropy_risk = entropy_probe.predict_proba(slt_hidden[l0h:l1h])
        |
        |-- Per-sentence probe scoring (claims only):
        |     |-- For each claim sentence's last-token hidden state:
        |     |     energy_risk_i = 1.0 - energy_probe.predict_proba(h[l0:l1])
        |     |     entropy_risk_i = entropy_probe.predict_proba(h[l0h:l1h])
        |     |     probe_risk_i = 0.70 * energy + 0.30 * entropy
        |     +-- Override sentence level from probe_risk
        |
        |-- Aggregate (token-length adaptive):
        |     |-- Short (<=100 tok): 0.5 * slt_combined + 0.5 * mean(sent_risks)
        |     +-- Long  (>100 tok):  weighted blend of slt + max_sent + mean_sent
        |
        +-> Returns {answer, energy_risk, entropy_risk, combined_risk,
                      confidence_level, sentence_scores, sentence_avg_confidence}
```

### 5.3 Fast TBG Mode (Two-Phase)

```
User types question, clicks Send (mode = "tbg")
  |
  v
script.js sendMessage() [line 785]
  |
  |  Phase 1: POST /score_fast_tbg  {prompt}
  v
app.py score_fast_tbg() [line 228]
  +-- engine.score_with_tbg_probe(question, probe_bundle) [engine.py:403]
        |-- Forward pass on prompt only (no generation)
        |-- Extract hidden state at last prompt token
        |-- Energy probe + entropy probe -> combined_risk
        +-> Returns {energy_risk, entropy_risk, combined_risk, confidence_level}
  |
  |  Phase 2: POST /score_fast_slt  {prompt}
  v
  [Same as SLT flow above — generates answer + sentence scoring]
  |
  v
script.js renders:
  |-- Badge shows TBG risk (pre-generation estimate)
  |-- Metrics panel shows TBG probe risks
  |-- Answer text + sentence analysis uses SLT probe scores
  +-- Score history records TBG combined_risk
```

---

## 6. Key Algorithms and Formulas

### 6.1 B1 Sigmoid Logit Confidence (engine.py:248–270)

Per-sentence confidence from raw token logits:

```
mean_logit = mean([logit for token in sentence])
confidence = 1 / (1 + exp(-(mean_logit - 33.0) / 3.0))
```

- **Center = 33.0:** logit value mapping to 50% confidence (calibrated for Llama 3.1 8B 8-bit)
- **Scale = 3.0:** steepness — a 10-point logit range spans ~32% confidence change

Margin boost (when the model is very certain about each token choice):
```
if mean_margin > 3.0:
    margin_boost = min(0.1, (mean_margin - 3.0) * 0.02)
    confidence = min(1.0, confidence + margin_boost)
```

### 6.2 Boltzmann Energy (engine.py:39–40)

Per-response energy (simplified from Fermi-Dirac):
```
energy_i = -mean(logits_per_token for response_i)
```
Higher logits (more confident tokens) → lower energy → more confident answer.

### 6.3 Cluster Energy Flow (engine.py:19–28)

After clustering N responses into semantic groups:
```
cluster_prob_i   = sum(normalized_response_probs in cluster_i)
cluster_energy_i = -sum(response_energies in cluster_i)
SE_confidence    = sum_normalize(cluster_energies)[main_cluster_index]
```

### 6.4 Probe Risk Aggregation (engine.py:537–653)

**Per-sentence combined risk:**
```
W_ENERGY  = 0.70     # energy probe weighted higher — more reliable
W_ENTROPY = 0.30
probe_risk = W_ENERGY * energy_risk + W_ENTROPY * entropy_risk
```

**Overall answer risk (short, <=100 tokens):**
```
slt_combined = 0.70 * energy_risk + 0.30 * entropy_risk

if num_claim_sentences >= 2:
    combined_risk = 0.5 * slt_combined + 0.5 * mean(per_sentence_risks)
elif num_claim_sentences == 1:
    combined_risk = 0.5 * slt_combined + 0.5 * sentence_risk
else:
    combined_risk = slt_combined
```

**Overall answer risk (long, >100 tokens):**
```
slt_weight  = 0.15
max_weight  = 0.25 / (1 + log(num_claims))
mean_weight = 1.0 - slt_weight - max_weight

combined_risk = slt_weight  * slt_combined
              + max_weight  * max(per_sentence_risks)
              + mean_weight * mean(per_sentence_risks)
```

Rationale: For long answers, the SLT token only captures the state at the end of the answer, so we reduce its weight and rely more heavily on per-sentence probe scores. The `max` term ensures a single risky claim can flag the whole answer.

### 6.5 Level Thresholds

| combined_risk | Level | Badge | UI Color |
|---|---|---|---|
| < 0.35 | high | OK | Green |
| 0.35–0.65 | medium | WARN | Yellow |
| >= 0.65 | low | RISK | Red |

Full SE mode uses different thresholds: > 0.80 = "high", > 0.50 = "medium".

See [scoring_formulas.md](scoring_formulas.md) for the complete reference.

---

## 7. Probe Creation Pipeline

The probes are logistic regression classifiers trained on hidden states extracted from a dataset of LLM-answered questions. The pipeline runs in two notebooks.

### 7.1 Dataset Generation (notebooks/01_generate_dataset.ipynb)

For each of 500–1000 TriviaQA questions, the notebook:

1. **Generates 5 diverse responses** via `engine.generate_responses(question, num_samples=5)` with `temperature=0.7`
2. **Clusters responses** via `engine.find_semantic_clusters()` — pairwise LLM semantic equivalence checks
3. **Computes energy teacher score:** the normalized cluster energy of the main answer's cluster via `cal_flow → sum_normalize`. High energy = model consistently gave this answer = confident.
4. **Computes entropy teacher score:** Shannon entropy over cluster size distribution. `H = -sum(p_i * log(p_i))` where `p_i = cluster_size_i / total_samples`. 1 cluster = 0 entropy (certain). 5 clusters of size 1 = max entropy (uncertain).
5. **Computes correctness:** normalized substring match against TriviaQA reference aliases. Used for evaluation only — probes are NOT trained on correctness labels.
6. **Extracts hidden states** via a separate forward pass on `prompt + answer`:
   - TBG hidden: `(33 layers, 4096 dims)` at last prompt token
   - SLT hidden: `(33 layers, 4096 dims)` at second-to-last answer token
7. **Extracts logit features:** mean/min/std of chosen-token logits, mean/min/std of top-2 logit margins, answer length

Each record is saved to `backend/data/probe_dataset_{model}_{dataset}.pkl` (~540 MB for 1000 questions).

**Key helper functions defined in the notebook:**
- `is_correct_triviaqa(predicted, references)` — normalized substring match
- `cluster_assignment_entropy(clusters)` — Shannon entropy
- `extract_hidden_states(engine, prompt, answer)` — separate forward pass (same logic as `engine._extract_hidden_states`)
- `extract_logit_feats(generated_data_0)` — logit statistics (mean/min/std of chosen logit, margin, answer length)
- `generate_record(engine, example, num_samples)` — full pipeline for one question

**Time:** 4–6 hours on GPU (each question requires 5 generations + up to 10 pairwise comparisons + 1 hidden state extraction).

### 7.2 Probe Training (notebooks/02_train_se_probes.ipynb)

The training notebook is fully self-contained and produces the `.pkl` bundle used at inference time.

**Step 1 — Load and split dataset:**
- 80% train / 10% validation / 10% test split
- Loads from the pkl generated by notebook 01

**Step 2 — Binarize teacher scores:**
- Uses SEP-style (Semantic Entropy Probes paper) within-group MSE minimization to find optimal thresholds
- Energy: scores above threshold → label 1 (correct/confident), below → label 0
- Entropy: scores above threshold → label 1 (uncertain/hallucinating), below → label 0
- **No correctness labels** are used for binarization — the probes learn to predict the model's own energy/entropy signals, not ground truth correctness

**Step 3 — Layer sweep:**
- For each 4-layer window across all 33 transformer layers (e.g., layers 0–4, 1–5, ..., 29–33):
  - Flatten hidden states: `X = hidden[l0:l1, :].reshape(1, -1)` → shape `(n_samples, 4 * 4096)`
  - Fit StandardScaler + LogisticRegression(max_iter=2000)
  - Evaluate AUROC on validation set
- Select the best layer range per probe type (TBG energy, TBG entropy, SLT energy, SLT entropy)

**Step 4 — Train final probes:**
- Retrain on best layers with full training set
- Bootstrap AUROC evaluation on test set (100 bootstrap iterations for confidence intervals)
- Feature ablation to verify hidden states outperform logit features alone

**Step 5 — Save probe bundle:**
The pickle bundle contains:
```python
{
    "tbg_energy_probe":    LogisticRegression,
    "tbg_energy_scaler":   StandardScaler,
    "best_energy_tbg_range": (28, 32),     # layer indices
    "tbg_entropy_probe":   LogisticRegression,
    "tbg_entropy_scaler":  StandardScaler,
    "best_entropy_tbg_range": (21, 25),
    "slt_energy_probe":    LogisticRegression,
    "slt_energy_scaler":   StandardScaler,
    "best_energy_slt_range": (17, 21),
    "slt_entropy_probe":   LogisticRegression,
    "slt_entropy_scaler":  StandardScaler,
    "best_entropy_slt_range": (20, 24),
}
```

Saved to `backend/models/probes_{model}_{dataset}.pkl` (~2 MB).

### 7.3 Why This Approach Works

The key insight is that a transformer's hidden states at specific token positions encode information about how confident the model is in its answer — even before it starts generating. The probes learn to decode this signal:

- **TBG probes** read the hidden state at the last prompt token. At this point, the model has already processed the question and computed attention across all layers. The probe asks: "Given the model's internal representation of this question, does the model 'know' the answer?"
- **SLT probes** read the hidden state at the second-to-last generated token. By this point, the model has committed to an answer. The probe asks: "Given the model's internal state after generating most of the answer, does this look like a state that typically produces correct answers?"

The teacher signals (energy and entropy) come from the expensive multi-sample Full SE pipeline. The probes distill this into a single forward pass.

---

## 8. File Quick Reference

| Component | File | Key Lines |
|---|---|---|
| Helper math functions | engine.py | 14–48 |
| SemanticEngine.__init__ | engine.py | 52–92 |
| generate_responses | engine.py | 102–172 |
| split_into_sentences | engine.py | 176–179 |
| align_tokens_to_sentences | engine.py | 181–213 |
| score_sentences (B1) | engine.py | 215–299 |
| semantic_analyse | engine.py | 301–332 |
| find_semantic_clusters | engine.py | 334–349 |
| _extract_hidden_states | engine.py | 351–401 |
| score_with_tbg_probe | engine.py | 403–459 |
| score_with_slt_probe | engine.py | 461–676 |
| ClaimFilter.is_claim | claim_filter.py | 44–74 |
| FastAPI setup + CORS | app.py | 1–19 |
| Global state + lock | app.py | 22–29 |
| Probe loading | app.py | 50–57 |
| Startup | app.py | 65–77 |
| GET /status | app.py | 83–87 |
| POST /switch_model | app.py | 92–131 |
| POST /chat | app.py | 134–236 |
| POST /score_fast_tbg | app.py | 228–249 |
| POST /score_fast_slt | app.py | 247–278 |
| API config + getBaseUrl | script.js | 1–18 |
| Score history + chart | script.js | 33–134 |
| Model polling | script.js | 139–178 |
| Message rendering | script.js | 353–690 |
| sendMessage (main) | script.js | 692–849 |
| CSS design tokens | styles.css | 1–21 |
| Sentence highlighting | styles.css | 578–614 |

---

## 9. Common Debugging Patterns

### Slow inference
- Check VRAM: `nvidia-smi`. If near capacity, quantization may be swapping to RAM.
- Check `num_samples` — Full SE with 5 samples triggers ~15 LLM forward passes (5 generations + up to 10 pairwise comparisons).
- The SLT/TBG modes should be 5–10x faster than Full SE.

### Poor scoring / everything shows "medium"
- Verify probe bundle matches loaded model: `PROBE_BUNDLES` in app.py (line 33) maps model IDs to `.pkl` files.
- Probes trained on TriviaQA short answers. Long-form outputs hit the `TOKEN_THRESHOLD=100` adaptive path (engine.py:604).
- Check template mismatch: `_extract_hidden_states` (engine.py:363) uses user-only template (no system prompt) to match training data. If you change the system prompt in `generate_responses`, probes may miscalibrate.

### CUDA out of memory
- Default model needs ~9 GB VRAM. Close other GPU applications.
- Model switching calls `del engine.model; gc.collect(); torch.cuda.empty_cache()`. If OOM persists, restart the process.

### Sentence segmentation issues
- pysbd may merge or split sentences unexpectedly.
- `align_tokens_to_sentences` (engine.py:181) uses character-span midpoint matching — can misalign on unusual Unicode or very long tokens.
- Non-claim sentences are filtered by `ClaimFilter.is_claim()`. If too many sentences show "SKIP", the regex patterns may be too aggressive for your use case.

### Adding a new model
1. Add model ID to `PROBE_BUNDLES` dict in app.py (line 33)
2. Add dropdown option in frontend/index.html (line 79 area)
3. Train probes for the new model using notebooks 01 + 02
4. Place the `.pkl` bundle in `backend/models/`
