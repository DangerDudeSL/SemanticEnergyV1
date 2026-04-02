# Probe Training Pipeline & Inference Reference

End-to-end documentation covering dataset preparation, LLM data extraction, teacher signal computation, probe training, and runtime inference for all 4 probe types.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Dataset: TriviaQA](#2-dataset-triviaqa)
3. [Asking the LLM Questions](#3-asking-the-llm-questions)
4. [What We Extract From the LLM](#4-what-we-extract-from-the-llm)
5. [Semantic Clustering](#5-semantic-clustering)
6. [Teacher Signal Computation](#6-teacher-signal-computation)
7. [Complete Record Structure](#7-complete-record-structure)
8. [Data Splitting](#8-data-splitting)
9. [Label Binarization](#9-label-binarization)
10. [The Four Probes](#10-the-four-probes)
11. [Per-Layer AUROC Sweep](#11-per-layer-auroc-sweep)
12. [Layer Range Selection](#12-layer-range-selection)
13. [Final Probe Training](#13-final-probe-training)
14. [Feature Ablation](#14-feature-ablation)
15. [Probe Bundle Assembly](#15-probe-bundle-assembly)
16. [Runtime Inference: TBG Mode](#16-runtime-inference-tbg-mode)
17. [Runtime Inference: SLT Mode](#17-runtime-inference-slt-mode)
18. [Non-Probe Signals Used at Inference](#18-non-probe-signals-used-at-inference)

---

## 1. Pipeline Overview

```
TriviaQA (1000 questions)
    |
    v
[Generate 5 responses per question]  ──>  token_ids, logits, probs, top2_logits
    |
    ├── [Semantic Clustering]  ──>  cluster assignments
    |       |
    |       ├── [Energy Teacher]   ──>  energy_score_raw   (cluster energy distribution)
    |       └── [Entropy Teacher]  ──>  entropy_score_raw  (cluster assignment entropy)
    |
    ├── [Hidden State Extraction]  ──>  TBG hidden (33, 4096) + SLT hidden (33, 4096)
    |       (separate forward pass on prompt + answer)
    |
    └── [Logit Feature Extraction]  ──>  9 scalar features
            |
            v
    [Binarize Teacher Scores]  ──>  energy_label (0/1), entropy_label (0/1)
            |
            v
    [80/10/10 Train/Val/Test Split]
            |
            v
    [Per-Layer AUROC Sweep]  ──>  33 AUROC values per probe x 4 probes
            |
            v
    [Layer Range Selection]  ──>  best contiguous window per probe
            |
            v
    [Train Final Probes]  ──>  4 LogisticRegression + 4 StandardScaler
            |
            v
    [Bundle + Pickle]  ──>  backend/models/probes_{model}_{dataset}.pkl
```

**Source notebooks:**
- `notebooks/01_generate_dataset.ipynb` — Steps 1-7 (dataset generation)
- `notebooks/02_train_se_probes.ipynb` — Steps 8-15 (training)
- `notebooks/05_complete_probe_training_pipeline.ipynb` — Combined single-notebook variant

---

## 2. Dataset: TriviaQA

**Source:** `notebooks/01_generate_dataset.ipynb` Section 2

```python
from datasets import load_dataset
triviaqa = load_dataset('trivia_qa', 'rc', split='validation', trust_remote_code=True)
dataset_subset = triviaqa.select(range(NUM_QUESTIONS))
```

| Property | Value |
|---|---|
| **Dataset** | TriviaQA |
| **Config** | `rc` (Reading Comprehension — has clean reference answers) |
| **Split** | `validation` |
| **Volume** | `NUM_QUESTIONS = 1000` (configurable) |
| **Library** | HuggingFace `datasets` |

### All Available TriviaQA Columns

The `rc` configuration of TriviaQA provides these fields for each record:

| Field | Type | Description |
|---|---|---|
| `question` | str | The trivia question text |
| `question_id` | str | Unique identifier for the question |
| `question_source` | str | Source URL where the question originated |
| `entity_pages` | dict | Wikipedia entity pages — contains sub-fields: `doc_source`, `filename`, `title`, `wiki_context` |
| `search_results` | dict | Web search results — contains sub-fields: `description`, `filename`, `rank`, `title`, `url`, `search_context` |
| `answer` | dict | Answer structure with 7 sub-fields (see below) |

**Answer sub-fields:**

| Sub-field | Type | Description |
|---|---|---|
| `answer.value` | str | The canonical answer text (e.g., "Paris") |
| `answer.aliases` | list[str] | Alternative correct phrasings (e.g., ["Paris", "paris", "City of Paris"]) |
| `answer.normalized_value` | str | Lowercased, cleaned version of the canonical answer |
| `answer.normalized_aliases` | list[str] | Normalized versions of all aliases |
| `answer.matched_wiki_entity_name` | str | Wikipedia entity matched to this answer |
| `answer.normalized_matched_wiki_entity_name` | str | Normalized version of the Wikipedia entity name |
| `answer.type` | str | Answer type/category |

### What We Actually Use (3 Fields Only)

Out of all the available fields, we only use three:

| Field | How It's Used |
|---|---|
| **`question`** | Fed to the LLM as the user prompt to generate 5 sample responses |
| **`question_id`** | Stored as `uid` in the generated record for identification |
| **`answer.aliases`** | Used to check correctness of the main answer via normalized substring matching (for evaluation, NOT training) |

Everything else — `entity_pages`, `search_results`, `question_source`, `answer.value`, `answer.type`, etc. — is completely ignored. We don't provide any context documents to the LLM; it answers from its own knowledge only. This is intentional: we want to measure the model's internal confidence about its own knowledge, not its ability to extract answers from given passages.

**Why TriviaQA?**
- Questions have clear correct/incorrect answers (needed for evaluation, not training)
- Short factual answers align well with hallucination detection
- Multiple reference aliases handle answer variations
- The `rc` config specifically has well-curated, clean reference answers with comprehensive alias lists

**Filtering:** None during selection. Records producing empty hidden states (answer too short for SLT extraction) are dropped post-hoc before saving.

---

## 3. Asking the LLM Questions

**Source:** `backend/engine.py:102-172` (`generate_responses`)

For each of the 1000 questions, the LLM generates 5 independent responses:

```python
NUM_SAMPLES = 5
gen_data = engine.generate_responses(question, num_samples=NUM_SAMPLES)
```

### Generation Configuration

```python
gen_cfg = GenerationConfig(
    do_sample=True,          # stochastic sampling (not greedy)
    temperature=0.7,         # diversity control
    max_new_tokens=512,      # max response length
    pad_token_id=tokenizer.pad_token_id,
)
```

### Chat Template

```python
messages = [
    {"role": "system", "content": "Answer the question directly and concisely. Do not explain your reasoning."},
    {"role": "user", "content": question},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The system prompt encourages short factual answers, which aligns with the TriviaQA evaluation format.

### Generation Call

```python
outputs = model.generate(
    **inputs,
    generation_config=gen_cfg,
    return_dict_in_generate=True,
    output_scores=True,      # per-token post-filtered softmax scores
    output_logits=True,      # raw unfiltered logits (for top-2 margin)
)
```

**Key flag distinction:**
- `output_scores=True` — scores AFTER temperature/top-k/top-p filtering (used for chosen-token probability)
- `output_logits=True` — raw logits BEFORE filtering (used for top-2 margin, since filtered scores have `-inf` for masked tokens)

Each of the 5 samples is generated independently in a loop (not batched), ensuring maximum diversity from the temperature sampling.

---

## 4. What We Extract From the LLM

### 4a. Per-Token Generation Data (from `generate`)

**Source:** `backend/engine.py:139-160`

For each generation step, we extract:

```python
for step_idx, score_tensor in enumerate(outputs.scores):
    token_id = gen_ids[step_idx]

    logits = score_tensor[0]
    prob = F.softmax(logits, dim=-1)[token_id].item()    # softmax prob of chosen token
    logit_val = logits[token_id].item()                  # raw logit of chosen token

    # Top-2 from UNFILTERED logits (not scores)
    raw_logit_vec = outputs.logits[step_idx][0]
    top2_vals = raw_logit_vec.topk(2).values
    top2_logits_list.append((top2_vals[0].item(), top2_vals[1].item()))
```

**Per-sample output dict:**

| Field | Type | Description |
|---|---|---|
| `answer` | str | Decoded text of generated response |
| `logits` | list[float] | Raw logit of chosen token at each step |
| `probs` | list[float] | Softmax probability of chosen token at each step |
| `token_ids` | list[int] | Generated token IDs (excluding EOS) |
| `top2_logits` | list[(float, float)] | (top-1 logit, top-2 logit) pairs from unfiltered logits |

### 4b. Hidden States (separate forward pass)

**Source:** `notebooks/01_generate_dataset.ipynb` Section 4, `backend/engine.py:351-401`

Hidden states are extracted in a **separate forward pass** after generation completes:

```python
# Why NOT model.generate(output_hidden_states=True)?
# That stores 512 steps x 33 layers x 4096 dims = ~1.3 GB per call — infeasible.

# Instead: single forward pass on (prompt + answer)
full_text = prompt_only + answer_text
outputs = model(**full_inputs, output_hidden_states=True)

hidden = torch.stack(outputs.hidden_states, dim=0)  # (33, 1, seq_len, 4096)
hidden = hidden[:, 0, :, :].float().cpu()             # (33, seq_len, 4096)
```

**Two positions are extracted:**

| Hidden State | Name | Position | Shape | Description |
|---|---|---|---|---|
| **TBG** | Token Before Generation | `hidden[:, prompt_len - 1, :]` | (33, 4096) | Last token of the prompt, before any answer is generated. Captures the model's "readiness state" — what it knows/doesn't know about the question. |
| **SLT** | Second-to-Last Token | `hidden[:, full_len - 2, :]` | (33, 4096) | Second-to-last token of the full sequence (token before EOS). Captures the model's accumulated state after generating the answer. |

**Dimensions for Llama 3.1 8B:**
- 33 layers (32 transformer layers + 1 embedding layer)
- 4096 hidden dimension
- Memory per record: `2 * 33 * 4096 * 4 bytes = ~1 MB`

**Chat template difference for hidden state extraction:**

```python
# During generation: includes system prompt
messages = [
    {"role": "system", "content": "Answer the question directly and concisely..."},
    {"role": "user", "content": question},
]

# During hidden state extraction: user-only (no system prompt)
messages = [
    {"role": "user", "content": question},
]
```

This user-only template for extraction ensures consistency between training and inference. The probes are trained on hidden states extracted without a system prompt.

### 4c. Logit Features (9 scalar summary statistics)

**Source:** `notebooks/01_generate_dataset.ipynb` Section 4 (`extract_logit_feats`)

From the first generated sample, 9 summary features are computed:

| Feature | Formula | Purpose |
|---|---|---|
| `mean_chosen_logit` | `mean(logits)` | Average model confidence across all tokens |
| `min_chosen_logit` | `min(logits)` | Weakest single-token prediction |
| `std_chosen_logit` | `std(logits)` | Consistency of confidence |
| `mean_logit_margin` | `mean(top1 - top2)` for finite margins | Average certainty gap |
| `min_logit_margin` | `min(finite margins)` | Least certain step |
| `std_logit_margin` | `std(finite margins)` | Consistency of certainty |
| `answer_len` | `len(logits)` | Number of generated tokens |
| `mean_chosen_prob` | `mean(probs)` | Average softmax probability |
| `min_chosen_prob` | `min(probs)` | Weakest token probability |

**Infinite margin handling:** When the model is fully certain at a step, top-2 logit is `-inf`, making the margin `+inf`. These infinite margins are filtered out before computing statistics to prevent NaN propagation.

---

## 5. Semantic Clustering

**Source:** `backend/engine.py:301-349`

The 5 generated answers are grouped by semantic equivalence using the LLM itself as a judge.

### Pairwise Equivalence Check

```python
def semantic_analyse(self, question, answer_a, answer_b):
    messages = [
        {"role": "system", "content": "You verify if two answers are semantically equivalent..."},
        {"role": "user", "content": f"Question: {question}\nGround Truth: {answer_a}\nStudent: {answer_b}\n..."},
    ]
    # Greedy decoding (do_sample=False, max_new_tokens=50)
    return "Yes" in result
```

### Greedy Clustering Algorithm

```python
def find_semantic_clusters(self, question, answer_list):
    clusters = []
    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and self.semantic_analyse(question, answer_list[i], answer_list[j]):
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)
    return clusters
```

**Properties:**
- **Order-dependent:** Answer 0 is always its own cluster prototype
- **Not transitive:** If A~B and B~C, A may not be verified ~C (greedy, no transitivity enforcement)
- **Worst-case:** `N*(N-1)/2 = 10` LLM calls for 5 samples

**Example outputs:**
- All agree: `[[0, 1, 2, 3, 4]]` — 1 cluster
- Split opinion: `[[0, 2, 4], [1, 3]]` — 2 clusters
- All disagree: `[[0], [1], [2], [3], [4]]` — 5 clusters

---

## 6. Teacher Signal Computation

### 6a. Energy Teacher

**Source:** `backend/engine.py:15-48`

The energy teacher measures how much of the model's "probability energy" is concentrated in the main answer's semantic cluster.

**Step 1 — Per-sample probability product:**
```
probs[i] = p_t1 * p_t2 * ... * p_tK    (product of all token probabilities)
```

**Step 2 — Per-sample logit mean:**
```
logits[i] = -mean(logit_t1, logit_t2, ..., logit_tK)    (negated mean)
```

**Step 3 — Cluster-level aggregation:**
```
normalized_probs = probs / sum(probs)

For each cluster c:
    probs_se[c]  = sum(normalized_probs[i] for i in cluster_c)
    logits_se[c] = -sum(logits[i] for i in cluster_c)
```

**Step 4 — Energy normalization:**
```
cluster_energies = logits_se / sum(logits_se)    # each in [0,1], all sum to 1
```

**Step 5 — Extract main cluster energy:**
```
energy_score_raw = cluster_energies[main_cluster_idx]
# where main_cluster_idx = cluster containing sample 0
```

| Property | Value |
|---|---|
| **Range** | [0, 1] |
| **Orientation** | Higher = more confident (more energy in main cluster) |
| **Correlation with correctness** | Positive (Spearman rho, verified in notebook) |

### 6b. Entropy Teacher

**Source:** `notebooks/01_generate_dataset.ipynb` Section 4

Shannon entropy over the cluster size distribution:

```python
def cluster_assignment_entropy(clusters):
    sizes = [len(c) for c in clusters]
    n = sum(sizes)                          # always 5 (num_samples)
    probs = [s / n for s in sizes]
    return -sum(p * math.log(p + 1e-10) for p in probs)
```

**Examples (5 samples):**

| Cluster Configuration | Entropy | Interpretation |
|---|---|---|
| `[[0,1,2,3,4]]` (all agree) | ~0.0 | Most certain |
| `[[0,1,2], [3,4]]` (3+2 split) | ~0.673 | Moderate uncertainty |
| `[[0,1], [2,3], [4]]` (2+2+1) | ~1.055 | High uncertainty |
| `[[0],[1],[2],[3],[4]]` (all disagree) | ~1.609 | Maximum uncertainty |

| Property | Value |
|---|---|
| **Range** | [0, ln(5)] ~= [0, 1.609] for 5 samples |
| **Orientation** | Higher = more uncertain (higher hallucination risk) |
| **Correlation with correctness** | Negative (verified in notebook) |

### 6c. Correctness (for evaluation only)

**Source:** `notebooks/01_generate_dataset.ipynb` Section 4

```python
def is_correct_triviaqa(predicted_answer, reference_aliases):
    norm_pred = normalize_answer(predicted_answer)       # lowercase, remove articles/punctuation
    for ref in reference_aliases:
        if normalize_answer(ref) in norm_pred:           # substring match
            return 1.0
    return 0.0
```

**Important:** Correctness is used ONLY for:
- Evaluating probe performance (hallucination detection AUROC)
- Computing teacher upper bounds
- It is **NOT** used as a training label

---

## 7. Complete Record Structure

**Source:** `notebooks/01_generate_dataset.ipynb` Section 6

Each of the ~1000 records contains:

```python
record = {
    # Identity
    'uid':                     str,           # question_id from TriviaQA
    'question':                str,           # the question text
    'main_answer':             str,           # first sample's generated answer

    # Correctness (evaluation only, NOT used for training)
    'correctness':             float,         # 0.0 or 1.0 (normalized substring match)

    # Teacher signals (continuous)
    'energy_score_raw':        float,         # [0, 1] — cluster energy confidence
    'entropy_score_raw':       float,         # [0, ~1.6] — cluster assignment entropy

    # Binarized labels (set in notebook 02, initially None)
    'energy_label':            int | None,    # 1 = high energy (confident)
    'entropy_label':           int | None,    # 1 = high entropy (uncertain)

    # Hidden states (the probe training features)
    'emb_last_tok_before_gen': np.ndarray,    # (33, 4096) — TBG hidden state
    'emb_tok_before_eos':      np.ndarray,    # (33, 4096) — SLT hidden state

    # Logit features (secondary, for ablation)
    'logit_feats':             dict,          # 9 scalar features from sample 0
    'token_ids':               list[int],     # generated token IDs

    # Clustering metadata
    'num_clusters':            int,           # number of semantic clusters
    'cluster_sizes':           list[int],     # size of each cluster
}
```

**Dataset file:** `backend/data/probe_dataset_{model}_{dataset}.pkl` (~541 MB for 1000 records)

### How the pkl File Is Saved

The generated dataset is saved as a simple Python list of dictionaries via pickle:

```python
# Filter out records where the answer was too short for SLT extraction
final_records = [r for r in all_records if r['emb_last_tok_before_gen'] is not None]

with open(output_path, 'wb') as f:
    pickle.dump(final_records, f)
```

The file is a flat list — no special indexing or database structure. When loaded in notebook 02, the `ProbeDataset` class unpacks it into separate arrays:

```python
# What ProbeDataset extracts from the pkl:
energy_score_raw  → (N,) array       # continuous teacher score
entropy_score_raw → (N,) array       # continuous teacher score
correctness       → (N,) array       # for evaluation only
tbg_states        → (N, 33, 4096)    # stacked emb_last_tok_before_gen
slt_states        → (N, 33, 4096)    # stacked emb_tok_before_eos
logit_feats       → (N, 4) array     # only 4 of the 9 features used (see note below)
```

**Note on logit features at training time:** Although 9 features are stored in the pkl, only 4 are actually used during probe training: `mean_chosen_logit`, `min_chosen_logit`, `std_chosen_logit`, and `answer_len`. The margin-based features (`mean_logit_margin`, `min_logit_margin`, `std_logit_margin`) are excluded because the `transformers` library's post-filtering step can produce `-inf` values for top-2 logits, making those margins unreliable. The probability features (`mean_chosen_prob`, `min_chosen_prob`) are also excluded as they are redundant with the logit features.

---

## 8. Data Splitting

**Source:** `notebooks/02_train_se_probes.ipynb` Section 3

```python
np.random.seed(42)                # reproducible splits
N = len(all_records)
idx = np.random.permutation(N)

n_train = int(0.80 * N)          # 80% — fit probes and scalers
n_val   = int(0.10 * N)          # 10% — layer sweep AUROC evaluation
n_test  = N - n_train - n_val    # 10% — final evaluation with bootstrap CI
```

| Split | Purpose | Used For |
|---|---|---|
| **Train (80%)** | Fit LogisticRegression + StandardScaler | Binarization thresholds, scaler fitting, probe fitting |
| **Val (10%)** | Model selection | Per-layer AUROC evaluation, layer range selection |
| **Test (10%)** | Final evaluation | Bootstrap 95% CI, teacher fidelity, ablation |

**No cross-validation** is used. Hyperparameters are fixed (not tuned).

---

## 9. Label Binarization

**Source:** `notebooks/02_train_se_probes.ipynb` Section 4

The continuous teacher scores are binarized into 0/1 labels for LogisticRegression training. The threshold is chosen to minimize within-group variance (SEP-style).

### Threshold Selection Algorithm

```python
def find_best_threshold(scores, label='scores'):
    best_thresh = None
    best_mse    = float('inf')

    for pct in np.arange(10, 91, 1):          # sweep percentiles 10th to 90th
        thresh = np.percentile(scores, pct)
        g0 = scores[scores <  thresh]          # low group
        g1 = scores[scores >= thresh]          # high group
        if len(g0) == 0 or len(g1) == 0:
            continue

        # Weighted within-group MSE
        mse = (len(g0) * g0.var() + len(g1) * g1.var()) / len(scores)

        if mse < best_mse:
            best_mse    = mse
            best_thresh = thresh

    return best_thresh
```

**Key property:** This is computed on the TRAIN split only. The same thresholds are then applied to val and test splits.

### Applying Binarization

```python
T_energy  = find_best_threshold(D_train.energy_score_raw)   # e.g., ~0.75
T_entropy = find_best_threshold(D_train.entropy_score_raw)  # e.g., ~0.21

for D in [D_train, D_val, D_test]:
    D.energy_label  = (D.energy_score_raw  >= T_energy).astype(int)   # 1 = high confidence
    D.entropy_label = (D.entropy_score_raw >= T_entropy).astype(int)  # 1 = high uncertainty
```

**Important design choice:** No correctness labels are involved in binarization. The thresholds are derived purely from the teacher score distributions. This means the probes learn to approximate the teacher signals (energy/entropy), not to directly predict correctness.

---

## 10. The Four Probes

| Probe | Token Position | Teacher Label | What It Predicts | Deployment Mode | Risk Inversion |
|---|---|---|---|---|---|
| **TBG Energy** | Last prompt token | `energy_label` | P(high confidence) | Pre-generation | risk = 1 - output |
| **TBG Entropy** | Last prompt token | `entropy_label` | P(high uncertainty) | Pre-generation | risk = output directly |
| **SLT Energy** | 2nd-to-last generated token | `energy_label` | P(high confidence) | Post-generation | risk = 1 - output |
| **SLT Entropy** | 2nd-to-last generated token | `entropy_label` | P(high uncertainty) | Post-generation | risk = output directly |

**Why two positions?**
- **TBG (pre-generation):** Can estimate risk before generating any answer. Fastest mode — single forward pass, no generation needed. Useful for deciding whether to attempt a question at all.
- **SLT (post-generation):** Has access to what the model actually generated. More informative but requires generation + extra forward pass.

**Why two teachers?**
- **Energy** measures how concentrated the probability mass is in the main answer's cluster (Boltzmann-inspired)
- **Entropy** measures how spread out the cluster assignments are (information-theoretic)
- They capture overlapping but not identical signals (Spearman rho ~ -0.7, confirmed in cross-signal analysis)

---

## 11. Per-Layer AUROC Sweep

**Source:** `notebooks/02_train_se_probes.ipynb` Sections 6-7

Before selecting layer ranges, every single layer is evaluated independently to understand where the most informative representations live.

### Training Function

```python
def sklearn_train_eval(X_train, y_train, X_val, y_val, scale=True):
    X_train = clean_X(X_train)                    # replace inf/nan with 0
    X_val   = clean_X(X_val)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)   # fit on train only
        X_val   = scaler.transform(X_val)          # apply train statistics

    probe = LogisticRegression(max_iter=1000, C=1.0)   # L2 regularization, default C
    probe.fit(X_train, y_train)

    y_score = probe.predict_proba(X_val)[:, 1]    # probability of class 1
    auroc = roc_auc_score(y_val, y_score)

    return probe, scaler, auroc
```

### Sweep Process

```python
for layer in range(33):  # all layers including embedding
    X_train = get_layer_X(D_train, layer, 'slt')   # (N_train, 4096)
    X_val   = get_layer_X(D_val,   layer, 'slt')   # (N_val, 4096)
    _, _, auroc = sklearn_train_eval(X_train, D_train.energy_label, X_val, D_val.energy_label)
    energy_slt_aurocs.append(auroc)
```

This runs **4 sweeps** (2 teachers x 2 token positions), training `4 x 33 = 132` probes total. Each probe:
- Uses a single layer's hidden state: `(N, 4096)` features
- Is trained with StandardScaler + LogisticRegression
- Is evaluated by AUROC on the validation set

The sweep produces 4 lists of 33 AUROC values, which are plotted as layer-vs-AUROC curves.

**Typical pattern:**
- Early layers (0-5): near chance (0.5) — too low-level
- Middle layers (10-20): rising AUROC — semantic information forming
- Later layers (20-30): peak AUROC — highest semantic content
- Final layer (32): sometimes drops — may be too task-specific

---

## 12. Layer Range Selection

**Source:** `notebooks/02_train_se_probes.ipynb` Section 8

Rather than using a single best layer, a contiguous window of layers is concatenated for richer features.

### Selection Algorithm

```python
def decide_layer_range(auroc_list, window_sizes=[4, 8, 16], min_window=4):
    aucs = np.array(auroc_list)    # 33 AUROC values
    best_mean  = -np.inf
    best_range = (0, min_window)

    for window in window_sizes:               # try windows of 4, 8, and 16
        for start in range(len(aucs) - window + 1):  # all valid start positions
            end = start + window
            mean_auc = aucs[start:end].mean()         # average AUROC in this window

            if mean_auc > best_mean:
                best_mean  = mean_auc
                best_range = (start, end)

    return best_mean, best_range
```

**How it works:**
1. Try all window sizes: 4, 8, and 16 contiguous layers
2. Slide each window across all 33 layers
3. Compute mean AUROC within the window
4. Pick the window with the highest mean AUROC

**Output:** `(start, end)` in Python slice notation, e.g., `(20, 24)` means layers 20, 21, 22, 23.

**Feature dimensions after selection:**

| Window Size | Features per Probe |
|---|---|
| 4 layers | 4 x 4096 = 16,384 |
| 8 layers | 8 x 4096 = 32,768 |
| 16 layers | 16 x 4096 = 65,536 |

The 4 probes typically select different layer ranges, since energy and entropy signals may concentrate at different depths, and TBG/SLT positions capture different information.

---

## 13. Final Probe Training

**Source:** `notebooks/02_train_se_probes.ipynb` Section 10

Once layer ranges are selected, final probes are trained and evaluated.

### Training

```python
configs = [
    ('slt_energy',  'energy_label',  'slt', best_energy_slt_range),
    ('tbg_energy',  'energy_label',  'tbg', best_energy_tbg_range),
    ('slt_entropy', 'entropy_label', 'slt', best_entropy_slt_range),
    ('tbg_entropy', 'entropy_label', 'tbg', best_entropy_tbg_range),
]

for probe_name, label_key, token, layer_range in configs:
    X_train = get_range_X(D_train, layer_range, token)  # (N_train, window*4096)
    y_train = getattr(D_train, label_key)

    probe, scaler, val_auroc = sklearn_train_eval(
        X_train, y_train,
        get_range_X(D_val, layer_range, token), getattr(D_val, label_key)
    )
```

### Bootstrap Evaluation (95% CI)

```python
def bootstrap_auroc(probe, scaler, X_test, y_test, n_boot=1000, ci=0.95):
    y_score = probe.predict_proba(scaler.transform(X_test))[:, 1]
    base_auroc = roc_auc_score(y_test, y_score)

    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_test), len(y_test))  # resample with replacement
        if len(np.unique(y_test[idx])) < 2:
            continue                                       # skip single-class samples
        boot_aurocs.append(roc_auc_score(y_test[idx], y_score[idx]))

    lo = np.percentile(boot_aurocs, 2.5)
    hi = np.percentile(boot_aurocs, 97.5)
    return {'mean': base_auroc, 'lo': lo, 'hi': hi}
```

### Teacher Fidelity

Spearman rho between probe's continuous output (`predict_proba[:, 1]`) and the raw teacher score is computed to verify the probe faithfully approximates the teacher.

### Teacher Upper Bounds

Full teachers (expensive multi-sample clustering) serve as upper bounds:
```python
energy_upper_auroc  = roc_auc_score(hallucination_labels, 1 - D_test.energy_score_raw)
entropy_upper_auroc = roc_auc_score(hallucination_labels, D_test.entropy_score_raw)
```

Probes should approach but not exceed these values (they approximate the teacher with a single forward pass).

---

## 14. Feature Ablation

**Source:** `notebooks/02_train_se_probes.ipynb` Section 9

Three configurations are compared to determine whether logit features add value:

| Configuration | Features | Scaling |
|---|---|---|
| **Hidden only** | `hidden[l0:l1, :].reshape(1, -1)` | Single StandardScaler |
| **Logit only** | 4 scalar logit features | Single StandardScaler |
| **Hidden + Logit** | Concatenated after separate scaling | Two StandardScalers, then concatenate |

```python
def ablation_eval(D_train, D_val, label_key, token, layer_range):
    # 1. Hidden states only
    _, _, auc_h = sklearn_train_eval(X_train_h, y_train, X_val_h, y_val)

    # 2. Logit features only (4 scalars)
    _, _, auc_l = sklearn_train_eval(D_train.logit_feats, y_train, D_val.logit_feats, y_val)

    # 3. Hidden + logit (separately scaled, then concatenated)
    X_train_hl = np.concatenate([scaler_h.fit_transform(X_h), scaler_l.fit_transform(X_l)], axis=1)
    probe_hl = LogisticRegression(max_iter=1000, C=1.0)
    probe_hl.fit(X_train_hl, y_train)
    auc_hl = roc_auc_score(y_val, probe_hl.predict_proba(X_val_hl)[:, 1])

    return {'hidden_only': auc_h, 'logit_only': auc_l, 'hidden_logit': auc_hl}
```

The final probes use **hidden states only** (logit features did not significantly improve AUROC in practice).

---

## 15. Probe Bundle Assembly

**Source:** `notebooks/02_train_se_probes.ipynb` Section 16

All trained probes, scalers, and metadata are packaged into a single pickle file.

### Bundle Structure

```python
probe_bundle = {
    # ── Metadata ──
    'model_id':                'meta-llama/Llama-3.1-8B-Instruct',
    'dataset':                 'trivia_qa',
    'num_train_records':       D_train.N,

    # ── Binarization thresholds (from train split) ──
    'energy_threshold':        T_energy,       # float, e.g., 0.75
    'entropy_threshold':       T_entropy,      # float, e.g., 0.21

    # ── Best layer ranges (4 probes, each a (start, end) tuple) ──
    'best_energy_slt_range':   (l0, l1),       # e.g., (17, 21)
    'best_energy_tbg_range':   (l0, l1),       # e.g., (28, 32)
    'best_entropy_slt_range':  (l0, l1),       # e.g., (20, 24)
    'best_entropy_tbg_range':  (l0, l1),       # e.g., (21, 25)

    # ── Trained probes (4 LogisticRegression objects) ──
    'slt_energy_probe':        LogisticRegression,
    'tbg_energy_probe':        LogisticRegression,
    'slt_entropy_probe':       LogisticRegression,
    'tbg_entropy_probe':       LogisticRegression,

    # ── Fitted scalers (4 StandardScaler objects) ──
    'slt_energy_scaler':       StandardScaler,  # fit on train hidden states
    'tbg_energy_scaler':       StandardScaler,
    'slt_entropy_scaler':      StandardScaler,
    'tbg_entropy_scaler':      StandardScaler,

    # ── Per-layer AUROC results (for analysis/plots) ──
    'layer_auroc_table': {
        'energy_slt':  [auroc_L0, auroc_L1, ..., auroc_L32],   # 33 floats
        'energy_tbg':  [...],
        'entropy_slt': [...],
        'entropy_tbg': [...],
    },

    # ── Test set evaluation results ──
    'id_test_results': {
        'slt_energy':  {'mean': float, 'lo': float, 'hi': float},
        'tbg_energy':  {'mean': float, 'lo': float, 'hi': float},
        'slt_entropy': {'mean': float, 'lo': float, 'hi': float},
        'tbg_entropy': {'mean': float, 'lo': float, 'hi': float},
    },
}
```

### Saving

```python
output_path = os.path.join(MODELS_DIR, 'probes_llama3-8b_triviaqa.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(probe_bundle, f)
```

**File size:** ~2 MB per bundle (LogisticRegression weights are small)

### Model-to-Bundle Mapping

At runtime, `backend/app.py` maps model IDs to their bundle files:

```python
PROBE_BUNDLES = {
    "meta-llama/Llama-3.1-8B-Instruct": "probes_llama3-8b_triviaqa.pkl",
    "Qwen/Qwen3-8B": "probes_qwen3-8b_triviaqa.pkl",
}
```

---

## 16. Runtime Inference: TBG Mode

**Source:** `backend/engine.py:403-459` (`score_with_tbg_probe`), `backend/app.py:238-264` (`/score_fast_tbg`)

TBG mode provides **pre-generation risk estimation**. It answers: "Before generating anything, how likely is this question to produce a hallucination?"

### Step-by-Step Flow

```
User prompt
    |
    v
[Single forward pass on prompt only — NO generation]
    |
    v
[Extract TBG hidden state at last prompt token]
    tbg_hidden = hidden[:, prompt_len - 1, :]    # (33, 4096)
    |
    ├── [Energy Probe]
    |   X = tbg_hidden[l0:l1, :].reshape(1, -1)          # slice to best layers
    |   X_scaled = tbg_energy_scaler.transform(X)          # normalize features
    |   energy_confidence = tbg_energy_probe.predict_proba(X_scaled)[0, 1]
    |   energy_risk = 1.0 - energy_confidence               # INVERT: probe predicts confidence
    |
    └── [Entropy Probe]
        X = tbg_hidden[l0h:l1h, :].reshape(1, -1)         # different layer range
        X_scaled = tbg_entropy_scaler.transform(X)
        entropy_risk = tbg_entropy_probe.predict_proba(X_scaled)[0, 1]  # NO inversion: predicts uncertainty
    |
    v
[Combined Risk]
    combined_risk = (energy_risk + entropy_risk) / 2.0      # simple average
    |
    v
[Confidence Level]
    < 0.35  ->  "high"    (low risk)
    < 0.65  ->  "medium"
    >= 0.65 ->  "low"     (high risk)
```

### What TBG Mode Does NOT Use

- No generated answer (no generation step at all)
- No B1 logit confidence (no logits to analyze)
- No sentence-level scoring (no sentences)
- No claim filtering
- **Purely probe-based:** only the two probes on the prompt hidden state

### Output

```python
{
    "mode": "tbg_pre_generation",
    "energy_risk": float,       # [0, 1]
    "entropy_risk": float,      # [0, 1]
    "combined_risk": float,     # [0, 1]
    "confidence_level": str,    # "high" | "medium" | "low"
}
```

---

## 17. Runtime Inference: SLT Mode

**Source:** `backend/engine.py:461-671` (`score_with_slt_probe`), `backend/app.py:267-293` (`/score_fast_slt`)

SLT mode provides **post-generation risk estimation** with sentence-level granularity. It generates one answer, then scores it using both probe-based and logit-based signals.

### Step-by-Step Flow

#### Step 1: Generate One Answer

```python
gen_data = engine.generate_responses(question, num_samples=1)
answer_text = gen_data[0]["answer"]
```

Extracts: `token_ids`, `logits`, `probs`, `top2_logits`

#### Step 2: B1 Sentence-Level Logit Confidence (non-probe baseline)

**Source:** `engine.py:215-299`

This is NOT from the probe bundle. It's computed purely from the generation logits.

1. **Split** answer into sentences using `pysbd`
2. **Align** tokens to sentences via character-span midpoint overlap
3. **Group** logits and margins by sentence
4. **Sigmoid mapping** per sentence:
   ```
   mean_logit = mean(chosen_token_logits_in_sentence)
   confidence = 1 / (1 + exp(-(mean_logit - 33.0) / 3.0))
   ```
   - Center=33.0, Scale=3.0, calibrated for Llama 3.1 8B 4-bit
5. **Margin boost** (if mean margin > 3.0):
   ```
   boost = min(0.1, (mean_margin - 3.0) * 0.02)
   confidence = min(1.0, confidence + boost)
   ```
6. **Claim filtering** via `ClaimFilter` (regex-based):
   - Non-claims (filler, meta, hedging) get `level="none"`, `confidence=None`
   - Only claim sentences are scored
7. **Level bucketing:**
   ```
   >= 0.6 -> "high"
   >= 0.3 -> "medium"
   < 0.3  -> "low"
   ```

#### Step 3: Extract Hidden States

```python
# Find last token position of each claim sentence
sent_end_positions = [last_token_idx_of_sentence_s for s in sentences]
valid_positions = [pos for (s, pos) in ... if is_claim]

# One forward pass extracts all positions
tbg_hidden, slt_hidden, sent_hiddens = _extract_hidden_states(
    question, answer_text,
    extra_positions=valid_positions   # sentence-end positions for per-sentence probes
)
```

This single forward pass yields:
- `slt_hidden`: (33, 4096) at second-to-last token — for overall scoring
- `sent_hiddens`: list of (33, 4096) at each claim sentence's end token — for per-sentence scoring

#### Step 4: Overall SLT Probe Scores

```python
# Energy probe (confidence -> invert for risk)
X_e = slt_hidden[l0:l1, :].reshape(1, -1)
energy_risk = 1.0 - slt_energy_probe.predict_proba(slt_energy_scaler.transform(X_e))[0, 1]

# Entropy probe (uncertainty = risk directly)
X_h = slt_hidden[l0h:l1h, :].reshape(1, -1)
entropy_risk = slt_entropy_probe.predict_proba(slt_entropy_scaler.transform(X_h))[0, 1]
```

#### Step 5: Per-Sentence Dual-Probe Scoring

For each claim sentence, both probes run on the hidden state at that sentence's end token:

```python
W_ENTROPY = 0.515    # weight from validation AUROC 0.773
W_ENERGY  = 0.485    # weight from validation AUROC 0.727

for each claim sentence s:
    h = sent_hiddens[s]    # (33, 4096) at sentence-end token

    # Energy risk at this sentence
    sent_energy_risk = 1.0 - slt_energy_probe.predict_proba(scaled(h[l0:l1]))[0, 1]

    # Entropy risk at this sentence
    sent_entropy_risk = slt_entropy_probe.predict_proba(scaled(h[l0h:l1h]))[0, 1]

    # AUROC-weighted combination
    probe_risk = W_ENTROPY * sent_entropy_risk + W_ENERGY * sent_energy_risk
```

**Probe override rule:** Probes can only **raise** the risk level, never lower it below the B1 baseline:
```python
if probe_risk >= 0.65:
    sentence_scores[s]["level"] = "low"       # override to low confidence
elif probe_risk >= 0.35:
    sentence_scores[s]["level"] = "medium"    # override to medium
# If probe_risk < 0.35: keep existing B1 logit-based level (don't downgrade)
```

#### Step 6: Token-Length Conditional Aggregation

The final `combined_risk` is computed differently based on answer length, because the SLT token's representational quality degrades for long answers.

**Short answers (<=100 tokens):**
```python
slt_combined = (energy_risk + entropy_risk) / 2.0
per_sent_risks = [s.probe_risk for s in claim_sentences]

if len(per_sent_risks) >= 2:
    # SLT only sees the end; blend with per-sentence average
    combined_risk = 0.5 * slt_combined + 0.5 * mean(per_sent_risks)
else:
    # 0-1 claims: SLT covers whole answer adequately
    combined_risk = slt_combined
```

**Long answers (>100 tokens):**
```python
n = len(per_sent_risks)
max_sent_risk  = max(per_sent_risks)
mean_sent_risk = mean(per_sent_risks)

slt_weight  = 0.15                            # SLT unreliable for long answers
max_weight  = 0.25 / (1.0 + ln(max(n, 1)))   # diminishes with more claims
mean_weight = 1.0 - slt_weight - max_weight   # gets the remainder

combined_risk = slt_weight  * entropy_risk       # SLT anchor (entropy only)
              + max_weight  * max_sent_risk       # worst-case sentence
              + mean_weight * mean_sent_risk      # average across sentences
```

**Design rationale:**
- **SLT weight = 15%:** Minimal anchor because SLT represents only the end of a long answer
- **Max weight decreases logarithmically:** With many claims, the single worst sentence shouldn't dominate
- **Only entropy_risk for SLT component:** In long answers, entropy probe was found more reliable than energy

#### Step 7: Final Output

```python
{
    "mode": "slt_post_generation",
    "answer": str,
    "energy_risk": float,                   # overall SLT energy risk
    "entropy_risk": float,                  # overall SLT entropy risk
    "combined_risk": float,                 # aggregated final risk
    "confidence_level": "high"|"medium"|"low",
    "sentence_scores": [                    # per-sentence breakdown
        {
            "text": str,
            "confidence": float | None,     # B1 logit confidence
            "level": str,                   # may be overridden by probes
            "num_tokens": int,
            "mean_chosen_logit": float,
            "mean_logit_margin": float | None,
            "is_claim": bool,
            "energy_risk": float | None,    # per-sentence energy probe
            "entropy_risk": float | None,   # per-sentence entropy probe
            "probe_risk": float | None,     # AUROC-weighted combo
        },
        ...
    ],
    "sentence_avg_confidence": float | None,  # mean B1 confidence across claims
}
```

---

## 18. Non-Probe Signals Used at Inference

### Summary Table

| Signal | TBG Mode | SLT Mode | Source | From Probe Bundle? |
|---|---|---|---|---|
| **Energy probe** | Yes | Yes (overall + per-sentence) | `probe_bundle` | Yes |
| **Entropy probe** | Yes | Yes (overall + per-sentence) | `probe_bundle` | Yes |
| **B1 logit confidence** | No | Yes | Generation logits | No |
| **Claim filtering** | No | Yes | `ClaimFilter` regex | No |
| **Sentence segmentation** | No | Yes | `pysbd` library | No |
| **Token-to-sentence alignment** | No | Yes | Character-span midpoint | No |
| **Margin boost** | No | Yes | Top-2 logit gap | No |

### Detail on Non-Probe Signals in SLT Mode

**1. B1 Logit Confidence** (`engine.py:215-299`)
- Sigmoid-mapped mean chosen-token logit per sentence
- Provides a baseline confidence estimate from the generation step alone
- No extra model calls required — purely post-processes existing per-token logits
- Calibration constants (center=33.0, scale=3.0) are hardcoded for Llama 3.1 8B 4-bit

**2. Margin Boost** (`engine.py:272-275`)
- When the average gap between top-1 and top-2 logit choices exceeds 3.0 for a sentence
- Boosts B1 confidence by up to +0.1 (rate: 0.02 per unit above threshold)
- Captures cases where the model is very certain about its token choices

**3. Claim Filtering** (`backend/claim_filter.py`)
- Regex-based classification of sentences as claims vs. non-claims
- Non-claims (filler, meta, hedging, questions, greetings) are excluded from all scoring
- Zero overhead — no ML model, just compiled regex patterns
- Conservative default: sentences are treated as claims unless matched by a non-claim pattern

**4. Sentence Segmentation** (`pysbd` library)
- Splits answer text into sentences using the `pysbd` (Pragmatic Sentence Boundary Disambiguation) library
- Required for per-sentence probe scoring and B1 scoring
- Only runs when answer has 2+ sentences

**5. Token-to-Sentence Alignment** (`engine.py:181-213`)
- Maps each token to its parent sentence using character-span midpoint overlap
- Required to: (a) group logits by sentence for B1 scoring, (b) find sentence-end token positions for per-sentence probe extraction

### How Probe and Non-Probe Signals Interact

```
B1 Logit Confidence (non-probe)
    |
    v
[Sets initial per-sentence confidence level]
    |
    v
Per-Sentence Probe Risk (probe-based)
    |
    v
[Can OVERRIDE level UPWARD only]
    - probe_risk >= 0.65 -> force level to "low"
    - probe_risk >= 0.35 -> force level to "medium"
    - probe_risk < 0.35  -> keep B1 level unchanged
    |
    v
[Token-length conditional aggregation]
    - Blends overall SLT probe scores with per-sentence probe risks
    - Weights depend on answer length (short vs long strategy)
    |
    v
Final combined_risk and confidence_level
```

The B1 logit confidence serves as a conservative floor — if the logits already indicate low confidence, the probes won't override that downward. The probes add value by catching cases where logits look fine but the hidden state representation reveals uncertainty.
