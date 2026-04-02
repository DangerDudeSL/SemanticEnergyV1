# 7.3 Core Functionalities Implementation

This section presents the design, implementation, and evaluation of SemanticEnergy, a local hallucination detection system for large language model (LLM) outputs. The discussion follows a narrative structure: beginning with the theoretical foundations, progressing through dataset construction and probe training, and culminating in the system architecture and integration that delivers real-time hallucination scoring.

---

## 7.3.1 Foundational Frameworks: Semantic Entropy and Semantic Energy

The detection of hallucinations in LLM outputs requires a principled measure of model uncertainty. Two complementary frameworks provide this foundation: Semantic Entropy and Semantic Energy.

### Semantic Entropy

Semantic Entropy, introduced by Kuhn et al. (2023), measures uncertainty by examining whether a model produces consistent answers across multiple independent generations. The procedure is as follows:

1. Generate *N* independent responses to the same prompt (using stochastic sampling).
2. Cluster the responses by **semantic equivalence** — grouping answers that convey the same meaning, regardless of exact wording.
3. Compute the Shannon entropy over the cluster size distribution.

If all responses converge to a single meaning, entropy is near zero (high certainty). If responses scatter across many distinct meanings, entropy is high (significant uncertainty, elevated hallucination risk).

**Formula 1 — Shannon Entropy over Cluster Assignments:**

$$H = -\sum_{c \in \text{clusters}} \frac{|c|}{N} \cdot \ln\!\left(\frac{|c|}{N}\right)$$

where |*c*| is the number of samples in cluster *c*, and *N* is the total number of generated samples.

### Semantic Energy

Semantic Energy extends the entropy approach by incorporating **token-level logit magnitudes** — a measure of how confident the model was about each individual token it generated. The analogy is drawn from statistical mechanics: in a physical system at low temperature, particles settle into a single low-energy state (ordered, certain); at high temperature, they scatter across many states (disordered, uncertain).

Rather than merely counting how many samples fall into each cluster, Semantic Energy weights each sample by the model's per-token confidence, then aggregates these weights by cluster.

**Formula 2 — Boltzmann Energy per Sample:**

$$E_i = -\frac{1}{K_i} \sum_{t=1}^{K_i} \ell_{i,t}$$

where *ℓ*<sub>*i,t*</sub> is the raw logit of the chosen token at generation step *t* for sample *i*, and *K*<sub>*i*</sub> is the number of tokens generated. A more negative energy indicates higher token-level confidence.

**Formula 3 — Cluster-Level Energy Aggregation and Normalisation:**

$$E_{\text{cluster}}(c) = -\sum_{i \in c} E_i$$

$$\text{energy\_score}(c) = \frac{E_{\text{cluster}}(c)}{\sum_{c'} E_{\text{cluster}}(c')}$$

The final confidence score for the main answer is the normalised energy share of its cluster — a value in [0, 1] where higher indicates greater confidence.

### Comparison of the Two Frameworks

| Property | Semantic Entropy | Semantic Energy |
|---|---|---|
| Input signal | Cluster sizes only | Cluster sizes **and** per-token logits |
| Output range | [0, ln *N*] | [0, 1] |
| Orientation | Higher = more uncertain | Higher = more confident |
| Sensitive to token quality | No | Yes — high-logit tokens increase cluster weight |
| Correlation with correctness | Negative | Positive |

*Table 7.1 — Comparison of Semantic Entropy and Semantic Energy frameworks.*

Both frameworks share the same clustering step but produce complementary views of model certainty: entropy captures **structural** uncertainty (how many distinct meanings exist), while energy captures both structural and **qualitative** uncertainty (how confidently each meaning was produced).

> **Figure 7.1** (Process Diagram): Five independent generations for a single prompt are passed through pairwise semantic equivalence checking and grouped into clusters. Two parallel teacher signals are computed from the same clustering: the **energy teacher** aggregates per-token logit magnitudes by cluster, while the **entropy teacher** computes Shannon entropy over cluster sizes.

---

## 7.3.2 The Computational Cost Problem and Motivation for Linear Probes

The full Semantic Energy pipeline, while theoretically sound, is computationally prohibitive for real-time applications.

| Operation | Forward Passes | Typical Latency (RTX 3060) |
|---|---|---|
| Generate *N* = 5 responses | 5 | ~40s |
| Pairwise semantic equivalence checks | Up to *N*(*N*−1)/2 = 10 | ~30s |
| Energy and entropy computation | Arithmetic only | <1ms |
| **Total** | **Up to 15** | **60–120s** |

*Table 7.2 — Computational cost breakdown of the full Semantic Energy pipeline.*

A 60–120 second latency per query is unacceptable for interactive use. This motivates the central research question of the probe-based approach:

> *Can a single forward pass through the model's internal hidden states predict what the full multi-sample pipeline would have computed?*

The hypothesis rests on a key insight: when an LLM processes a question, its internal representations (hidden states) already encode whether it possesses strong knowledge about the answer. The full pipeline makes this confidence visible through expensive behavioural analysis; a **linear probe** can learn to read the same signal directly from hidden states, bypassing the multi-sample generation and clustering entirely.

Two token positions yield two probe families:

- **Token Before Generation (TBG):** The hidden state at the last prompt token, captured before any answer is generated. This enables pre-generation risk screening.
- **Second-to-Last Token (SLT):** The hidden state at the second-to-last generated token, captured after an answer has been produced. This enables post-generation confidence scoring with per-sentence granularity.

Two teacher signals (energy and entropy) combined with two token positions produce a matrix of **four probes**.

| Probe | Token Position | Teacher Signal | Availability | Primary Use Case |
|---|---|---|---|---|
| TBG Energy | Last prompt token | Binarised energy score | Before generation | Ultra-fast pre-screening |
| TBG Entropy | Last prompt token | Binarised entropy score | Before generation | Ultra-fast pre-screening |
| SLT Energy | Second-to-last generated token | Binarised energy score | After generation | Per-sentence scoring |
| SLT Entropy | Second-to-last generated token | Binarised entropy score | After generation | Per-sentence scoring |

*Table 7.3 — The four linear probes and their characteristics.*

---

## 7.3.3 Dataset Construction

### 7.3.3.1 TriviaQA as Source Corpus

The training dataset is derived from TriviaQA (Joshi et al., 2017), a large-scale reading comprehension dataset with trivia-style factual questions.

| Property | Value |
|---|---|
| Dataset | TriviaQA |
| Configuration | `rc` (Reading Comprehension) |
| Split | Validation |
| Questions processed | 1,000 |
| Library | HuggingFace `datasets` |

*Table 7.4 — TriviaQA dataset configuration.*

The `rc` configuration provides well-curated reference answers with comprehensive alias lists, essential for robust correctness evaluation. TriviaQA was selected because its short factual questions produce unambiguous ground truth — a prerequisite for evaluating hallucination detection accuracy.

**Available fields in TriviaQA:** Each record contains `question`, `question_id`, `question_source`, `entity_pages` (Wikipedia context), `search_results` (web search context), and `answer` (with sub-fields: `value`, `aliases`, `normalized_value`, `normalized_aliases`, `matched_wiki_entity_name`, `type`).

**Fields actually used:** Of these, only three are utilised:

| Field | Purpose |
|---|---|
| `question` | Prompt input to the LLM |
| `question_id` | Unique record identifier |
| `answer.aliases` | Reference answer variants for normalised substring correctness matching |

*Table 7.5 — TriviaQA fields used in dataset generation.*

Crucially, no context documents are provided to the LLM — it answers purely from its parametric knowledge. This is intentional: the goal is to measure the model's internal confidence about its own knowledge, not its ability to extract answers from given passages.

### 7.3.3.2 Per-Question Record Generation

For each of the 1,000 TriviaQA questions, the following pipeline produces a single training record:

1. **Multi-sample generation:** Generate 5 independent responses using stochastic sampling (temperature = 0.7, max new tokens = 512), recording per-token logits and softmax probabilities for each sample.
2. **Semantic clustering:** Compare all sample pairs via LLM-based equivalence checking (greedy agglomerative clustering), producing cluster assignments.
3. **Energy teacher computation:** Apply Boltzmann energy aggregation (Formulas 2–3) and normalise across clusters to obtain `energy_score_raw` ∈ [0, 1].
4. **Entropy teacher computation:** Compute Shannon entropy (Formula 1) over cluster size distribution to obtain `entropy_score_raw` ∈ [0, ln 5].
5. **Correctness evaluation:** Check the main answer against all TriviaQA aliases using normalised substring matching, producing a binary correctness label. This label is used only for evaluation, **not** for training.
6. **Hidden state extraction:** Run a **separate** forward pass on the concatenation of prompt and main answer with `output_hidden_states=True`, extracting hidden states at:
   - The **TBG position** (last prompt token): shape (33, 4096)
   - The **SLT position** (second-to-last generated token): shape (33, 4096)
7. **Logit feature extraction:** Compute 9 scalar summary statistics from the main answer's per-token logits (mean, min, and standard deviation of chosen-token logits; mean, min, and standard deviation of top-1/top-2 margins; answer length; mean and min softmax probability).

> **Note on hidden state extraction:** Hidden states are captured via a separate forward pass rather than during generation because storing hidden states during autoregressive generation would require 512 steps × 33 layers × 4096 dimensions ≈ 1.3 GB per sample — infeasible within GPU memory constraints.

> **Figure 7.2** (Flowchart): Per-question record generation pipeline showing the seven processing steps, which model calls occur at each step, and the data flowing between steps.

The complete record schema is presented below.

| Field | Type / Shape | Description |
|---|---|---|
| `uid` | str | TriviaQA question ID |
| `question` | str | Question text |
| `main_answer` | str | First generated answer (sample 0) |
| `energy_score_raw` | float [0, 1] | Energy teacher score (confidence) |
| `entropy_score_raw` | float [0, ~1.61] | Entropy teacher score (uncertainty) |
| `correctness` | float {0.0, 1.0} | Normalised substring match (evaluation only) |
| `emb_last_tok_before_gen` | ndarray (33, 4096) | TBG hidden states — all 33 layers |
| `emb_tok_before_eos` | ndarray (33, 4096) | SLT hidden states — all 33 layers |
| `logit_feats` | dict (9 keys) | Token-level logit summary statistics |
| `num_clusters` | int | Number of semantic clusters |
| `cluster_sizes` | list[int] | Size of each cluster |

*Table 7.6 — Complete record schema in the generated probe training dataset.*

After filtering records where the generated answer was too short for SLT extraction (fewer than 2 tokens), the final dataset comprises **500 valid records**, saved as a serialised Python pickle file (`probe_dataset_llama3-8b_triviaqa.pkl`, approximately 541 MB).

| Split | Records | Percentage | Purpose |
|---|---|---|---|
| Training | 400 | 80% | Binarisation thresholds, probe training |
| Validation | 50 | 10% | Layer sweep AUROC, range selection |
| Test | 50 | 10% | Final evaluation (held out) |

*Table 7.7 — Dataset split allocation (seed = 42 for reproducibility).*

---

## 7.3.4 Probe Training

### 7.3.4.1 Label Binarisation

The raw teacher scores (energy and entropy) are continuous values. Since the probes are logistic regression classifiers, binary labels are required. The binarisation threshold is selected by sweeping from the 10th to the 90th percentile of the training distribution and choosing the value that minimises within-group mean squared error — the same approach used in Semantic Entropy Probes (Kossen et al., 2024).

**Formula 4 — Binarisation:**

$$y_i = \begin{cases} 1 & \text{if } s_i \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

where *s*<sub>*i*</sub> is the raw teacher score and *τ* is the selected threshold. Thresholds are computed on the training split only and applied uniformly to validation and test splits.

| Teacher Signal | Threshold (*τ*) | Label = 1 Interpretation |
|---|---|---|
| Energy | 0.7504 | High confidence (energy ≥ threshold) |
| Entropy | 0.2052 | High uncertainty (entropy ≥ threshold) |

*Table 7.8 — Binarisation thresholds for Llama 3.1 8B on TriviaQA.*

### 7.3.4.2 Per-Layer AUROC Sweep

To identify which transformer layers contain the strongest predictive signal, a per-layer sweep is conducted. For each of the 33 layers (32 transformer layers + 1 embedding layer), a logistic regression classifier (max iterations = 1000, regularisation *C* = 1.0) is trained on the single-layer hidden state vector (4,096 features) with StandardScaler preprocessing. Validation AUROC is recorded for each layer.

This sweep is performed independently for all four probe types, revealing that different probes peak at different layers.

| Probe | Best Single Layer | Validation AUROC |
|---|---|---|
| TBG Energy | Layer 14 | **0.8706** |
| TBG Entropy | Layer 24 | 0.8693 |
| SLT Entropy | Layer 20 | 0.7929 |
| SLT Energy | Layer 19 | 0.7412 |

*Table 7.9 — Best single-layer AUROC results from the per-layer sweep (Llama 3.1 8B).*

> **Figure 7.3** (Line Chart): Per-layer AUROC sweep curves for all four probes across layers 0–32. TBG Energy peaks sharply at layer 14 (middle network), while TBG Entropy peaks at layer 24 (deeper). SLT probes show broader peaks in the upper-middle layers (19–20). Reference: `notebooks/figures/energy_layer_sweep_qwen3-8b.png`, `notebooks/figures/entropy_layer_sweep_qwen3-8b.png`.

### 7.3.4.3 Layer Range Selection and Hidden State Semantics

Rather than relying on a single peak layer, a contiguous window of layers is selected to improve robustness. Window sizes of 4, 8, and 16 layers are evaluated; the window with the highest mean validation AUROC is chosen.

| Probe | Layer Range | Window Size | Mean Validation AUROC |
|---|---|---|---|
| Energy TBG | (28, 32) | 4 | 0.7840 |
| Entropy TBG | (21, 25) | 4 | 0.8523 |
| Entropy SLT | (20, 24) | 4 | 0.7725 |
| Energy SLT | (17, 21) | 4 | 0.7270 |

*Table 7.10 — Selected layer ranges for the final probes (Llama 3.1 8B).*

**Why different probes peak at different layers.** The divergence in optimal layers reflects what different transformer layers encode:

- **Early layers (0–8)** encode surface-level patterns: token identity, syntax, and basic word relationships. These layers recognise that a question is about geography but do not yet encode deep factual knowledge.
- **Middle layers (9–20)** encode semantic meaning and factual associations. This is where the model "retrieves" relevant knowledge from its training data. TBG Energy peaks here (layer 14) because factual confidence is most active in middle-layer representations.
- **Late layers (21–32)** encode output preparation: the model is resolving ambiguities, finalising token selection, and preparing the output distribution. TBG Entropy peaks here (layer 24) because uncertainty manifests in the output-preparation layers.

SLT probes operate on post-generation hidden states, which contain additional context from the generated answer. Their optimal ranges (layers 17–24) sit in the upper-middle region, where the model's accumulated generation state intersects with its confidence representations.

### 7.3.4.4 Final Probe Training and Evaluation

The final probes are trained using logistic regression on the selected layer ranges. The feature vector for each record is formed by concatenating hidden state vectors across the selected layers and flattening: `hidden[l₀:l₁, :].reshape(1, -1)`, producing a vector of (window_size × 4,096) features. StandardScaler normalisation is applied before training.

Evaluation on the held-out test set with 1,000-sample bootstrap 95% confidence intervals:

| Probe | Test AUROC | 95% CI |
|---|---|---|
| SLT Entropy | **0.7877** | [0.6547, 0.8949] |
| TBG Entropy | 0.7857 | [0.6389, 0.8988] |
| TBG Energy | 0.7480 | [0.5151, 0.9377] |
| SLT Energy | 0.6667 | [0.4742, 0.8370] |

*Table 7.11 — Final probe test AUROC with bootstrap confidence intervals (Llama 3.1 8B).*

**Feature ablation** was conducted to evaluate whether logit summary features (mean/min/std of chosen-token logit, answer length) improve performance when combined with hidden states. The results confirm that hidden states alone are sufficient — adding logit features does not improve test AUROC. This is notable because logit features alone achieve high validation AUROC (~0.92), but this does not transfer to the test set, suggesting overfitting to the validation distribution.

---

## 7.3.5 Teacher Fidelity and Cross-Signal Analysis

### 7.3.5.1 Teacher Fidelity

Teacher fidelity measures how well each probe's continuous output reproduces the ranking of its corresponding teacher signal, assessed via Spearman rank correlation on the test set.

| Probe | Spearman ρ | *p*-value |
|---|---|---|
| SLT Entropy | **0.4330** | 1.68 × 10⁻³ |
| TBG Entropy | 0.4329 | 1.69 × 10⁻³ |
| TBG Energy | 0.2616 | 6.64 × 10⁻² |
| SLT Energy | 0.2507 | 7.91 × 10⁻² |

*Table 7.12 — Teacher fidelity (Spearman ρ) between probe outputs and raw teacher scores.*

Entropy probes exhibit higher fidelity (ρ ≈ 0.43) than energy probes (ρ ≈ 0.25). This is expected: entropy depends only on cluster sizes (a simpler structural signal), while energy additionally depends on per-token logit magnitudes within each cluster — a more complex pattern that a linear probe captures less faithfully.

Crucially, low fidelity does not imply poor hallucination detection. The probes are trained on binarised labels (confident vs. uncertain), not on reproducing exact continuous rankings. A probe can correctly classify most questions as confident or uncertain (high AUROC) while imperfectly ordering them within each class (modest ρ).

### 7.3.5.2 Hallucination Detection Performance

The practical measure of probe quality is how well they detect actual hallucinations, evaluated using model correctness as the ground truth.

| System | Hallucination AUROC | Cost |
|---|---|---|
| Full Energy Teacher (upper bound) | 0.7103 | 60–120s |
| Full Entropy Teacher (upper bound) | 0.7143 | 60–120s |
| **SLT Energy Probe** | **0.7163** | 5–15s |
| TBG Entropy Probe | 0.6806 | 0.5–2s |
| SLT Entropy Probe | 0.6726 | 5–15s |
| TBG Energy Probe | 0.6409 | 0.5–2s |
| Logit Features Only | 0.8075 | <1ms |

*Table 7.13 — Hallucination detection AUROC on the test set (Llama 3.1 8B, TriviaQA).*

A notable finding is that the SLT Energy probe (AUROC 0.7163) slightly exceeds the energy teacher upper bound (0.7103). This may be attributed to a regularisation benefit from binarisation — the probe learns a cleaner decision boundary by ignoring fine-grained score differences that are noise-dominated.

### 7.3.5.3 Cross-Signal Correlation

At the teacher level, energy and entropy are near-perfectly anti-correlated:

$$\rho(\text{energy}_{\text{raw}},\ \text{entropy}_{\text{raw}}) = -0.9958 \quad (p = 1.48 \times 10^{-51})$$

This is expected: both are computed from the same semantic clustering — they are different mathematical summaries of the same underlying phenomenon. The remaining ~0.4% of unexplained variance corresponds to the logit-magnitude information that energy captures and entropy does not.

At the probe level, the anti-correlation is substantially weaker:

$$\rho(\text{energy}_{\text{probe}},\ \text{entropy}_{\text{probe}}) = -0.7444$$

This divergence is significant: it demonstrates that the two probes extract partially independent information from hidden states, which justifies combining both signals in the per-sentence scoring pipeline (Section 7.3.7.2).

---

## 7.3.6 Sentence-Level Decomposition, Claim Filtering, and Risk Scoring

Before a hallucination risk score can be assigned at the sentence level, three sequential steps are applied to each generated answer: segmentation, claim classification, and per-sentence scoring.

### 7.3.6.1 Segmentation and Token Alignment

Sentence boundaries are detected using `pysbd` (Pragmatic Sentence Boundary Disambiguation), a rules-based library requiring no model loading or GPU resources. It handles abbreviations, decimals, and numbered lists without false splits, and is initialised with `language='en'` and `clean=False`.

Each token is then aligned to its parent sentence using a character-midpoint method: the tokeniser's `return_offsets_mapping=True` provides a character span per token, and the token is assigned to whichever sentence span contains its midpoint.

### 7.3.6.2 Claim Filtering

Not every sentence warrants hallucination scoring. The `ClaimFilter` class applies a sequential pipeline of compiled regex patterns to exclude non-factual sentences from downstream scoring. Excluded sentences receive `confidence = None` and are omitted from all aggregation. The filter is conservative by default — any sentence that matches no pattern is treated as a claim.

| Category | Pattern Trigger | Example |
|---|---|---|
| Filler phrases | Exact set match (11 phrases) | "Of course", "Sure" |
| Transitional/meta | `^(here are\|let me\|in summary...)` | "Here are the key points:" |
| Headings | ≤60 chars ending with `:` | "Overview:" |
| Questions | Question word + `?` | "What do you think?" |
| Hedging/opinion | `^(i think\|perhaps\|maybe...)` | "I believe this is correct" |
| Advisory | `^(please note\|keep in mind...)` | "Please note that..." |
| Greeting/sign-off | `^(hi\|hello\|hope this helps...)` | "Hope this helps!" |
| Markdown headers | `^\*\*[^*]+\*\*:?` | "**Overview:**" |
| Short non-numeric | < 3 words, no digits | "Okay." |

*Table 7.14 — ClaimFilter non-claim categories with example triggers.*

### 7.3.6.3 Per-Sentence Scoring

Each retained claim sentence is scored via two parallel mechanisms.

**B1 Logit Confidence** maps the mean chosen-token logit for the sentence to a [0, 1] confidence value via a calibrated sigmoid (Formula 5). The centre *C* = 33.0 and scale *S* = 3.0 are calibrated to the Llama 3.1 8B 4-bit quantised logit distribution (~25–45 range). An optional margin boost of up to +0.10 is applied when the mean top-1/top-2 logit gap exceeds 3.0.

$$\text{confidence}(s) = \frac{1}{1 + \exp\!\left(-\dfrac{\overline{\ell}(s) - C}{S}\right)} + \min\!\left(0.1,\ (\overline{\Delta}(s) - 3.0) \cdot 0.02\right) \cdot \mathbf{1}[\overline{\Delta}(s) > 3.0] \tag{5}$$

**Dual-Probe Scoring** (SLT mode only) extracts the hidden state at the last token of the sentence and runs both SLT probes. The energy probe outputs P(confident), inverted to risk; the entropy probe outputs P(uncertain) directly as risk. The two are combined with AUROC-derived weights (Formula 6):

$$\text{probe\_risk}(s) = 0.515 \cdot \text{entropy\_risk}(s)\ +\ 0.485 \cdot \text{energy\_risk}(s) \tag{6}$$

Weights reflect validation AUROC: entropy (0.773) vs energy (0.727). The probe risk can only escalate the B1 level — probe\_risk ≥ 0.65 → "low" confidence; ≥ 0.35 → "medium"; otherwise the B1 level is retained unchanged.

---

## 7.3.7 Aggregate Hallucination Score

The per-sentence scores and overall SLT probe outputs are combined into a single `combined_risk` value using a token-length conditional strategy. The SLT hidden state (at the second-to-last generated token) is most informative for short answers; for longer answers, per-sentence probes provide better coverage.

**Short answers (≤ 100 tokens):** The overall SLT probe average is blended 50-50 with the mean per-sentence risk when two or more claims are present; otherwise the SLT score alone is used.

$$\text{combined\_risk} = \begin{cases} 0.5 \cdot \dfrac{\text{E\_risk} + \text{H\_risk}}{2} + 0.5 \cdot \overline{R}_{\text{sent}} & n_{\text{claims}} \geq 2 \\[6pt] \dfrac{\text{E\_risk} + \text{H\_risk}}{2} & \text{otherwise} \end{cases} \tag{7}$$

**Long answers (> 100 tokens):** The SLT entropy component contributes a fixed anchor weight of 0.15; the highest-risk sentence gets a logarithmically decaying weight; and the mean sentence risk takes the remainder.

$$\text{combined\_risk} = 0.15 \cdot \text{H\_risk}\ +\ \underbrace{\frac{0.25}{1+\ln n}}_{w_{\max}} \cdot \max R_{\text{sent}}\ +\ (0.85 - w_{\max}) \cdot \overline{R}_{\text{sent}} \tag{8}$$

The max-sentence weight *w*<sub>max</sub> shrinks from 0.25 at *n* = 1 to 0.063 at *n* = 20, preventing a single high-risk outlier sentence from dominating when many claims are present.

| Answer Type | SLT Weight | Per-Sentence (mean) | Max-Sentence |
|---|---|---|---|
| Short, 0–1 claims | 1.0 | — | — |
| Short, 2+ claims | 0.5 | 0.5 | — |
| Long (> 100 tokens) | 0.15 (entropy only) | 0.85 − *w*<sub>max</sub> | 0.25 / (1 + ln *n*) |

*Table 7.15 — Aggregation strategy by answer type.*

The final `combined_risk` is mapped to a confidence level for display: below 0.35 → **High** confidence; below 0.65 → **Medium**; 0.65 and above → **Low** (highest hallucination risk).

---

## 7.3.9 System Architecture and Module Design

This section covers the system's structure, code design, and module integration. Table 7.16 first delineates which components are adopted from prior work and which are original contributions of this project.

| Component | Origin | Notes |
|---|---|---|
| Semantic Entropy framework | Kuhn et al. (2023) | Multi-sample generation, semantic clustering, Shannon entropy formula |
| Semantic Energy teacher | Prior work | Boltzmann energy per sample, cluster energy normalisation |
| `pysbd` sentence segmentation | Open-source library | Adopted unchanged; `language='en'`, `clean=False` |
| `sklearn` LogisticRegression | Open-source library | Used as probe classifier; hyperparameters tuned in this work |
| **Probe architecture (4-probe matrix)** | **This work** | TBG/SLT position rationale, 2 × 2 teacher × position design, layer range selection |
| **Hidden state extraction pipeline** | **This work** | Separate forward pass design, tensor shape handling, sentence-end token positions |
| **Label binarisation scheme** | **This work** | MSE-minimising threshold sweep on training distribution |
| **`ClaimFilter`** | **This work** | Sequential regex pipeline, 9 non-claim pattern categories |
| **Conditional risk aggregation** | **This work** | Token-length conditional strategy, logarithmic max-sentence decay (Formula 8) |
| **FastAPI scoring API** | **This work** | Three-endpoint architecture (Full SE / Fast SLT / Fast TBG) |
| **Frontend SPA** | **This work** | Vanilla JS chat interface with per-sentence risk bars and SVG score history |

*Table 7.16 — Contribution scope: adopted components (plain rows) and original contributions (bold rows).*

### 7.3.9.1 Backend Architecture

The backend follows a modular design with clear separation of concerns.

| Module | File | Responsibility |
|---|---|---|
| API Layer | `backend/app.py` (297 lines) | FastAPI application, endpoint routing, model lifecycle management, probe bundle loading |
| Semantic Engine | `backend/engine.py` (671 lines) | LLM loading (8-bit quantised), generation, semantic clustering, hidden state extraction, probe scoring, sentence-level scoring, risk aggregation |
| Claim Filter | `backend/claim_filter.py` (74 lines) | Regex-based sentence classification (claim vs non-claim), zero external dependencies |
| Probe Bundle | `backend/models/*.pkl` | Serialised bundle containing 4 LogisticRegression probes, 4 StandardScaler preprocessors, and 4 layer range tuples |

*Table 7.17 — Backend module responsibilities.*

The `SemanticEngine` class is the core computational module, responsible for:

- **Model management:** Loading Llama 3.1 8B or Qwen3 8B with BitsAndBytes 8-bit quantisation, validating CUDA availability.
- **Response generation:** Multi-sample stochastic generation with per-token logit and probability capture.
- **Semantic analysis:** Pairwise equivalence checking via LLM greedy decoding, greedy agglomerative clustering.
- **Hidden state extraction:** Forward pass with `output_hidden_states=True`, layer slicing to selected windows.
- **Probe inference:** StandardScaler normalisation, logistic regression `predict_proba`, risk conversion.
- **Sentence processing:** pysbd segmentation, token-to-sentence alignment, claim filtering, B1 logit confidence mapping, conditional aggregation.

### 7.3.9.2 SemanticEngine Code Structure

The `SemanticEngine` class in `backend/engine.py` is instantiated once at application startup and held as a module-level variable in `app.py` — a **Singleton pattern** that keeps the LLM (~8 GB VRAM) resident across requests, eliminating the 30–60 second reload cost on every query.

| Method | Lines | Purpose |
|---|---|---|
| `__init__` | 51–93 | CUDA validation, BitsAndBytes config, model/tokeniser loading, pysbd and ClaimFilter init, probe bundle loading |
| `generate_responses` | ~94–160 | Stochastic multi-sample generation; returns per-token logits and softmax probabilities |
| `check_equivalence` | ~161–220 | LLM-based pairwise semantic equivalence; greedy decoding |
| `cluster_responses` | ~221–270 | Greedy agglomerative clustering using pairwise equivalence matrix |
| `score_sentences` | ~271–350 | pysbd segmentation, claim filtering, B1 logit confidence per sentence |
| `_extract_hidden_states` | 351–401 | Separate forward pass; TBG/SLT hidden state extraction; sentence-end token positions |
| `score_with_tbg_probe` | 403–459 | TBG probe inference: layer slice → StandardScaler → predict_proba → risk mapping |
| `score_with_slt_probe` | 461–540+ | SLT probe inference: generation + hidden extraction + per-sentence scoring + aggregation |
| `run_full_pipeline` | ~541–671 | Full Semantic Energy: 5 generations, clustering, energy/entropy computation |

*Table 7.18 — `SemanticEngine` method map with approximate line references (`backend/engine.py`).*

**Singleton initialisation — CUDA check and 8-bit model loading:**

```python
# backend/engine.py, lines 51–72
class SemanticEngine:
    def __init__(self, model_name: str, probe_bundle_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("SemanticEngine requires a CUDA-capable GPU.")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.tokenizer    = AutoTokenizer.from_pretrained(model_name)
        self.segmenter    = pysbd.Segmenter(language="en", clean=False)
        self.claim_filter = ClaimFilter()

        with open(probe_bundle_path, "rb") as f:
            self.probe_bundle = pickle.load(f)
```

The BitsAndBytes 8-bit configuration reduces Llama 3.1 8B from approximately 16 GB (FP16) to approximately 8 GB VRAM, enabling operation on a consumer RTX 3060 12 GB. The probe bundle is loaded here alongside the model so that model and probes are always co-loaded atomically.

**Core energy computation — Boltzmann teacher:**

```python
# backend/engine.py, lines 39–48
def cal_boltzmann_logits(logits_list):
    return [-mean(sublist) for sublist in logits_list]

def cal_flow(probs_list, logits_list, clusters, fermi_mu=None):
    probs = cal_probs(probs_list)
    if fermi_mu is not None:
        logits = cal_fermi_dirac_logits(logits_list, mu=fermi_mu)
    else:
        logits = cal_boltzmann_logits(logits_list)
    return cal_cluster_ce(probs, logits, clusters)
```

`cal_boltzmann_logits` implements Formula 2: the negative mean of the chosen-token logits gives the per-sample energy. `cal_flow` is the central teacher computation: it assembles per-sample probabilities, selects the energy variant (Boltzmann or Fermi-Dirac), and delegates cluster-level aggregation and normalisation to `cal_cluster_ce` (Formula 3).

**Hidden state extraction — separate forward pass design:**

```python
# backend/engine.py, lines 378–401
with torch.no_grad():
    outputs = self.model(**full_inputs, output_hidden_states=True)

hidden = torch.stack(outputs.hidden_states, dim=0)  # (num_layers, 1, seq_len, hidden_dim)
hidden = hidden[:, 0, :, :].float().cpu()            # (num_layers, seq_len, hidden_dim)

tbg_hidden = hidden[:, prompt_len - 1, :].numpy()   # (num_layers, hidden_dim)
slt_hidden = hidden[:, full_len - 2, :].numpy()     # (num_layers, hidden_dim)
```

`torch.stack` converts the tuple of per-layer tensors from HuggingFace into a single `(layers, batch, seq_len, hidden_dim)` tensor. The batch dimension is dropped since inference is always single-sequence. Index `full_len - 2` targets the second-to-last token because the final position is always the EOS token — the token immediately before EOS carries the accumulated generation-state representation used by SLT probes. Both arrays are moved to CPU before probe inference to free VRAM.

**Probe inference — layer slice and logistic regression:**

```python
# backend/engine.py, lines 431–451
l0, l1 = probe_bundle["best_energy_tbg_range"]
X_tbg = tbg_hidden[l0:l1, :].reshape(1, -1)          # (1, window_size × 4096)
X_e   = probe_bundle["tbg_energy_scaler"].transform(X_tbg)
energy_confidence = probe_bundle["tbg_energy_probe"].predict_proba(X_e)[0, 1]
energy_risk = 1.0 - energy_confidence                 # invert: probe outputs P(high-energy/confident)

l0h, l1h = probe_bundle["best_entropy_tbg_range"]
X_tbg_h  = tbg_hidden[l0h:l1h, :].reshape(1, -1)
X_h      = probe_bundle["tbg_entropy_scaler"].transform(X_tbg_h)
entropy_risk = probe_bundle["tbg_entropy_probe"].predict_proba(X_h)[0, 1]  # P(uncertain) = risk directly

combined_risk = (energy_risk + entropy_risk) / 2.0
```

The layer range tuples stored in the probe bundle at training time ensure that inference always slices the exact layers used during training. `reshape(1, -1)` flattens the `(window_size, 4096)` slice into the vector expected by `predict_proba`. The energy probe outputs P(confident) and is inverted to risk; the entropy probe outputs P(uncertain) and is used as risk directly.

### 7.3.9.3 API Endpoints

Three scoring endpoints implement the three speed-accuracy tradeoff points:

| Endpoint | Method | Scoring Mode | Model Calls | Response Time |
|---|---|---|---|---|
| `/chat` | POST | Full Semantic Energy | 5 generations + up to 10 equivalence checks | 60–120s |
| `/score_fast_slt` | POST | Fast SLT (probe-based) | 1 generation + 1 forward pass | 5–15s |
| `/score_fast_tbg` | POST | Fast TBG (probe-based) | 1 forward pass (no generation) | 0.5–2s |
| `/status` | GET | Health check | None | < 1ms |
| `/switch_model` | POST | Model swap | Full model reload | 30–60s |

*Table 7.19 — API endpoints and their characteristics.*

> **Figure 7.4** (Architecture Diagram): System architecture showing three tiers — Frontend (Vanilla JS), Backend (FastAPI with SemanticEngine, ClaimFilter, and ProbeBundle), and ML Layer (LLM with Hidden State Extractor). Arrows indicate data flow between components.

> **Figure 7.5** (Sequence Diagram): Side-by-side comparison of the three scoring modes, showing which processing steps each mode executes. Full SE performs all steps; Fast SLT skips clustering; Fast TBG skips both generation and clustering.

### 7.3.9.4 Training Pipeline

The training infrastructure consists of four Jupyter notebooks forming a sequential pipeline:

| Notebook | Purpose | Output |
|---|---|---|
| `00_preflight.ipynb` | Formula verification, score orientation checks | Validated mathematical foundations |
| `01_generate_dataset.ipynb` | TriviaQA processing, 5-sample generation, teacher computation | `probe_dataset_llama3-8b_triviaqa.pkl` (~541 MB) |
| `02_train_se_probes.ipynb` | Binarisation, layer sweep, probe training, evaluation | `probes_llama3-8b_triviaqa.pkl` (~2.1 MB) |
| `04_sentence_baseline.ipynb` | B1 per-sentence logit confidence validation | Calibrated sigmoid parameters |

*Table 7.20 — Training pipeline notebooks and their outputs.*

### 7.3.9.5 Frontend Architecture

The frontend is implemented as a vanilla JavaScript single-page application with no framework dependencies. Key components include:

- **Three-mode selector** (Full SE / Fast SLT / Fast TBG) persisted in `localStorage`, allowing users to select their preferred speed-accuracy tradeoff.
- **Per-message rendering** with a colour-coded confidence badge (green/yellow/red), response timer, and collapsible metrics panel showing per-sentence analysis with risk bars.
- **Score history chart** (SVG-based) visualising confidence trends across the conversation, with colour-coded zones for high/medium/low confidence.
- **Model switching** via dropdown, with a loading overlay during model swap.

**State management:** `localStorage` stores persistent preferences (mode, model selection); `sessionStorage` stores ephemeral state (chat message HTML, score history).

### 7.3.9.6 Module Integration

**Method-level call sequence — `/score_fast_slt`:**

The Fast SLT endpoint illustrates the full integration path, combining generation, sentence scoring, hidden state extraction, probe inference, and risk aggregation in a single request.

```
POST /score_fast_slt  (app.py)
  │
  ├─ Readiness check: engine not None, probe_bundle loaded
  │
  └─ engine.score_with_slt_probe(prompt, max_new_tokens=512)
       │
       ├─ 1. generate_responses(n=1, temperature=0.7)
       │       → answer_text, logits_per_token, probs_per_token
       │
       ├─ 2. score_sentences(answer_text, logits_per_token)
       │       ├─ pysbd.Segmenter.segment()         → sentence spans
       │       ├─ ClaimFilter.is_claim()             → bool per sentence
       │       └─ B1 sigmoid(mean_logit, margin)     → confidence per claim
       │
       ├─ 3. _align_tokens_to_sentences(tokens, sentence_spans)
       │       → token → sentence index (character midpoint method)
       │
       ├─ 4. _extract_hidden_states(prompt_ids, answer_ids)
       │       ├─ model(**full_inputs, output_hidden_states=True)
       │       ├─ stack → (33, seq_len, 4096) → move to CPU
       │       ├─ tbg_hidden = [:, prompt_len−1, :]    (33, 4096)
       │       ├─ slt_hidden = [:, full_len−2, :]      (33, 4096)
       │       └─ sentence_end_positions               per-sentence last-token indices
       │
       ├─ 5. Overall SLT probe inference
       │       ├─ slt_hidden[l0:l1].reshape(1,−1) → scaler → slt_energy_probe
       │       │       energy_risk = 1 − predict_proba[:,1]
       │       └─ slt_hidden[l0h:l1h].reshape(1,−1) → scaler → slt_entropy_probe
       │               entropy_risk = predict_proba[:,1]
       │
       ├─ 6. Per-sentence probe scoring
       │       for each claim sentence s:
       │           hidden_at_sent_end = hidden[:, sent_end_idx, :]
       │           probe_risk(s) = 0.515 × entropy_risk + 0.485 × energy_risk
       │           level(s)      = max(B1_level, probe_risk_level)   ← risk only escalates
       │
       └─ 7. Conditional aggregation (Formula 7 or 8 based on token_len)
               → combined_risk, confidence_level
               → JSON {answer, confidence_level, combined_risk, sentence_scores, …}
```

**Integration design decisions:**

- **Singleton engine:** `app.py` holds the engine as a module-level variable initialised at startup. Every request handler accesses the same instance — the correct pattern for a single-GPU server where model loading is the dominant cost and concurrency is not a requirement.
- **Atomic model–probe binding:** When `/switch_model` is called, the engine is deleted (`del engine`), CUDA cache cleared, and a new engine instantiated with the model-specific probe bundle path. This enforces probe–model consistency structurally: it is not possible to run a Qwen3 probe against a Llama hidden state by accident, because the bundle is always co-loaded with its corresponding model.
- **Claim filter decoupling:** `ClaimFilter` holds only compiled `re` patterns and has no GPU dependencies. It is instantiated inside `SemanticEngine.__init__` but operates purely on strings, ensuring sentence classification never blocks on GPU operations.
- **TBG as a pre-generation gate:** `/score_fast_tbg` runs a single forward pass on the prompt with no generation, returning a risk estimate before the model produces any answer. This enables a qualitatively different interaction mode: the frontend can display a confidence warning to the user *before* potentially hallucinated text is shown.

---

## 7.3.10 Functional Requirement Traceability

The following table maps each functional requirement to its implementing module(s).

| ID | Requirement | Implementation | Module(s) |
|---|---|---|---|
| FR1 | Analyse Answer (Local) | All inference runs on local GPU via 8-bit quantised LLM; no external API calls | `engine.py` (`SemanticEngine.__init__`) |
| FR2 | Compute Paragraph Risk | `combined_risk` and `confidence_level` computed in all three scoring modes | `engine.py` (`score_with_slt_probe`, `score_with_tbg_probe`), `app.py` (`/chat`) |
| FR3 | Factoid/Sentence Decomposition | pysbd sentence segmentation with character-midpoint token alignment | `engine.py` (`score_sentences`, `align_tokens_to_sentences`) |
| FR4 | Per-Factoid Risk Labels | Per-sentence `probe_risk`, `level`, and `is_claim` fields in SLT and Full SE modes | `engine.py` (`score_sentences`, `score_with_slt_probe`) |
| FR5 | Single-Pass Probe Estimator | TBG mode: 1 forward pass; SLT mode: 1 generation + 1 forward pass | `engine.py` (`score_with_tbg_probe`, `score_with_slt_probe`) |
| FR6 | Risk Summary View | Colour-coded confidence badge + overall score displayed per message | `frontend/script.js` (response rendering) |
| FR7 | Detailed Rationale | Collapsible metrics panel with per-sentence breakdown showing energy risk, entropy risk, logit confidence, and margin statistics | `frontend/script.js` (metrics panel) |
| FR8 | Threshold & Display Settings | Mode selector (Full SE / SLT / TBG) providing three speed-accuracy tradeoff points | `frontend/script.js` (mode selector), `app.py` (endpoint routing) |
| FR9 | Multi-user Accounts / Auth | Not implemented — out of scope | — |
| FR10 | Public Web/App Hosting | Not implemented — local deployment only | — |
| FR11 | Multilingual Detection | Not implemented — English only | — |

*Table 7.21 — Functional requirement traceability matrix.*

All mandatory requirements (FR1–FR5, priority M) are fully implemented. The "should have" requirement (FR6) and "could have" requirement (FR7) are also implemented. FR8 is partially addressed through the three-mode selection mechanism. FR9–FR11 are explicitly documented as out of scope.
