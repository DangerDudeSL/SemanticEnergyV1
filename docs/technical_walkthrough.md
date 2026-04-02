# SemanticEnergy — Technical Walkthrough

A narrative explanation of the entire framework, from the physics intuition behind Semantic Energy to how linear probes and logit confidence detect hallucinations at the sentence level. Written for presenting to technical experts.

---

## Table of Contents

1. [The Core Idea: What Is Semantic Energy?](#1-the-core-idea-what-is-semantic-energy)
2. [How We Calculate Semantic Energy from Logits](#2-how-we-calculate-semantic-energy-from-logits)
3. [The Cost Problem and Why We Built Probes](#3-the-cost-problem-and-why-we-built-probes)
4. [How the Probes Are Trained](#4-how-the-probes-are-trained)
5. [Logit Confidence: The Per-Token Signal](#5-logit-confidence-the-per-token-signal)
6. [Sentence-Level Hallucination Detection](#6-sentence-level-hallucination-detection)
7. [The Three Inference Modes](#7-the-three-inference-modes)
8. [Evaluation Summary](#8-evaluation-summary)

---

## 1. The Core Idea: What Is Semantic Energy?

The central insight of this framework is borrowed from statistical mechanics. In a physical system, particles at low temperature settle into a single low-energy state — they are ordered and predictable. At high temperature, they scatter across many states — disordered and uncertain.

We apply the same logic to language model outputs. If a model is genuinely confident about an answer, then asking it the same question multiple times (with sampling enabled, so the outputs can vary) will produce answers that all mean the same thing. The model keeps "falling into" the same semantic basin. But if the model is hallucinating or uncertain, different runs produce different answers — the model scatters across multiple meanings.

**Example — high confidence:**

Ask the model "What is the capital of France?" five times:

```
Sample 1: "Paris"
Sample 2: "The capital of France is Paris."
Sample 3: "Paris is the capital."
Sample 4: "Paris"
Sample 5: "It's Paris."
```

All five answers express the same meaning. When we group them, they fall into a single cluster. All of the model's "energy" is concentrated in one place. This is like a cold physical system — ordered and certain.

**Example — hallucination risk:**

Ask "Who invented the telephone?" five times:

```
Sample 1: "Alexander Graham Bell"
Sample 2: "Antonio Meucci"
Sample 3: "Bell invented the telephone in 1876."
Sample 4: "Elisha Gray"
Sample 5: "Alexander Graham Bell"
```

Now we get three distinct answers, forming three clusters: {Bell, Bell, Bell's}, {Meucci}, {Gray}. The energy is dispersed. This is like a hot system — disordered. That dispersion is our signal that the model does not actually "know" the answer and is likely hallucinating.

---

## 2. How We Calculate Semantic Energy from Logits

The full Semantic Energy calculation has three stages. Let us walk through each one with concrete numbers.

### Stage 1: Generate Multiple Responses and Collect Token-Level Data

We generate N responses (default: 5) for a given question. For each response, the model produces tokens one at a time. At each step, we record:

- **The chosen token's logit value** — the raw, unnormalized score the model assigned to the token it selected. These are typically in the range of 20–45 for a confident prediction. For example, if the model generates the token "Paris" and the logit for that token was 38.5, we store 38.5.
- **The chosen token's probability** — obtained by applying softmax to the full logit vector. If "Paris" had a logit of 38.5 and the total softmax distribution gave it a probability of 0.92, we store 0.92.

So after generating one response like "Paris is the capital", we might have:

| Token  | Logit | Probability |
|--------|-------|-------------|
| Paris  | 38.5  | 0.92        |
| is     | 35.2  | 0.85        |
| the    | 37.1  | 0.88        |
| capital| 36.8  | 0.87        |

Each response produces a list of logits and a list of probabilities.

### Stage 2: Semantic Clustering

Next, we determine which responses mean the same thing. We use the model itself as a judge — for each pair of responses, we ask:

> "Is answer A semantically equivalent to answer B?"

The model outputs "Yes" or "No". Using these pairwise comparisons, we build clusters of semantically equivalent answers. If samples 1, 3, and 5 all say "Paris" (in different phrasings), they form one cluster. Sample 2 says "London" — that is a second cluster.

### Stage 3: The Energy Calculation

Now we compute the energy for each cluster. The calculation uses the **Boltzmann formulation** (not Fermi-Dirac, which is an alternative we support but do not use by default):

**Step 3a — Sequence probability for each response:**

Each response's probability is the product of its per-token probabilities. If response 1 had token probabilities [0.92, 0.85, 0.88, 0.87], its sequence probability is:

```
P(response_1) = 0.92 × 0.85 × 0.88 × 0.87 = 0.597
```

**Step 3b — Normalize probabilities across all responses:**

We sum all five sequence probabilities and divide each by the total so they sum to 1.0.

```
P_norm = [0.35, 0.33, 0.15, 0.12, 0.05]
```

**Step 3c — Cluster energy from logits:**

For each response, we compute the negative mean of its per-token logits. This is the "Boltzmann energy" of that response — a response where the model assigned high logits has low energy (= confident), and a response with low logits has high energy (= uncertain).

For a cluster, we sum the negative mean logits of all responses in that cluster:

```
cluster_energy[0] = -(mean_logit_response_1) + -(mean_logit_response_3) + -(mean_logit_response_5)
```

Since all three said "Paris" with high logits, this cluster's energy is large (in absolute terms, strongly negative, but after the negation it is a large positive number, indicating the model poured a lot of energy into this meaning).

**Step 3d — Normalize to get a confidence score:**

We normalize the cluster energies so they sum to 1.0 using `sum_normalize`. The main answer's cluster energy becomes the confidence score.

If cluster 0 ("Paris") gets 85% of the total energy and cluster 1 ("London") gets 15%, then the confidence score is **0.85**. The model is very confident in "Paris".

In code, this entire pipeline is:

```python
probs_se, logits_se = cal_flow(probs_list, logits_list, clusters)
cluster_energies = sum_normalize(logits_se)
main_confidence = cluster_energies[main_cluster_idx]
```

The elegance of this approach is that it captures something softmax probabilities alone cannot: **semantic consistency across multiple generations**. A model can assign high probability to each individual token while still producing different answers each time. Semantic Energy catches this.

---

## 3. The Cost Problem and Why We Built Probes

The full Semantic Energy calculation described above is powerful but expensive. For a single question it requires:

1. **5 generations** — each taking 2-10 seconds depending on answer length
2. **10 pairwise semantic comparisons** — each a separate model call (5 choose 2 = 10 pairs)
3. **One additional forward pass** for hidden state extraction

Total: **60-120 seconds per question** on typical hardware (single GPU with 8B parameter model).

This is acceptable for a research experiment but not for a real-time interface where a user is waiting for a response. We needed a way to get a comparable signal in under 5 seconds.

The idea: **train small linear probes that can predict the Semantic Energy score from internal model representations, without running the full pipeline.** If the model's hidden states already "know" whether the answer will be reliable — before we even run clustering — then a probe should be able to detect that signal.

This turned out to be true. The hidden states at specific token positions encode information about the model's confidence, and a simple LogisticRegression trained on those hidden states achieves AUROC of 0.78 for predicting whether the full Semantic Energy score would classify the answer as hallucinated.

---

## 4. How the Probes Are Trained

The probe training pipeline (notebook 05) has four phases. Here is what happens in each.

### Phase 1: Dataset Generation

We process 1,000 questions from TriviaQA (Reading Comprehension split). For each question:

1. Generate 5 diverse responses (temperature = 0.7)
2. Run the full Semantic Energy pipeline — clustering, energy calculation
3. Compute a **correctness label** by comparing the main answer against TriviaQA's reference answers using normalized string matching
4. Extract two key hidden states from the model
5. Compute the cluster assignment entropy (how spread out the clusters are)

The two hidden states we extract are:

- **TBG (Token Before Generation):** The hidden state at the very last prompt token, right before the model starts generating the answer. This captures what the model "knows" about the question before producing any output. Think of it as the model's internal state of readiness.

- **SLT (Second-to-Last Token):** The hidden state at the second-to-last token of the full sequence (prompt + answer). This is the position right before the EOS token. It captures the model's internal state after it has seen the entire answer — a summary of how the generation went.

Each hidden state is a matrix of shape `(num_layers, hidden_dim)`. For a 33-layer model with 4096-dimensional hidden states, that is 33 × 4096 = 135,168 values per position. But we do not use all layers — we select the best contiguous range (more on this below).

The training record for each question looks like:

```python
{
    'question': "What is the capital of France?",
    'main_answer': "Paris",
    'correctness': 1.0,                        # matched reference answers
    'energy_score_raw': 0.847,                  # from full SE pipeline
    'entropy_score_raw': 0.0,                   # all in one cluster = 0 entropy
    'emb_last_tok_before_gen': tbg_hidden,      # shape (33, 4096)
    'emb_tok_before_eos': slt_hidden,           # shape (33, 4096)
    'num_clusters': 1,
    'cluster_sizes': [5],
}
```

### Phase 2: Label Binarization and Layer Sweeping

The raw energy and entropy scores are continuous (e.g., 0.847, 0.312). To train a binary classifier, we need to convert them into 0/1 labels. We use **MSE-optimal thresholding**: we sweep through percentile thresholds from the 10th to the 90th percentile and find the threshold that minimizes the within-group variance (similar to Otsu's method in image processing). Scores above the threshold become 1 (high energy = confident), scores below become 0 (low energy = hallucination risk).

Now comes the key innovation: **layer range sweeping**. Not all transformer layers are equally informative. Early layers encode surface-level syntax, middle layers encode semantics, and later layers prepare the final token prediction. We need to find which contiguous block of layers carries the strongest signal for predicting hallucination.

We sweep every single layer first, training a separate LogisticRegression probe on each layer's hidden state (one layer × 4096 dimensions = 4096 features). We evaluate each probe's AUROC on a validation set:

```
Layer 0:  AUROC = 0.52  (near random — embedding layer)
Layer 1:  AUROC = 0.54
...
Layer 15: AUROC = 0.72  (getting informative)
Layer 16: AUROC = 0.74
...
Layer 28: AUROC = 0.76  (peak region)
Layer 29: AUROC = 0.75
...
Layer 32: AUROC = 0.71  (slightly drops at final layers)
```

This produces a per-layer AUROC curve. We then find the best contiguous window of 4, 8, or 16 layers that maximizes the mean AUROC. For example, if layers 24–32 (window of 8) have the highest mean, we use that range.

The function:

```python
def decide_layer_range(auroc_list, window_sizes=[4, 8, 16]):
    for window in window_sizes:
        for start in range(num_layers - window + 1):
            mean_auroc = auroc_list[start:start+window].mean()
            # keep the best
```

This is done independently for each of the four probes (Energy-SLT, Energy-TBG, Entropy-SLT, Entropy-TBG), because the optimal layers differ between probe types.

### Phase 3: Final Probe Training

Once we know the best layer ranges, we train the final probes. Each probe is a **scikit-learn LogisticRegression** with StandardScaler preprocessing:

1. Extract hidden states for the chosen layer range → flatten into a single vector
   - For a range of layers 24–32 (8 layers) and hidden_dim 4096: input is 8 × 4096 = 32,768 features
2. Apply StandardScaler (zero mean, unit variance per feature)
3. Fit LogisticRegression (max_iter=1000, C=1.0)

We train four probes in total:

| Probe | Token Position | Teacher Signal | What It Predicts |
|-------|---------------|---------------|-----------------|
| SLT Energy | Second-to-last (after answer) | Semantic Energy score | "Is the energy concentrated in one cluster?" |
| TBG Energy | Last prompt token (before answer) | Semantic Energy score | Same, but from prompt alone |
| SLT Entropy | Second-to-last (after answer) | Cluster entropy | "Are answers spread across many clusters?" |
| TBG Entropy | Last prompt token (before answer) | Cluster entropy | Same, but from prompt alone |

The SLT probes are post-generation (they see the answer) and the TBG probes are pre-generation (they only see the question). SLT probes are more accurate because they have more information, but TBG probes are faster because they do not require generating an answer first.

### Phase 4: Saving the Probe Bundle

All four probes, their scalers, the optimal layer ranges, and evaluation metrics are serialized into a single pickle file:

```python
probe_bundle = {
    'slt_energy_probe':  LogisticRegression,
    'slt_energy_scaler': StandardScaler,
    'best_energy_slt_range': (24, 32),      # example values
    'tbg_energy_probe':  LogisticRegression,
    'tbg_energy_scaler': StandardScaler,
    'best_energy_tbg_range': (20, 28),
    # ... same for entropy probes
}
```

This bundle is loaded at server startup and used for fast inference.

---

## 5. Logit Confidence: The Per-Token Signal

While Semantic Energy and the probes give us answer-level or question-level confidence, we also developed a third signal that works at the **individual token level**, and by extension, at the **sentence level**. This is the logit confidence metric.

### The Origin

During generation, the model produces a logit vector at every step — one value per vocabulary token (typically 32,000+ values). The logit assigned to the token the model actually chose tells us something direct: **how strongly did the model prefer this token over all alternatives?**

The raw logit values for the chosen token in typical 8-bit quantized 8B-parameter models fall in a range of roughly 20 to 45:

- A chosen-token logit of **40+** means the model was very certain — this token dominated the distribution
- A chosen-token logit of **25-35** means moderate confidence — reasonable but with alternatives
- A chosen-token logit of **<25** means low confidence — the model was somewhat guessing

### The Challenge: Turning Logits into Scores

Raw logits are not directly interpretable as probabilities or confidence scores. A logit of 38 does not mean "38% confident" — it is just an unnormalized score. We needed a mapping function that:

1. Produces values between 0 and 1
2. Has a meaningful center point (50% confidence)
3. Is monotonically increasing (higher logit = higher confidence)
4. Is calibrated to the model's actual logit distribution

### The Sigmoid Solution

We use a **parametric sigmoid** with two calibration constants:

```
confidence = 1 / (1 + exp(-(mean_logit - CENTER) / SCALE))
```

Where:
- **CENTER = 33.0** — the logit value that maps to exactly 50% confidence. This was calibrated by analyzing the distribution of chosen-token logits across hundreds of TriviaQA responses. A mean logit of 33 corresponds roughly to the boundary between "the model had a clear preference" and "the model was choosing among several plausible tokens."
- **SCALE = 3.0** — controls the steepness of the sigmoid. A scale of 3.0 means that moving 3 logit points above or below the center shifts confidence by about 24 percentage points (from 50% to ~73% or ~27%).

**Example calculation for one sentence:**

Suppose a sentence "Paris is the capital of France" has four tokens with chosen logits [38.5, 35.2, 37.1, 36.8]:

```
mean_logit = (38.5 + 35.2 + 37.1 + 36.8) / 4 = 36.9

confidence = 1 / (1 + exp(-(36.9 - 33.0) / 3.0))
           = 1 / (1 + exp(-1.3))
           = 1 / (1 + 0.2725)
           = 0.786
```

So this sentence gets a logit confidence of **78.6%**.

### The Margin Boost

We also capture the **top-2 logit margin** — the gap between the model's top choice and its second choice at each token position. If the model assigns logit 38.5 to "Paris" and logit 31.2 to "London", the margin is 7.3. A large margin means the model was decisive.

When the mean margin across a sentence exceeds 3.0, we apply a small boost:

```
margin_boost = min(0.1, (mean_margin - 3.0) × 0.02)
confidence = min(1.0, confidence + margin_boost)
```

This rewards sentences where the model was not just confident on average, but decisively preferred its choices at each step. The boost is capped at 10 percentage points to prevent overconfidence.

### Why This Works

The logit confidence metric succeeds because it measures something fundamentally different from Semantic Energy:

- **Semantic Energy** asks: "If I regenerate the answer many times, do I get the same meaning?" — it is about *consistency across samples*.
- **Logit confidence** asks: "At each token, how strongly did the model prefer its choice?" — it is about *decisiveness within a single generation*.

These are complementary signals. A model can be decisive at each token (high logit confidence) but produce different answers each time (low Semantic Energy). Or it can be indecisive at each token (low logit confidence) but still land on the same answer by chance. Using both gives a more complete picture.

The logit confidence metric is particularly valuable at the sentence level because it is natively per-token — we simply average across the tokens in each sentence. The probes, by contrast, were trained on whole-answer hidden states, making them somewhat out-of-distribution when applied to individual sentences.

---

## 6. Sentence-Level Hallucination Detection

A critical insight is that hallucinations do not always affect an entire answer uniformly. A model might produce three correct sentences followed by one fabricated claim. We need to detect which specific sentences are unreliable.

### The Isolated Forward Pass Approach

The probes were originally trained on hidden states from short (1-2 sentence) TriviaQA answers, extracted at the second-to-last token of the entire sequence. When we try to apply these probes to individual sentences within a longer answer by extracting hidden states at sentence boundary positions from a single shared forward pass, we run into a problem rooted in how transformers work.

**The causal attention problem:** In a transformer with causal (autoregressive) attention, the hidden state at any position contains information from ALL preceding tokens, not just the current sentence. So the hidden state at the end of sentence 3 is contaminated by the content of sentences 1 and 2. This makes it a fundamentally different distribution from what the probes were trained on (short, isolated answers).

**The solution: isolated forward passes.** For each claim sentence in the answer, we run a separate forward pass through the model with just the prompt + that single sentence. This produces a hidden state that matches the probe's training distribution — a short answer to a question, with the hidden state at the second-to-last token.

```
Shared forward pass (out-of-distribution):
  [prompt] + [sentence1] + [sentence2] + [sentence3]
                                          ↑ hidden state here contains info from all 3 sentences

Isolated forward pass (in-distribution):
  [prompt] + [sentence2]
              ↑ hidden state here is clean — only prompt + this sentence
```

Each isolated forward pass takes roughly 50-200ms on a GPU, so for a 5-sentence answer, the additional cost is 0.25-1.0 seconds — well within the acceptable range for a real-time system.

### Three-Way Signal Blending

For each claim sentence, we now have three independent risk signals:

1. **Energy probe risk** — from the isolated forward pass hidden state, scored by the SLT energy probe
2. **Entropy probe risk** — from the same hidden state, scored by the SLT entropy probe
3. **Logit risk** — derived from the per-token logit confidence (inverted: risk = 1 - confidence)

We combine these with calibrated weights:

```
W_ENERGY  = 0.35    (best cross-dataset AUROC: ~0.704 average)
W_ENTROPY = 0.30    (slightly weaker: ~0.683 average)
W_LOGIT   = 0.35    (natively per-token, most reliable at sentence level)
```

The blending is adaptive: if any signal is missing (e.g., a sentence too short for probe extraction), the remaining signals are automatically renormalized:

```python
signals, weights = [], []
if energy_risk is not None:
    signals.append(energy_risk);  weights.append(0.35)
if entropy_risk is not None:
    signals.append(entropy_risk); weights.append(0.30)
if logit_risk is not None:
    signals.append(logit_risk);   weights.append(0.35)

w_total = sum(weights)
probe_risk = sum(s * w / w_total for s, w in zip(signals, weights))
```

### Claim Filtering

Not every sentence in an answer is a factual claim. Sentences like "Here are some examples:" or "I hope this helps!" are transitional or filler text. Scoring these for hallucination would add noise.

We use a lightweight regex-based claim filter that identifies non-claim patterns:

- Meta/transitional: "Here are...", "Let me explain..."
- Questions: "What do you think?"
- Hedging: "I think...", "In my opinion..."
- Greetings: "Hello!", "Hope this helps!"

Only sentences classified as claims are scored. Non-claim sentences are skipped in the hallucination analysis and excluded from the sentence average confidence.

### The Sentence Average Confidence

The final overall confidence score for an answer is the mean of (1 - probe_risk) across all claim sentences. This is the number displayed in the confidence badge:

```
sentence_avg_confidence = mean([1.0 - probe_risk for each claim sentence])
```

If this value is >= 0.65, the answer is labeled **high confidence**. Between 0.35 and 0.65 is **medium**. Below 0.35 is **low**.

---

## 7. The Three Inference Modes

The system offers three modes that trade off between speed and accuracy.

### Mode 1: Full Semantic Energy (Full SE)

This is the complete pipeline as described in Section 2:
- Generate 5 diverse responses
- Cluster them by semantic equivalence
- Calculate cluster energies and entropy
- Also compute per-sentence logit confidence and probe risk

**Time:** ~60-120 seconds per question
**Accuracy:** Highest — uses all available signals including cross-generation consistency

### Mode 2: Fast SLT (Post-Generation)

- Generate 1 response
- Run 1 additional forward pass on (prompt + answer) for hidden state extraction
- Run isolated forward passes per claim sentence for sentence-level scoring
- Score with SLT energy and entropy probes + logit confidence

**Time:** ~3-8 seconds per question
**Accuracy:** Good — approximates the full pipeline using probes trained on full SE labels

### Mode 3: Fast TBG (Pre-Generation)

- Run 1 forward pass on the prompt only (no generation at all)
- Score with TBG energy and entropy probes

**Time:** ~0.5-1 second per question
**Accuracy:** Lowest — predicts risk before seeing the answer, based only on the question. No sentence-level scoring.

This mode is useful for real-time risk estimation: before the model even starts generating, we can warn the user that this question is likely to produce an unreliable answer.

---

## 8. Evaluation Summary

Probes were trained on TriviaQA and evaluated both in-distribution and cross-dataset.

**In-distribution (TriviaQA test set, Qwen3-8B):**

| Probe | AUROC |
|-------|-------|
| SLT Energy | 0.782 |
| SLT Entropy | 0.762 |
| TBG Energy | 0.680 |
| TBG Entropy | 0.671 |

**Cross-dataset generalization:**

| Dataset | Energy AUROC | Entropy AUROC |
|---------|-------------|--------------|
| TriviaQA (in-dist.) | 0.782 | 0.762 |
| NQ-Open | 0.636 | 0.645 |
| SQuAD | 0.694 | 0.642 |

The probes generalize reasonably well to unseen datasets, with some expected degradation. The energy probe tends to generalize better than the entropy probe across datasets.

The full Semantic Energy calculation (the teacher) naturally achieves higher accuracy than the probes (the students), but at 10-20x the computational cost.
