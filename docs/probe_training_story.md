# The Story of Probe Training in SemanticEnergy

This guide walks through, in narrative form, how logistic regression probes are trained to detect hallucinations by reading the internal hidden states of a transformer model. It covers the full lifecycle: what gets fed in, how training works, the role of layer sweeps, and what happens at inference time.

---

## Chapter 1: The Core Idea — Reading the Model's Mind

Imagine you ask someone a trivia question. Before they answer, there are subtle cues — a pause, a shift in eye contact — that hint whether they *actually know* the answer or are about to make something up. Transformer models have an analogous tell: their **hidden states**.

At every layer of a transformer (Llama 3.1 8B has 32 transformer layers + 1 embedding layer = 33 total), the model maintains an internal representation vector of dimension 4096. These vectors evolve as information flows upward through the layers — early layers encode syntactic/sub-word information, middle layers build semantic meaning, and late layers prepare the final output distribution.

The insight behind SemanticEnergy's probes: **a simple logistic regression model can be trained to look at these hidden states and predict whether the transformer is about to hallucinate**. No need to generate multiple answers and compare them (which is slow). Just one peek inside the model's "brain" at the right layer and the right token position.

---

## Chapter 2: Two Observation Points — TBG and SLT

The probes observe hidden states at two specific token positions, each serving a different purpose:

### TBG (Token Before Generation)
- **What it is:** The hidden state at the **last token of the prompt**, right before the model starts generating.
- **Extracted as:** `hidden[:, prompt_len - 1, :]` — shape `(33, 4096)`
- **Analogy:** Reading the model's "readiness state." It's like watching someone's face right after you ask the question but before they speak. Do they look confident or confused?
- **Speed:** Very fast (0.5-2 sec) — only requires a forward pass on the prompt, no generation needed.

### SLT (Second-to-Last Token)
- **What it is:** The hidden state at the **second-to-last token of the full sequence** (prompt + generated answer), just before the EOS token.
- **Extracted as:** `hidden[:, full_len - 2, :]` — shape `(33, 4096)`
- **Analogy:** Reading the model's "reflection state" after it has committed to an answer. Like watching someone's body language right after they finish speaking — do they look certain about what they just said?
- **Speed:** Slower (5-15 sec) — requires generating an answer first, then doing a separate forward pass.

**Why a separate forward pass for extraction?** Because using `model.generate(output_hidden_states=True)` would store hidden states for every decoding step (up to 512 steps x 33 layers x 4096 dims ~ 1.3 GB). Instead, the system generates the answer first (cheaply, without storing hidden states), then runs one forward pass on the concatenated (prompt + answer) to extract hidden states at the positions of interest.

---

## Chapter 3: The Teacher Signals — How Do We Know What's "Right"?

Here's the chicken-and-egg problem: to train a probe, you need labels. But the whole point of the probe is to avoid generating multiple answers. So where do labels come from?

The answer: **during training (offline), we DO generate multiple answers and use semantic clustering to create "teacher signals."** These teacher signals become the ground truth labels for training the probes. At inference time, the trained probes replace this expensive process.

### Teacher Signal 1: Energy Score
*(Defined in [engine.py:19-28](../backend/engine.py#L19-L28), computed in [01_generate_dataset.ipynb](../notebooks/01_generate_dataset.ipynb))*

For each training question, 5 answers are generated and clustered by semantic similarity. The Energy Score measures **how much probability mass concentrates in the dominant cluster**.

The computation:
1. For each of the 5 generated answers, compute the product of token-level probabilities (how confident the model was about each token it chose).
2. Group these by semantic cluster.
3. Normalize across clusters to get a distribution.
4. The Energy Score = the share belonging to the main cluster.

**Range:** 0 to 1. **Higher = more confident** (the model keeps saying the same thing with high probability). Low energy = the model is scattering its probability across multiple different answers.

### Teacher Signal 2: Entropy Score
*(Computed in [01_generate_dataset.ipynb](../notebooks/01_generate_dataset.ipynb) Section 4)*

Shannon entropy over the cluster-size distribution:

```
entropy = -sum(p * log(p)) for each cluster's proportion p
```

Examples with 5 generated answers:
- All 5 agree -> 1 cluster of size 5 -> entropy ~ 0.0 (very certain)
- 3-2 split -> entropy ~ 0.67 (moderate disagreement)
- All 5 disagree -> 5 clusters of size 1 -> entropy ~ 1.61 (maximum confusion)

**Range:** 0 to ln(5) ~ 1.609. **Higher = more uncertain.** This is the inverse relationship from Energy.

### Why two teacher signals?
Energy captures *confidence concentration* — is the model putting its chips on one answer? Entropy captures *diversity of disagreement* — how many distinct answer clusters exist? These are complementary views of hallucination risk. A model might have moderate energy (split between two answers) but low entropy (only two clusters, not five). Using both gives a richer training signal.

---

## Chapter 4: Binarization — Turning Continuous Scores into Labels

*(Done in [02_train_se_probes.ipynb](../notebooks/02_train_se_probes.ipynb) Section 4)*

Logistic regression needs binary labels (0 or 1), but teacher signals are continuous. The solution: find a threshold that **minimizes within-group variance** (like a 1D K-means with K=2).

```python
for pct in range(10, 91):         # sweep 10th to 90th percentile
    thresh = percentile(scores, pct)
    group_0 = scores[scores < thresh]
    group_1 = scores[scores >= thresh]
    mse = weighted_variance(group_0, group_1)
    # keep threshold with lowest MSE
```

For Llama 3.1 8B on TriviaQA (400 training records):
- **Energy threshold:** ~0.7504 -> scores >= 0.7504 get label 1 (high confidence)
- **Entropy threshold:** ~0.2052 -> scores >= 0.2052 get label 1 (high uncertainty)

**Critical design choice:** These labels come purely from the teacher signal distribution. No "correct answer" labels are used. The probes learn to *approximate what multi-sample generation would tell you*, not to directly predict correctness. This is what makes the approach self-supervised — you don't need a ground-truth answer key.

---

## Chapter 5: The Layer Sweep — Finding Where the Signal Lives

*(Done in [02_train_se_probes.ipynb](../notebooks/02_train_se_probes.ipynb) Sections 6-7)*

Not all 33 layers are equally informative. Early layers encode low-level token features; late layers encode task-specific output preparation. The "sweet spot" for hallucination detection is somewhere in between — and it differs for each probe.

### How the Sweep Works

For **each of the 33 layers independently**, a throwaway logistic regression probe is trained:

```python
for layer in range(33):
    X_train = hidden_states[:, layer, :]   # (N_train, 4096) — just one layer
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(scaler.fit_transform(X_train), labels)
    auroc = roc_auc_score(val_labels, probe.predict_proba(X_val)[:, 1])
```

This produces 4 curves of 33 AUROC values each (one curve per combination of token position x teacher signal):

| Probe | Typical Best Layers | AUROC Pattern |
|-------|-------------------|---------------|
| TBG Energy | 28-32 (deep) | Signal concentrates in final layers |
| TBG Entropy | 21-25 (upper-mid) | Peaks slightly earlier |
| SLT Energy | 17-21 (mid) | Semantic region |
| SLT Entropy | 20-24 (upper-mid) | Similar to TBG entropy |

### Why Different Layers for Different Probes?

- **TBG probes** peek at the prompt's last token. At this point, the model hasn't generated anything yet. The "does this model know the answer?" signal appears to concentrate in the **deepest layers** (28-32), where the model has finished its full computation and is most "decided" about what to generate next.

- **SLT probes** look at the second-to-last token after the full answer. Here, the semantic understanding of "was my answer coherent and consistent?" lives in **middle-to-upper layers** (17-24), where semantic representations are richest but haven't yet been compressed into the final logit distribution.

This is why the layer sweep is essential — using the wrong layers would mean reading noise instead of signal.

---

## Chapter 6: Layer Range Selection — Strength in Numbers

*(Done in [02_train_se_probes.ipynb](../notebooks/02_train_se_probes.ipynb) Section 8)*

Instead of using just the single best layer, the system selects a **contiguous window of layers** and concatenates their hidden states. This gives the probe more features to work with and makes it more robust.

### The Algorithm

```python
for window_size in [4, 8, 16]:
    for start in range(33 - window_size + 1):
        mean_auroc = average(auroc_list[start : start + window_size])
        # track the window with highest mean AUROC
```

It tries windows of 4, 8, and 16 layers, slides each across all 33 positions, and picks the one with the highest average AUROC.

### What This Means for Feature Size

| Window | Feature Dimension |
|--------|------------------|
| 4 layers | 4 x 4096 = **16,384** features |
| 8 layers | 8 x 4096 = **32,768** features |
| 16 layers | 16 x 4096 = **65,536** features |

For the actual trained probes (Llama 3.1 8B), all four ended up using a **window of 4**, giving 16,384 features per probe. This is a massive reduction from using all 33 layers (135,168 features) while keeping the most informative ones.

### Final Layer Ranges

```
Energy SLT:  layers [17, 21)  — mid-depth semantic layers
Energy TBG:  layers [28, 32)  — deep task-preparation layers  
Entropy SLT: layers [20, 24)  — upper-mid semantic layers
Entropy TBG: layers [21, 25)  — upper-mid layers
```

Each probe "knows" exactly which slice of the model's internal representations to read.

---

## Chapter 7: Final Probe Training — The Main Event

*(Done in [02_train_se_probes.ipynb](../notebooks/02_train_se_probes.ipynb) Section 10)*

With layer ranges decided, the four production probes are trained on the full training set:

### For each probe, the training process is:

**Step 1: Feature Extraction**
```python
X_train = hidden_states[:, layer_start:layer_end, :].reshape(N, -1)
# e.g., for SLT Energy: shape (400, 4*4096) = (400, 16384)
```

**Step 2: StandardScaler Normalization**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
Each of the 16,384 features gets zero-mean and unit-variance normalization. This is critical because hidden state magnitudes vary wildly across layers and dimensions. The scaler is saved alongside the probe for inference.

**Step 3: Logistic Regression Training**
```python
probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X_train_scaled, binary_labels)
```

- **C=1.0:** Standard L2 regularization strength. This penalizes large weights, preventing overfitting to the 16,384-dimensional feature space with only ~400 training samples.
- **max_iter=1000:** Enough iterations for the solver to converge.
- **What the model learns:** A weight vector of 16,384 dimensions + 1 bias term. Each weight corresponds to one dimension in one layer's hidden state. The model learns which dimensions in which layers are most predictive of the teacher signal.

**Step 4: Validation**
```python
y_score = probe.predict_proba(X_val_scaled)[:, 1]  # P(class=1)
auroc = roc_auc_score(y_val, y_score)
ci = bootstrap_auroc(probe, scaler, X_test, y_test, n_boot=1000)
```

### Performance (Llama 3.1 8B, TriviaQA, 400 training records):

| Probe | AUROC | 95% CI |
|-------|-------|--------|
| SLT Energy | 0.667 | [0.522, 0.796] |
| TBG Energy | 0.748 | [0.621, 0.871] |
| SLT Entropy | 0.788 | [0.676, 0.893] |
| TBG Entropy | 0.786 | [0.679, 0.892] |

### The Four Trained Probes — Summary Table

| Probe | Position | Teacher | Output Meaning | Risk Inversion |
|-------|----------|---------|---------------|----------------|
| TBG Energy | Last prompt token | Energy label | P(high confidence) | risk = 1 - output |
| TBG Entropy | Last prompt token | Entropy label | P(high uncertainty) | risk = output directly |
| SLT Energy | 2nd-to-last generated token | Energy label | P(high confidence) | risk = 1 - output |
| SLT Entropy | 2nd-to-last generated token | Entropy label | P(high uncertainty) | risk = output directly |

Notice the asymmetry: Energy probes predict confidence (so risk = 1 - prediction), while Entropy probes predict uncertainty (so risk = prediction directly).

---

## Chapter 8: The Probe Bundle — Packaging Everything

*(Done in [02_train_se_probes.ipynb](../notebooks/02_train_se_probes.ipynb) Section 16)*

Everything gets pickled into a single ~2.1 MB file at `backend/models/probes_llama3-8b_triviaqa.pkl`:

```
probe_bundle = {
    # 4 LogisticRegression objects (the actual probes)
    # 4 StandardScaler objects (one per probe)
    # 4 layer ranges: (17,21), (28,32), (20,24), (21,25)
    # Binarization thresholds (energy=0.7504, entropy=0.2052)
    # Per-layer AUROC tables (33 values x 4 curves)
    # Test set performance with 95% CIs
    # Model ID and dataset metadata
}
```

At startup, the backend loads this bundle once, and the probes are ready for inference.

---

## Chapter 9: Inference — TBG Mode (The Quick Glance)

*(Implemented in [engine.py:404-460](../backend/engine.py#L404-L460))*

When a user submits a question and wants fast pre-generation risk assessment:

```
User question
    |
    v
[Tokenize prompt, run single forward pass]
    |
    v
[Extract hidden state at last prompt token]  ->  shape (33, 4096)
    |
    +---> [Slice layers 28:32] -> (4, 4096) -> flatten -> (1, 16384)
    |        -> StandardScaler -> LogisticRegression -> P(confident)
    |        -> energy_risk = 1 - P(confident)
    |
    +---> [Slice layers 21:25] -> (4, 4096) -> flatten -> (1, 16384)
             -> StandardScaler -> LogisticRegression -> P(uncertain)
             -> entropy_risk = P(uncertain)
    |
    v
combined_risk = 0.70 * energy_risk + 0.30 * entropy_risk
    |
    v
< 0.35 -> "high confidence" (green)
0.35-0.65 -> "medium" (yellow)  
>= 0.65 -> "low confidence" (red)
```

Energy gets 70% weight because it proved more reliable in validation. The whole process takes 0.5-2 seconds.

---

## Chapter 10: Inference — SLT Mode (The Deep Read)

*(Implemented in [engine.py:462-690](../backend/engine.py#L462-L690))*

When more accurate, post-generation scoring is needed:

### Step 1: Generate one answer (5-10 sec)
### Step 2: Run a forward pass on (prompt + answer) to extract hidden states
### Step 3: Score at two granularities

**Overall SLT scoring** — same as TBG but at the SLT position with SLT layer ranges:
- Energy probe on layers [17:21], entropy probe on layers [20:24]
- Combined: `0.65 * energy_risk + 0.35 * entropy_risk`

**Per-sentence scoring** — the unique power of SLT mode:
- The system identifies sentence boundaries in the answer
- Maps each sentence to its last token's position
- Extracts hidden states at each sentence-end position
- Runs both probes on each sentence independently
- Each sentence gets its own risk score and color badge

### Step 4: Aggregate (short vs. long answers)

For **short answers** (<=100 tokens):
```
combined_risk = 0.5 * overall_SLT + 0.5 * mean(per_sentence_risks)
```

For **long answers** (>100 tokens):
```
combined_risk = 0.15 * overall_SLT 
              + (0.25/log(n)) * max(per_sentence_risks)
              + remaining_weight * mean(per_sentence_risks)
```

The logic: in long answers, the overall SLT position is far from some sentences, so per-sentence scores get more weight. The max-risk sentence gets special attention (one bad sentence can tank credibility).

---

## Chapter 11: What the Logistic Regression Actually "Sees"

At its core, each probe is a linear classifier with 16,384 weights + 1 bias. When it scores a hidden state vector:

```
logit = w_1*x_1 + w_2*x_2 + ... + w_16384*x_16384 + bias
probability = sigmoid(logit) = 1 / (1 + exp(-logit))
```

Each weight corresponds to one neuron in one layer. A large positive weight on dimension `d` in layer `L` means: "when this neuron fires strongly, the model is more likely to be confident/uncertain." The probe has essentially learned which internal neurons are the hallucination detectors.

This is why the **StandardScaler is non-negotiable**: without normalization, dimensions with large absolute magnitudes would dominate the linear combination, drowning out informative but small-magnitude features.

And this is why **layer selection matters**: layers outside the informative window add 4,096 noise dimensions per layer. More noise = harder for the linear probe to find the signal, especially with only 400 training samples.

---

## Summary: The Full Picture

```
TRAINING (offline, one-time, ~hours)
=====================================
Questions (TriviaQA)
    -> Generate 5 answers each
    -> Cluster semantically
    -> Compute Energy score (concentration) and Entropy score (diversity)
    -> Binarize with MSE-optimal thresholds
    -> Extract hidden states at TBG and SLT positions
    -> Sweep all 33 layers to find AUROC curves
    -> Select best 4-layer windows per probe
    -> Train 4 LogisticRegression probes on concatenated layer features
    -> Package into pickle bundle

INFERENCE (runtime, per-question, seconds)
==========================================
Question
    -> Single forward pass
    -> Extract hidden states at target positions
    -> Slice the right layers for each probe
    -> Scale with saved StandardScaler
    -> LogisticRegression.predict_proba()
    -> Combine energy + entropy risks
    -> Map to confidence level
```

The elegance is in the trade-off: spend hours offline generating multiple answers and clustering them to create teacher signals, then distill all that knowledge into four tiny linear models that can replicate the judgment in milliseconds at inference time.
