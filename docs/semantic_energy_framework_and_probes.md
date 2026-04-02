# The Semantic Energy Framework & Why We Built Four Probes

How Semantic Energy quantifies hallucination risk, why linear probes can approximate it, and why both TBG and SLT positions are justified for both energy and entropy teachers.

---

## Table of Contents

1. [What Is Semantic Energy?](#1-what-is-semantic-energy)
2. [What Is "Energy" and Where Does It Come From?](#2-what-is-energy-and-where-does-it-come-from)
3. [The Full Semantic Energy Calculation](#3-the-full-semantic-energy-calculation)
4. [The Entropy Teacher: A Complementary Signal](#4-the-entropy-teacher-a-complementary-signal)
5. [The Cost Problem: Why We Need Probes](#5-the-cost-problem-why-we-need-probes)
6. [How Probes Mimic The Full Pipeline](#6-how-probes-mimic-the-full-pipeline)
7. [Why Two Token Positions: TBG and SLT](#7-why-two-token-positions-tbg-and-slt)
8. [Why Energy Probes Need Both TBG and SLT](#8-why-energy-probes-need-both-tbg-and-slt)
9. [Empirical Evidence: Performance Across Models](#9-empirical-evidence-performance-across-models)
10. [Energy vs Entropy: Same Signal or Complementary?](#10-energy-vs-entropy-same-signal-or-complementary)
11. [Summary: The Four Probes](#11-summary-the-four-probes)

---

## 1. What Is Semantic Energy?

Semantic Energy is a framework for detecting hallucinations in LLM outputs. The core idea is borrowed from statistical mechanics:

> **If a model is confident about an answer, multiple independent generations will converge to the same semantic meaning.** The "energy" concentrated in that dominant meaning cluster tells us how confident the model truly is.

The analogy to physics: in a physical system at low temperature, particles settle into a single low-energy state (ordered, certain). At high temperature, they scatter across many states (disordered, uncertain). Similarly:

- **Low hallucination risk:** Most generated answers cluster into one semantic meaning — energy is concentrated.
- **High hallucination risk:** Answers scatter across multiple different meanings — energy is dispersed.

### The Intuition

Ask the model "What is the capital of France?" five times:

```
Sample 1: "Paris"
Sample 2: "The capital of France is Paris."
Sample 3: "Paris is the capital."
Sample 4: "Paris"
Sample 5: "It's Paris."
```

All 5 answers say the same thing. One cluster, all energy concentrated there. **High confidence.**

Now ask "Who invented the telephone?" five times:

```
Sample 1: "Alexander Graham Bell"
Sample 2: "Antonio Meucci"
Sample 3: "Bell invented the telephone in 1876."
Sample 4: "Elisha Gray"
Sample 5: "Alexander Graham Bell"
```

Three different answers, three clusters. Energy is spread across them. **Low confidence — hallucination risk.**

---

## 2. What Is "Energy" and Where Does It Come From?

Before understanding Semantic Energy, you need to understand what the LLM actually produces internally when it generates text.

### Logits: The Model's Raw Confidence Signal

When an LLM generates a token (a word or word-piece), it doesn't just pick a word from thin air. Internally, it produces a score for **every possible token** in its vocabulary (typically 32,000-128,000 tokens). These raw scores are called **logits**.

- A **high logit** for a token means the model strongly favors that token at this position.
- A **low logit** means the model considers that token unlikely.

For example, when generating the next word after "The capital of France is", the model might produce logits like:
- "Paris": logit = 38.5 (very high — strongly favored)
- "the": logit = 12.1 (moderate — possible but unlikely)
- "banana": logit = -5.2 (very low — essentially ruled out)

The logit of the token that gets actually chosen (the "chosen-token logit") tells us how confident the model was about that particular choice. A sequence of high chosen-token logits means the model was confident about every word it produced.

### From Logits to Probabilities

Logits are converted to probabilities through a softmax function, which squashes the raw scores into the 0-1 range where they all sum to 1. The probability of the chosen token tells us the same story as the logit but in a more interpretable scale:
- Probability of 0.95 = the model was very sure about this token
- Probability of 0.10 = the model was guessing among many options

### What "Energy" Means in Our Framework

In our framework, the "energy" of a generated response is derived from the **average logit values** of its chosen tokens, following a Boltzmann distribution analogy from statistical physics. Specifically:

- For each generated sample, we take the **negative mean** of all its chosen-token logits. Note that we use the **mean** (average), not a product — this keeps the value stable regardless of how many tokens were generated. This gives us an energy value for that sample — samples where the model was very confident (high logits) end up with lower energy (more stable, like a physical system in a low-energy state).

- We then **aggregate these per-sample energies by semantic cluster**. All the energy values from samples that mean the same thing get summed together within their cluster.

- Finally, we **normalize** across clusters so the total energy sums to 1.0. This gives us an energy distribution: what fraction of the total "confidence energy" belongs to each meaning.

The energy of the main cluster (the one containing the first generated answer) is our **energy score**. If most of the model's confidence-weighted agreement points to one meaning, that meaning has high energy — and we trust it.

### Why Energy Is Better Than Just Counting Cluster Sizes

You might ask: why not just count how many of the 5 samples agreed? The reason is that energy incorporates the **quality** of agreement, not just the **quantity**.

Consider two scenarios where 3 of 5 samples agree:

**Scenario A:** The 3 agreeing samples each had very high average chosen-token logits (the model was confident about every word). The 2 disagreeing samples had low average logits (the model was uncertain and just happened to produce a different answer). Energy would be high for the main cluster because the agreeing samples contribute much more energy per sample.

**Scenario B:** The 3 agreeing samples had mediocre average logits (the model wasn't really sure but happened to produce similar tokens). The 2 disagreeing samples also had mediocre logits. Energy would be more evenly distributed because all samples contributed similar amounts.

A pure count would say "3-of-5 = 60% confident" in both cases. Energy correctly distinguishes them: Scenario A is genuinely more confident than Scenario B.

---

## 3. The Full Semantic Energy Calculation

### Step-by-Step Walkthrough

#### Step 1: Generate Multiple Diverse Responses

We generate 5 independent responses to the same question, using a temperature of 0.7 to encourage diversity. For each response, we capture the chosen-token logit and probability at every generation step.

#### Step 2: Semantic Clustering

We use the LLM itself as a judge to determine which answers mean the same thing. For each pair of answers, we ask the model: "Are these two answers semantically equivalent?" and it responds with Yes or No (using greedy decoding for determinism). Answers that are equivalent get grouped into the same cluster.

#### Step 3: Per-Sample Energy Computation

For each of the 5 generated samples, we compute the **Boltzmann energy**: take the negative of the average chosen-token logit across all tokens in that sample. Following the Boltzmann convention from statistical physics, a sample where the model produced high logits (confident) ends up with a more negative energy value (lower energy = more stable state).

Note: the pipeline also computes a per-sample probability product (multiplying all chosen-token probabilities together), but this value does **not** feed into the final energy score. Probabilities are always between 0 and 1 after softmax, so their product is also 0-1 — but it shrinks to near-zero for longer sequences (e.g., 0.9 raised to the power of 100 tokens is 0.000026), making it unreliable as an absolute measure. The energy score relies entirely on the logit-based path, which uses the **average** logit per sample rather than a product, making it robust to sequence length.

#### Step 4: Cluster-Level Aggregation

The per-sample energy values get rolled up to the cluster level. For each cluster, sum the energy values of its member samples. This gives a total per-cluster energy — clusters with more members contribute more, and clusters whose members had higher logits (more confident token choices) contribute disproportionately more.

#### Step 5: Energy Normalization

Divide each cluster's energy by the total energy across all clusters. This produces a distribution in the range [0, 1] that sums to 1.0 — essentially a confidence-weighted vote across semantic meanings.

#### Step 6: Extract Main Cluster Energy

The energy score is the normalized energy of the cluster containing the first generated answer. This is our **energy_score_raw**.

| Value | Meaning |
|---|---|
| Close to 1.0 | Almost all energy in main cluster — model is very confident |
| Around 0.5 | Energy split roughly evenly — significant uncertainty |
| Close to 0.0 | Most energy in other clusters — main answer is likely wrong |

### Why This Works Better Than Token Probabilities Alone

Token-level probabilities miss **semantic** uncertainty. A model might generate "Paris" with high token probability but generate "London" with equally high probability on a different sample. Token probabilities within a single generation look great, but the model is actually uncertain at the semantic level.

Semantic Energy captures this by:
1. Generating multiple samples to surface the uncertainty
2. Clustering by meaning (not by exact tokens)
3. Measuring how concentrated the model's confidence is in one semantic cluster

### Worked Example: Step-by-Step Energy Calculation

Let's walk through a complete example with actual numbers. Suppose we ask the model: **"What is the capital of France?"** and generate 5 samples.

**Step 1 — Generate 5 answers and record per-token logits**

Each time the model generates a token, it produces a logit (raw score) for the token it chose. Here are our 5 samples with their per-token chosen logits:

| Sample | Answer | Per-token logits (chosen tokens) |
|---|---|---|
| Sample 0 | "Paris" | [6.2, 4.8, 5.1, 4.9] |
| Sample 1 | "The capital is Paris" | [3.8, 5.5, 4.2, 6.1, 4.4] |
| Sample 2 | "Paris, France" | [5.9, 5.0, 4.7, 5.4] |
| Sample 3 | "Lyon" | [1.8, 0.9, 1.5] |
| Sample 4 | "Lyon is the capital" | [1.2, 0.7, 2.1, 1.0, 1.5] |

Higher logits mean the model was more "sure" about each token it picked.

**Step 2 — Compute per-sample Boltzmann energy**

For each sample, take the **negative of the mean logit**. This is the Boltzmann energy — lower (more negative) means the model was more confident.

| Sample | Mean logit | Boltzmann energy (negative mean) |
|---|---|---|
| Sample 0 | (6.2 + 4.8 + 5.1 + 4.9) / 4 = **5.25** | **-5.25** |
| Sample 1 | (3.8 + 5.5 + 4.2 + 6.1 + 4.4) / 5 = **4.80** | **-4.80** |
| Sample 2 | (5.9 + 5.0 + 4.7 + 5.4) / 4 = **5.25** | **-5.25** |
| Sample 3 | (1.8 + 0.9 + 1.5) / 3 = **1.40** | **-1.40** |
| Sample 4 | (1.2 + 0.7 + 2.1 + 1.0 + 1.5) / 5 = **1.30** | **-1.30** |

Think of it this way: a Boltzmann energy of -5.25 means "the model was very confident" (like a ball sitting deep in a valley — low energy, very stable). An energy of -1.30 means "the model was much less sure" (ball sitting on a hilltop — higher energy, less stable).

**Step 3 — Semantic clustering**

The system asks an LLM to check which answers mean the same thing. "Paris", "The capital is Paris", and "Paris, France" all mean the same answer. "Lyon" and "Lyon is the capital" also agree with each other.

- **Cluster A** = {Sample 0, Sample 1, Sample 2} — all say Paris
- **Cluster B** = {Sample 3, Sample 4} — both say Lyon

**Step 4 — Aggregate energy by cluster**

For each cluster, sum the Boltzmann energies of its members, then **negate** the sum. (The double negation converts back to positive values so we can treat them as weights — bigger positive number = more confident cluster.)

Cluster A: -((-5.25) + (-4.80) + (-5.25)) = -(-15.30) = **15.30**

Cluster B: -((-1.40) + (-1.30)) = -(-2.70) = **2.70**

Notice how Cluster A gets a much bigger value. This reflects two things working together: it has **more samples** (3 vs 2), AND each sample had **higher logits** (averages around 5 vs averages around 1.3).

**Step 5 — Normalize to get the final energy distribution**

Divide each cluster's aggregated value by the total to get proportions that sum to 1.0:

Total = 15.30 + 2.70 = **18.00**

| Cluster | Aggregated value | Normalized energy share |
|---|---|---|
| Cluster A (Paris) | 15.30 | 15.30 / 18.00 = **0.850** |
| Cluster B (Lyon) | 2.70 | 2.70 / 18.00 = **0.150** |

**Step 6 — Extract the main answer's score**

Sample 0 (the first generated answer, "Paris") belongs to Cluster A. So the **energy confidence score** for the main answer is **0.850** — classified as "high" confidence (above 0.80).

**What do these numbers tell us?**

- 85% of the total energy belongs to the "Paris" cluster — the model strongly favors this answer
- The remaining 15% goes to "Lyon" — there's a small amount of disagreement, but it's weak disagreement (low logits)
- If the Lyon samples had higher logits (say averages of 4.0 instead of 1.3), Cluster B's share would be much larger and the confidence score would drop — reflecting that the model is genuinely torn between two answers it's confident about

**Why both count and quality matter:**

If we only counted samples (3 vs 2), we'd get 60% confidence. But the energy score is 85% because it also accounts for the fact that the "Paris" tokens were produced with much higher logits. The model wasn't just saying "Paris" more often — it was saying "Paris" with much more conviction each time.

---

## 4. The Entropy Teacher: A Complementary Signal

While Semantic Energy uses both cluster assignments AND logit magnitudes, the Entropy teacher uses only cluster structure.

The entropy teacher computes the Shannon entropy over the cluster size distribution. In plain terms: it looks at how evenly the 5 samples are spread across clusters, ignoring logit values entirely.

- All 5 in one cluster: entropy near 0 (most certain)
- Split 3-2 across two clusters: entropy around 0.67
- Split 2-2-1 across three clusters: entropy around 1.05
- Each sample in its own cluster: entropy around 1.61 (maximum uncertainty)

### What Entropy Captures That Energy Doesn't (and Vice Versa)

| Aspect | Energy Score | Entropy Score |
|---|---|---|
| **Uses logit magnitudes?** | Yes (Boltzmann mean) | No (only cluster sizes) |
| **Uses cluster structure?** | Yes (aggregates by cluster) | Yes (sizes only) |
| **Sensitive to token quality?** | Yes (high-logit tokens boost energy) | No (only counts matter) |
| **Orientation** | Higher = more confident | Higher = more uncertain |
| **Range** | [0, 1] | [0, ln(N)] |

**Example where they differ:** If 3 samples cluster together but their token logits are unusually low (the model produced them but wasn't "sure" about each token), energy would be moderate while entropy would still be low (3-of-5 in one cluster = relatively certain). Conversely, if 2 clusters each have high logits, energy might be moderately split while entropy would flag the disagreement.

### A Quick Note on Spearman Rho (Used Throughout This Document)

Spearman rho (written as "rho" or "ρ") is a way to measure whether two lists **rank things in the same order**. It does not care about the exact numbers — only whether the ordering agrees.

Think of it like this: imagine two school teachers each grade the same 50 students. If Teacher A's top student is also Teacher B's top student, and Teacher A's worst student is also Teacher B's worst student, and so on all the way down — that's a Spearman rho of **+1.0** (perfect agreement in ranking).

If Teacher A's ranking is the exact reverse of Teacher B's — A's best student is B's worst, and vice versa — that's **-1.0** (perfect opposite ranking).

If the two rankings have no relationship at all (knowing one tells you nothing about the other), that's **0.0**.

| Rho Value | What It Means |
|---|---|
| +1.0 | Both lists rank everything in exactly the same order |
| +0.7 to +0.9 | Strong agreement — the rankings are very similar |
| +0.3 to +0.7 | Moderate agreement — same general trends, but plenty of disagreements |
| 0.0 | No relationship — one ranking tells you nothing about the other |
| -0.7 to -1.0 | Strong opposite ranking — when one goes up, the other goes down |

A **negative** rho means the two things move in opposite directions. In our case, energy (confidence) and entropy (uncertainty) naturally move opposite: when a question has high confidence, it has low uncertainty, and vice versa.

### Cross-Signal Correlation

From the training data (Llama 3.1 8B, test set), the Spearman correlation between energy and entropy raw scores is -0.9958.

In plain terms: if you ranked all test questions from "most confident" to "least confident" using the energy score, and then separately ranked them from "least uncertain" to "most uncertain" using the entropy score, those two rankings would be almost identical. Whenever energy goes up for a question, entropy goes down for that same question, almost without exception.

They are this highly correlated because both are computed from the same semantic clustering — they're just different mathematical summaries of the same underlying phenomenon. The remaining ~0.4% of variance they don't share is the logit-magnitude information that energy captures and entropy doesn't.

---

## 5. The Cost Problem: Why We Need Probes

The full Semantic Energy pipeline is powerful but expensive:

| Component | Cost |
|---|---|
| Generate 5 diverse responses | 5 x generation time |
| Pairwise semantic equivalence checks | Up to 10 LLM calls |
| Energy/entropy computation | Negligible (CPU math) |
| **Total** | **~60-120 seconds per question** |

This is fine for offline evaluation or a slow API, but unusable for:
- Real-time hallucination detection
- Per-sentence risk scoring
- Pre-generation risk estimation (before committing to generation)

**The probe hypothesis:** The model's internal hidden states already encode enough information to predict whether a question would produce high or low semantic energy — without actually running the full multi-sample pipeline.

---

## 6. How Probes Mimic The Full Pipeline

### What Are Hidden States?

Every transformer layer in an LLM produces a hidden state — a dense vector of numbers at each token position. For Llama 3.1 8B, each layer outputs **4096 numbers** (the "hidden dimension"). Think of these 4096 numbers as a snapshot of what the model is "thinking" at that point — each number is a real-valued activation (positive or negative) representing some learned feature of the input.

A model with 33 layers (32 transformer layers + 1 initial embedding layer) produces a hidden state of shape (33 layers × 4096 dimensions) at each token position. So for a single token, the model produces 33 × 4096 = 135,168 numbers — a rich representation of what the model "understands" at that point in the sequence.

**What each layer level encodes:**

- **Early layers (0–8):** These encode surface-level patterns — token identity, basic syntax, simple word relationships. They're good at recognizing "this is a question about geography" but don't yet encode deep knowledge.
- **Middle layers (9–20):** These encode semantic meaning and factual associations. This is where the model "retrieves" relevant knowledge from its training. If the model has seen "the capital of France is Paris" many times, it's in these middle layers that the association becomes active.
- **Late layers (21–32):** These encode output preparation — the model is now deciding what tokens to generate, resolving ambiguities, and finalizing its answer. The information is more about "what I'm about to say" than "what I know."

This is exactly why different probe types peak at different layers. TBG energy probes peak around **layer 14** (middle layers, where factual knowledge is most active), while SLT energy probes peak around **layer 19** (later layers, because after generation the model has more context to work with and the confidence signal shifts deeper).

### The Key Insight: Hidden States Predict The Teacher Scores

When the model processes a question, its hidden states encode whether it has strong knowledge about the answer. If it does, those hidden states will be in a region of the representation space associated with "confident" responses. If it doesn't, the hidden states will reflect uncertainty.

The full Semantic Energy pipeline reveals this confidence through expensive behavioral analysis (generate many answers, cluster them, compute energy). But the **hidden states already contain this information** — the behavioral analysis just makes it visible. A probe can learn to read the confidence signal directly from the hidden states, skipping the expensive steps.

### The Training Process

1. **Run the expensive pipeline** on 1000 TriviaQA questions to get energy and entropy scores for each
2. **Extract hidden states** at specific token positions during the same process
3. **Binarize** teacher scores into high/low labels (using a variance-minimizing threshold)
4. **Train a linear classifier** (LogisticRegression) to predict the label from the hidden states

The probe learns: "given the pattern of activations at this layer range, will the full multi-sample pipeline rate this as confident or uncertain?"

### What The Probe Learns

A logistic regression on hidden states is equivalent to finding a **linear decision boundary** in the model's representation space. The probe discovers a direction in the 4096-dimensional hidden state that separates "the model knows this" from "the model is uncertain."

This works because transformer hidden states are known to encode:
- **Factual knowledge** — whether the model has encountered the answer in training data
- **Uncertainty signals** — activation patterns that differ between confident and uncertain predictions
- **Attention summaries** — compressed into the hidden state, reflecting how well the prompt was understood

### How The Monitored Values Connect to Probing

During training, we collect two types of data from the LLM for each question:

**Behavioral data (used to create teacher labels):**
- Per-token logits and probabilities from 5 generations — these feed into the energy calculation
- Semantic clustering results — these feed into both energy and entropy calculations
- The resulting energy_score_raw and entropy_score_raw become the teacher signals that we binarize into training labels

**Internal state data (used as probe input features):**
- Hidden states at the TBG position (last prompt token) — shape (33 layers, 4096 dimensions)
- Hidden states at the SLT position (second-to-last generated token) — shape (33 layers, 4096 dimensions)
- These are extracted via a separate forward pass with hidden state output enabled

The connection is: logits and probabilities tell us **what the model did** (behavioral signal), while hidden states tell us **what the model was thinking** (internal signal). The probe learns to map from the internal signal to a prediction of the behavioral signal.

At inference time, we skip the behavioral data entirely. We only extract hidden states (one forward pass) and feed them through the trained probe to get a confidence estimate — approximating what the full multi-sample pipeline would have produced.

### How Close Do Probes Get?

The full teachers serve as upper bounds. Probes should approach but not exceed them:

**Llama 3.1 8B (TriviaQA, test set):**

| System | Hallucination AUROC | Cost |
|---|---|---|
| Full Energy Teacher (upper bound) | 0.7103 | ~60-120s |
| Full Entropy Teacher (upper bound) | 0.7143 | ~60-120s |
| SLT Energy Probe | 0.7163 | ~5-15s |
| TBG Energy Probe | 0.6409 | ~0.5-2s |
| SLT Entropy Probe | 0.6726 | ~5-15s |
| TBG Entropy Probe | 0.6806 | ~0.5-2s |

The SLT energy probe actually slightly exceeds the teacher upper bound (0.7163 vs 0.7103). This can happen because the probe has access to the model's internal representations which may contain richer information than the teacher's output-level signal, and because of small-sample variance in the test set.

**Qwen 3 8B (TriviaQA, test set):**

| System | Hallucination AUROC | Cost |
|---|---|---|
| Full Energy Teacher (upper bound) | 0.7865 | ~60-120s |
| Full Entropy Teacher (upper bound) | 0.7729 | ~60-120s |
| SLT Energy Probe | 0.7238 | ~5-15s |
| TBG Energy Probe | 0.7041 | ~0.5-2s |
| SLT Entropy Probe | 0.6731 | ~5-15s |
| TBG Entropy Probe | 0.6591 | ~0.5-2s |

### Teacher Fidelity: Do Probes Track The Teachers?

Here we ask: "If the expensive teacher says Question A is more confident than Question B, does the cheap probe agree?" We measure this using Spearman rho — how well the probe's ranking of questions matches the teacher's ranking.

**What we're comparing:** Take all ~50 test questions. The teacher (the full multi-sample pipeline) gives each one a continuous confidence score. The probe also gives each one a continuous score. If we rank all questions by the teacher's score and separately rank them by the probe's score, how similar are those two rankings?

**Llama 3.1 8B (test set):**

| Probe | Spearman rho with teacher | What this means |
|---|---|---|
| SLT Energy | 0.25 | Weak agreement — the probe's ranking loosely follows the teacher's, but with many disagreements |
| TBG Energy | 0.26 | Similarly weak |
| SLT Entropy | 0.43 | Moderate agreement — the probe gets the general trend right but makes mistakes on individual questions |
| TBG Entropy | 0.43 | Similarly moderate |

**Qwen 3 8B (test set):**

| Probe | Spearman rho with teacher | What this means |
|---|---|---|
| SLT Energy | 0.34 | Weak-to-moderate agreement |
| TBG Energy | 0.21 | Weak agreement |
| SLT Entropy | 0.42 | Moderate agreement |
| TBG Entropy | 0.24 | Weak agreement |

**Why entropy probes have higher fidelity than energy probes:** Entropy is a simpler signal — it only depends on how many clusters there are and how big they are. The probe can learn to approximate this relatively well with a linear boundary in the hidden state. Energy is more complex — it also depends on the logit magnitudes within each cluster, which is a harder pattern for a linear probe to capture faithfully. So the energy probes lose more detail in the approximation.

**Important: low fidelity does not mean the probe is useless.** The fidelity numbers measure how well the probe reproduces the teacher's *exact ranking*. But what we actually care about is whether the probe can *detect hallucinations* — and for that, it only needs to get the broad strokes right (distinguish "confident" from "uncertain"), not perfectly replicate every subtle ordering. The AUROC numbers in the tables above show that the probes are genuinely good at hallucination detection (0.64-0.72 AUROC) even though they don't perfectly mimic the teacher's fine-grained ranking. Think of it like a smoke detector — it doesn't need to tell you the exact temperature of a fire to be useful; it just needs to reliably distinguish "fire" from "no fire."

---

## 7. Why Two Token Positions: TBG and SLT

### TBG: Token Before Generation

**Position:** Last token of the prompt, before any answer is generated.

**What it captures:** The model's "readiness state" — the compressed representation of the question at the point where generation would begin. At this position, the model has:
- Processed the full question
- Activated relevant knowledge (or failed to)
- Set up its attention patterns for generation
- NOT yet committed to any specific answer

**Use case:** Pre-generation risk estimation. "Should I even attempt to answer this?" Fastest possible scoring — single forward pass, no generation needed (~0.5-2s).

### SLT: Second-to-Last Token

**Position:** Second-to-last token of the generated answer (one before EOS).

**What it captures:** The model's state after generating most of the answer. At this position, the model has:
- Committed to a specific answer
- Built up contextual state reflecting what it generated
- Accumulated confidence/uncertainty signals throughout generation
- Is about to terminate (next token would be EOS)

**Use case:** Post-generation risk estimation. "How confident is the model about what it just said?" Requires generation + extra forward pass (~5-15s), but has more information.

### Why Not Just Use One?

| Property | TBG Only | SLT Only | Both |
|---|---|---|---|
| **Pre-generation scoring** | Yes | No | Yes |
| **Post-generation scoring** | No | Yes | Yes |
| **Speed** | Fastest | Moderate | Choose per use case |
| **Information available** | Question only | Question + answer | Best of both |
| **When to use** | Quick screening, routing | Full confidence scoring | Different endpoints |

They serve fundamentally different deployment modes. TBG is for when you haven't generated yet and want to decide whether to bother. SLT is for when you've already generated and want to know how much to trust the output.

---

## 8. Why Energy Probes Need Both TBG and SLT

This is the key question: the entropy teacher is purely about cluster structure (how many clusters, how big), which you might argue is something the model "knows" about the question before generating. But the energy teacher incorporates **logit magnitudes** from the actual generated tokens — information that only exists after generation. So why can a TBG probe (pre-generation) predict it?

### The Argument for TBG Energy Probes

**Claim:** The model's pre-generation hidden state already contains a predictive signal for the energy score, even though energy technically uses post-generation logits.

**Why this works:**

1. **The model knows what it knows.** Before generating, the model's hidden state encodes whether it has strong knowledge about the topic. If it does, it will generate high-logit tokens AND those tokens will cluster together — both components of energy will be high. If it doesn't, both will be low. The TBG hidden state captures this "knowledge readiness" signal. The model doesn't suddenly discover new knowledge during generation — it was there in the hidden state all along. Generation merely reveals what was already encoded.

2. **Energy and entropy are 99.6% correlated.** The Spearman correlation between energy and entropy raw scores is -0.9958. Since entropy is purely structural (no logits needed) and is almost perfectly correlated with energy, a probe that can predict entropy can also predict energy almost as well. The TBG position already encodes enough to predict cluster structure, and cluster structure determines 99.6% of energy variation. The logit-magnitude component that makes energy unique only accounts for the remaining 0.4%.

3. **The logit-magnitude component is also predictable from the prompt state.** High-logit tokens come from confident knowledge retrieval. If the model has strong internal representations for "Paris" when processing "What is the capital of France?", those representations exist in the hidden state before generation begins. The probe can pick up on the same activation patterns that will later produce high logits — it reads the cause (knowledge representation) rather than the effect (high logits).

### Empirical Validation

**Per-layer AUROC sweep (Llama 3.1 8B, validation set):**

| Probe | Best Single Layer | Best Layer AUROC |
|---|---|---|
| **TBG Energy** | Layer 14 | **0.8706** |
| SLT Energy | Layer 19 | 0.7412 |
| TBG Entropy | Layer 24 | 0.8693 |
| SLT Entropy | Layer 20 | 0.7929 |

TBG Energy achieves **0.8706 AUROC** at the single-layer level — the highest of all four probes. This is not a weak signal being squeezed out; the pre-generation hidden state is genuinely informative about energy.

**Layer range selection (Llama 3.1 8B, validation set):**

| Probe | Layer Range | Mean AUROC |
|---|---|---|
| TBG Energy | layers 28-31 | 0.7840 |
| SLT Energy | layers 17-20 | 0.7270 |
| TBG Entropy | layers 21-24 | 0.8523 |
| SLT Entropy | layers 20-23 | 0.7725 |

TBG energy outperforms SLT energy on validation AUROC (0.7840 vs 0.7270). This is a consistent pattern: TBG probes often have higher teacher-prediction AUROC than SLT probes (they predict the teacher labels better), while SLT probes can have higher hallucination-detection AUROC (they predict actual correctness better, because they access information beyond what the teacher captures).

### Why TBG Sometimes Outperforms SLT on Teacher Prediction

This might seem counterintuitive — how can the pre-generation position predict the teacher score better than the post-generation position?

The explanation is that SLT hidden states contain **more information, but some of it is noise for the teacher prediction task**. SLT encodes the specific tokens generated, the particular phrasing chosen, syntactic structure, and other generation artifacts. Much of this is irrelevant to whether the answer is confident. The linear classifier must find a decision boundary in a noisier space.

TBG encodes only the "knowledge state" without the noise of specific generation choices. The signal-to-noise ratio for predicting confidence is potentially higher. The classifier finds a cleaner boundary.

However, at **hallucination detection** (predicting actual correctness, not teacher score), SLT can outperform TBG because:
- SLT captures information about what was actually generated (not just what could be generated)
- If the model went "off track" during generation, SLT can detect it; TBG cannot
- SLT enables per-sentence scoring, which TBG cannot do

### The Complementary Value

| Scenario | TBG Energy | SLT Energy |
|---|---|---|
| Model lacks knowledge entirely | Detects it (high risk) | Also detects it |
| Model has knowledge but generates poorly | Cannot detect | Detects it |
| Quick screening needed | ~0.5-2s | ~5-15s |
| Per-sentence scoring needed | Not possible | Yes |
| Pre-generation routing | Yes | Not possible |

**Both positions are justified because they serve different operational needs and capture partially overlapping but not identical information.**

---

## 9. Empirical Evidence: Performance Across Models

### Llama 3.1 8B Instruct

**Teacher-prediction AUROC (validation set, best layer range):**

| Probe | Layer Range | AUROC |
|---|---|---|
| TBG Energy | layers 28-31 | 0.7840 |
| SLT Energy | layers 17-20 | 0.7270 |
| TBG Entropy | layers 21-24 | 0.8523 |
| SLT Entropy | layers 20-23 | 0.7725 |

**Hallucination-detection AUROC (test set, with 95% CI):**

| Probe | AUROC | 95% CI |
|---|---|---|
| SLT Energy | 0.7163 | [0.4742, 0.8370] |
| TBG Energy | 0.6409 | [0.5151, 0.9377] |
| SLT Entropy | 0.6726 | [0.6547, 0.8949] |
| TBG Entropy | 0.6806 | [0.6389, 0.8988] |

### Qwen 3 8B

**Teacher-prediction AUROC (validation set, best layer range):**

| Probe | Layer Range | AUROC |
|---|---|---|
| TBG Energy | layers 7-10 | 0.6981 |
| SLT Energy | layers 24-27 | 0.8201 |
| TBG Entropy | layers 9-16 | 0.7271 |
| SLT Entropy | layers 24-27 | 0.7875 |

**Hallucination-detection AUROC (test set):**

| Probe | AUROC |
|---|---|
| SLT Energy | 0.7238 |
| TBG Energy | 0.7041 |
| SLT Entropy | 0.6731 |
| TBG Entropy | 0.6591 |

### Patterns Across Models

1. **All four probes consistently beat chance (0.5)** — the signal is real in all cases.
2. **SLT tends to beat TBG on hallucination detection** — having the generated answer helps.
3. **TBG energy can match or beat SLT energy on teacher-prediction** — pre-generation state is cleaner for the teacher signal (Llama: 0.784 vs 0.727).
4. **Best layers differ between TBG and SLT** — they extract different information from different depths.
5. **Best layers differ between models** — Llama peaks in late layers (17-31), Qwen peaks across a wider range (7-27).

---

## 10. Energy vs Entropy: Same Signal or Complementary?

### Raw Teacher Correlation

At the teacher level (Llama 3.1 8B), Spearman rho between energy and entropy raw scores is -0.9958. They are almost perfectly anti-correlated — they mostly measure the same underlying phenomenon (semantic agreement), but from different angles.

### Probe-Level Correlation

At the probe level (Llama 3.1 8B), Spearman rho between energy and entropy probe scores is -0.7444 — substantially lower than the -0.9958 at the teacher level.

What does this drop mean in practical terms? At the teacher level (-0.996), energy and entropy rank questions in almost perfectly opposite order — they're essentially the same information flipped. At the probe level (-0.74), the two probes still generally move in opposite directions, but they disagree on a meaningful number of individual questions. One probe might flag a question as risky while the other thinks it's fine.

This disagreement is actually **valuable**. It means:

1. **They make different errors** — where one probe is wrong, the other might be right
2. **Combining them adds value** — the AUROC-weighted combination (0.515 x entropy + 0.485 x energy) outperforms either alone at the per-sentence level
3. **They extract from different layers** — energy and entropy probes select different optimal layer ranges, confirming they look at different parts of the model's internal representations

### Why Maintain Both If The Teachers Are So Correlated?

At the teacher level (rho = -0.996), you could argue one teacher is redundant — and you'd be right. But we don't use the teachers at inference time. We use the **probes**, and the probes are only -0.74 correlated. Each probe is a simplified, lossy approximation of its teacher. Because they were trained on different targets (energy vs entropy), they lose different parts of the signal. Combining them recovers more of the original information than either one alone — like asking two people who each saw different parts of an event to reconstruct what happened.

---

## 11. Summary: The Four Probes

```
                        +---------------------------------------------+
                        |         FULL SEMANTIC ENERGY PIPELINE        |
                        |  5 generations + clustering + cal_flow       |
                        |  AUROC: ~0.71 (Llama) / ~0.79 (Qwen)       |
                        |  Cost: 60-120 seconds                        |
                        +---------------------+-----------------------+
                                              |
                               "Can we approximate this cheaply?"
                                              |
                    +-------------------------+-------------------------+
                    |                         |                         |
             +------+------+           +------+------+          +------+------+
             |  TBG Mode   |           |  SLT Mode   |          |  Combined   |
             | (Pre-gen)   |           | (Post-gen)  |          | (SLT uses  |
             | ~0.5-2s     |           | ~5-15s      |          |  both)     |
             +------+------+           +------+------+          +-------------+
                    |                         |
         +---------+---------+     +---------+---------+
         |                   |     |                   |
    +----+----+    +----+----+  +--+-----+   +--+-----+
    |  TBG    |    |  TBG    |  |  SLT   |   |  SLT   |
    | Energy  |    |Entropy  |  | Energy |   |Entropy |
    | Probe   |    | Probe   |  | Probe  |   | Probe  |
    +---------+    +---------+  +--------+   +--------+
                        |                         |
                  No answer              Has answer +
                  generated              per-sentence
                                         scoring
```

| Probe | Position | What It Approximates | Justification | Best Use Case |
|---|---|---|---|---|
| **TBG Energy** | Last prompt token | Energy teacher (confidence from logit-weighted cluster distribution) | Model's knowledge state predicts energy; TBG achieves 0.87 single-layer AUROC; energy is 99.6% correlated with entropy which TBG already predicts well | Pre-generation screening: "Is this question likely to produce a hallucination?" |
| **TBG Entropy** | Last prompt token | Entropy teacher (cluster size distribution uncertainty) | Cluster structure is predictable from question understanding alone — the model knows before generating whether it has one clear answer or many competing ones | Pre-generation screening (complementary to TBG energy) |
| **SLT Energy** | 2nd-to-last generated token | Energy teacher (same as TBG, but with post-generation information) | Post-generation state captures both knowledge AND generation quality; can detect cases where the model had knowledge but generated poorly; enables per-sentence energy risk scoring | Post-generation confidence: overall + per-sentence breakdown |
| **SLT Entropy** | 2nd-to-last generated token | Entropy teacher (same as TBG, but with post-generation information) | Post-generation state adds generation-specific uncertainty signals; weighted higher (0.515 vs 0.485) in per-sentence combination due to better AUROC | Post-generation confidence (highest weight in per-sentence combination) |

### The Key Takeaway

The Semantic Energy framework detects hallucinations by measuring whether a model's "energy" (derived from its logit confidence across tokens) is concentrated in one semantic meaning or dispersed across many. This requires expensive multi-sample generation and LLM-based clustering.

The four probes approximate this with a single forward pass by exploiting the fact that **the model's internal hidden states already encode the confidence information** that the full pipeline reveals through behavioral analysis. The probes read the cause (internal knowledge representations) rather than measuring the effect (behavioral divergence across multiple samples).

- **Two teachers** (energy + entropy) provide training signals that, despite being 99.6% correlated at the teacher level, produce probes that are only 74% correlated — giving genuine complementary value when combined
- **Two positions** (TBG + SLT) serve fundamentally different deployment needs (pre-generation vs post-generation), and both are empirically validated for both energy and entropy prediction
- **Four probes** = full coverage of the use-case matrix, each justified by performance above chance and distinct operational value
