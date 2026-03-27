# SemanticEnergy — Algorithm Design

## Algorithm 1: SLT Post-Generation Hallucination Scoring (Primary)

```
Algorithm 1: SLT Post-Generation Hallucination Scoring
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data:    Question Q, Probe Bundle P = {probes, scalers, layer_ranges}
Result:  Risk scores R = (energy_risk, entropy_risk, combined_risk),
         Sentence-level analysis S

 1:  /* Pass 1: Generate answer using LLM                          */
 2:  A, L, T ← model.generate(Q, temperature=0.7, max_tokens=512)
     /* A = answer text, L = per-token logits, T = token_ids       */

 3:  /* Sentence-level scoring from generation logits               */
 4:  sentences ← split_sentences(A)
 5:  token_map ← align_tokens_to_sentences(A, T, sentences)
 6:  S ← ∅
 7:  for each sentence sᵢ ∈ sentences do
 8:      Lᵢ ← {L[t] : token_map[t] = i}
         /* Collect logits belonging to sentence i                  */
 9:      if is_claim(sᵢ) = false then
10:          S ← S ∪ {(sᵢ, level="none", confidence=null)}
11:          continue
12:      end if
13:      μ ← mean(Lᵢ)
14:      confidence ← σ(μ; c=33.0, k=3.0)
         /* Sigmoid: 1 / (1 + exp(-(μ - c) / k))                  */
15:      S ← S ∪ {(sᵢ, level, confidence)}
16:  end for

17:  /* Pass 2: Extract hidden states via separate forward pass     */
18:  H ← model.forward(Q ⊕ A, output_hidden_states=True)
     /* H ∈ ℝ^(33 × seq_len × 4096)                               */
19:  h_slt ← H[:, |Q⊕A| - 2, :]
     /* Second-to-last token hidden state across all layers         */

20:  /* Probe prediction                                            */
21:  (l₀, l₁) ← P.energy_layer_range
22:  X_e ← flatten(h_slt[l₀:l₁])
23:  X_e ← P.energy_scaler.transform(X_e)
24:  energy_risk ← 1.0 - P.energy_probe.predict_proba(X_e)[1]

25:  (l₀', l₁') ← P.entropy_layer_range
26:  X_h ← flatten(h_slt[l₀':l₁'])
27:  X_h ← P.entropy_scaler.transform(X_h)
28:  entropy_risk ← P.entropy_probe.predict_proba(X_h)[1]

29:  /* Aggregate risk based on answer length                       */
30:  if |T| ≤ 100 then
31:      combined_risk ← (energy_risk + entropy_risk) / 2
32:  else
33:      R_sent ← {probe_risk(sᵢ) : sᵢ ∈ S, is_claim(sᵢ) = true}
34:      w_slt ← 0.15
35:      w_max ← 0.25 / (1 + log(|R_sent|))
36:      w_mean ← 1.0 - w_slt - w_max
37:      combined_risk ← w_slt · entropy_risk
                        + w_max · max(R_sent)
                        + w_mean · mean(R_sent)
38:  end if

39:  return R = (energy_risk, entropy_risk, combined_risk), S
```

---

## Algorithm 2: Full Semantic Energy Scoring

```
Algorithm 2: Full Semantic Energy Scoring
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data:    Question Q, Number of samples N = 5
Result:  Confidence score C ∈ [0,1], Cluster count K,
         Sentence analysis S

 1:  /* Generate N diverse responses                                */
 2:  G ← ∅
 3:  for i ← 1 to N do
 4:      (aᵢ, lᵢ, pᵢ) ← model.generate(Q, temperature=0.7)
         /* aᵢ = answer, lᵢ = logits, pᵢ = token probabilities    */
 5:      G ← G ∪ {(aᵢ, lᵢ, pᵢ)}
 6:  end for

 7:  /* Semantic clustering via pairwise LLM verification           */
 8:  visited ← {false}^N
 9:  clusters ← ∅
10:  for i ← 1 to N do
11:      if visited[i] then continue end if
12:      cluster ← {i}
13:      visited[i] ← true
14:      for j ← i + 1 to N do
15:          if not visited[j] then
16:              equiv ← semantic_analyse(Q, aᵢ, aⱼ)
                 /* LLM judges: semantically equivalent?            */
17:              if equiv = true then
18:                  cluster ← cluster ∪ {j}
19:                  visited[j] ← true
20:              end if
21:          end if
22:      end for
23:      clusters ← clusters ∪ {cluster}
24:  end for
25:  K ← |clusters|

26:  /* Compute Semantic Energy per cluster                         */
27:  for each cluster cₖ ∈ clusters do
28:      prob_cₖ ← Σᵢ∈cₖ (∏ⱼ pᵢ[j])
         /* Product of token probs per response, summed per cluster */
29:      energy_cₖ ← -Σᵢ∈cₖ mean(lᵢ)
         /* Negated mean logit per response, summed per cluster     */
30:  end for
31:  E ← normalize(energy_c₁, ..., energy_cₖ)
     /* Sum-normalize to get per-cluster energy shares              */

32:  /* Identify main cluster (contains first response)             */
33:  m ← k where 1 ∈ cₖ
34:  C ← E[m]
     /* Main cluster energy = overall confidence                    */

35:  /* Sentence-level analysis on main answer                      */
36:  S ← score_sentences(a₁, l₁, T₁)
     /* See Algorithm 1, lines 4-16                                 */

37:  return C, K, S
```

---

## Algorithm 3: TBG Pre-Generation Risk Estimation

```
Algorithm 3: TBG Pre-Generation Risk Estimation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data:    Question Q, Probe Bundle P
Result:  Pre-generation risk R_pre ∈ [0,1]

 1:  /* Single forward pass on prompt only — no generation          */
 2:  H ← model.forward(Q, output_hidden_states=True)
     /* H ∈ ℝ^(33 × prompt_len × 4096)                            */
 3:  h_tbg ← H[:, |Q| - 1, :]
     /* Last prompt token hidden state                              */

 4:  /* Energy probe on layers 28-32                                */
 5:  X_e ← flatten(h_tbg[28:32])
 6:  X_e ← P.tbg_energy_scaler.transform(X_e)
 7:  energy_risk ← 1.0 - P.tbg_energy_probe.predict_proba(X_e)[1]

 8:  /* Entropy probe on layers 21-25                               */
 9:  X_h ← flatten(h_tbg[21:25])
10:  X_h ← P.tbg_entropy_scaler.transform(X_h)
11:  entropy_risk ← P.tbg_entropy_probe.predict_proba(X_h)[1]

12:  R_pre ← (energy_risk + entropy_risk) / 2

13:  return R_pre
```

---

## LaTeX Code (for thesis)

Copy this into your thesis `.tex` file. Requires `\usepackage{algorithm2e}` in preamble.

```latex
\usepackage[ruled,vlined,linesnumbered]{algorithm2e}
```

### Algorithm 1 — LaTeX

```latex
\begin{algorithm}[H]
\caption{SLT Post-Generation Hallucination Scoring}
\label{alg:slt}
\KwData{Question $Q$, Probe Bundle $P = \{\text{probes, scalers, layer\_ranges}\}$}
\KwResult{Risk scores $R = (r_e, r_h, r_c)$, Sentence analysis $S$}

\tcp{Pass 1: Generate answer using LLM}
$A, L, T \gets \texttt{model.generate}(Q,\ \text{temp}=0.7,\ \text{max\_tokens}=512)$\;
\tcp{$A$ = answer, $L$ = per-token logits, $T$ = token IDs}

\tcp{Sentence-level scoring from generation logits}
$\textit{sentences} \gets \texttt{split\_sentences}(A)$\;
$\textit{token\_map} \gets \texttt{align\_tokens}(A, T, \textit{sentences})$\;
$S \gets \emptyset$\;
\ForEach{sentence $s_i \in \textit{sentences}$}{
    $L_i \gets \{L[t] : \textit{token\_map}[t] = i\}$\;
    \If{$\texttt{is\_claim}(s_i) = \textit{false}$}{
        $S \gets S \cup \{(s_i,\ \text{level}=\text{``none''},\ \text{conf}=\text{null})\}$\;
        \textbf{continue}\;
    }
    $\mu \gets \text{mean}(L_i)$\;
    $\text{conf} \gets \sigma(\mu;\ c\!=\!33.0,\ k\!=\!3.0) = \frac{1}{1 + e^{-(\mu - c)/k}}$\;
    $S \gets S \cup \{(s_i,\ \text{level},\ \text{conf})\}$\;
}

\tcp{Pass 2: Extract hidden states via separate forward pass}
$H \gets \texttt{model.forward}(Q \oplus A,\ \text{output\_hidden\_states}=\text{True})$\;
\tcp{$H \in \mathbb{R}^{33 \times |Q \oplus A| \times 4096}$}
$\mathbf{h}_{\text{slt}} \gets H[:,\ |Q \oplus A| - 2,\ :]$\;

\tcp{Probe prediction}
$(l_0, l_1) \gets P.\text{energy\_layer\_range}$\;
$X_e \gets P.\text{energy\_scaler.transform}(\text{flatten}(\mathbf{h}_{\text{slt}}[l_0\!:\!l_1]))$\;
$r_e \gets 1.0 - P.\text{energy\_probe.predict\_proba}(X_e)[1]$\;

$(l_0', l_1') \gets P.\text{entropy\_layer\_range}$\;
$X_h \gets P.\text{entropy\_scaler.transform}(\text{flatten}(\mathbf{h}_{\text{slt}}[l_0'\!:\!l_1']))$\;
$r_h \gets P.\text{entropy\_probe.predict\_proba}(X_h)[1]$\;

\tcp{Aggregate risk based on answer length}
\eIf{$|T| \leq 100$}{
    $r_c \gets (r_e + r_h) \mathbin{/} 2$\;
}{
    $R_s \gets \{\text{probe\_risk}(s_i) : s_i \in S,\ \texttt{is\_claim}(s_i)\}$\;
    $r_c \gets 0.15 \cdot r_h + \frac{0.25}{1 + \log|R_s|} \cdot \max(R_s) + w_{\text{mean}} \cdot \text{mean}(R_s)$\;
}

\Return{$R = (r_e,\ r_h,\ r_c),\ S$}
\end{algorithm}
```

### Algorithm 2 — LaTeX

```latex
\begin{algorithm}[H]
\caption{Full Semantic Energy Scoring}
\label{alg:full-se}
\KwData{Question $Q$, Number of samples $N = 5$}
\KwResult{Confidence $C \in [0,1]$, Cluster count $K$, Sentence analysis $S$}

\tcp{Generate $N$ diverse responses}
$G \gets \emptyset$\;
\For{$i \gets 1$ \KwTo $N$}{
    $(a_i, l_i, p_i) \gets \texttt{model.generate}(Q,\ \text{temp}=0.7)$\;
    $G \gets G \cup \{(a_i, l_i, p_i)\}$\;
}

\tcp{Semantic clustering via pairwise LLM verification}
$\textit{visited} \gets \{\textit{false}\}^N$\;
$\textit{clusters} \gets \emptyset$\;
\For{$i \gets 1$ \KwTo $N$}{
    \If{$\textit{visited}[i]$}{\textbf{continue}}
    $\textit{cluster} \gets \{i\}$;\ $\textit{visited}[i] \gets \textit{true}$\;
    \For{$j \gets i + 1$ \KwTo $N$}{
        \If{$\neg\textit{visited}[j]\ \wedge\ \texttt{semantic\_analyse}(Q, a_i, a_j)$}{
            $\textit{cluster} \gets \textit{cluster} \cup \{j\}$;\ $\textit{visited}[j] \gets \textit{true}$\;
        }
    }
    $\textit{clusters} \gets \textit{clusters} \cup \{\textit{cluster}\}$\;
}
$K \gets |\textit{clusters}|$\;

\tcp{Compute Semantic Energy per cluster}
\ForEach{cluster $c_k \in \textit{clusters}$}{
    $E_{c_k} \gets -\sum_{i \in c_k} \text{mean}(l_i)$\;
}
$\mathbf{E} \gets \text{sum\_normalize}(E_{c_1}, \ldots, E_{c_K})$\;

\tcp{Main cluster confidence}
$m \gets k$ where $1 \in c_k$\;
$C \gets \mathbf{E}[m]$\;

\tcp{Sentence-level analysis on main answer}
$S \gets \texttt{score\_sentences}(a_1, l_1, T_1)$ \tcp*{See Algorithm 1}

\Return{$C,\ K,\ S$}
\end{algorithm}
```

### Algorithm 3 — LaTeX

```latex
\begin{algorithm}[H]
\caption{TBG Pre-Generation Risk Estimation}
\label{alg:tbg}
\KwData{Question $Q$, Probe Bundle $P$}
\KwResult{Pre-generation risk $R_{\text{pre}} \in [0,1]$}

\tcp{Single forward pass on prompt only --- no generation}
$H \gets \texttt{model.forward}(Q,\ \text{output\_hidden\_states}=\text{True})$\;
$\mathbf{h}_{\text{tbg}} \gets H[:,\ |Q| - 1,\ :]$ \tcp*{Last prompt token}

\tcp{Energy probe (layers 28--32)}
$X_e \gets P.\text{tbg\_energy\_scaler.transform}(\text{flatten}(\mathbf{h}_{\text{tbg}}[28\!:\!32]))$\;
$r_e \gets 1.0 - P.\text{tbg\_energy\_probe.predict\_proba}(X_e)[1]$\;

\tcp{Entropy probe (layers 21--25)}
$X_h \gets P.\text{tbg\_entropy\_scaler.transform}(\text{flatten}(\mathbf{h}_{\text{tbg}}[21\!:\!25]))$\;
$r_h \gets P.\text{tbg\_entropy\_probe.predict\_proba}(X_h)[1]$\;

$R_{\text{pre}} \gets (r_e + r_h) \mathbin{/} 2$\;

\Return{$R_{\text{pre}}$}
\end{algorithm}
```
