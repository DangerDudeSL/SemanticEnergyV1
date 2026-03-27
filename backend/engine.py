import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from statistics import mean
from math import prod
import pysbd
from claim_filter import ClaimFilter

# Helper Math Functions (from semantic_energy.ipynb)
def sum_normalize(lst):
    total = sum(lst)
    return [x / total if total != 0 else 0 for x in lst]

def cal_cluster_ce(probs, logits, clusters):
    probs_se = []
    logits_se = []
    normalized_probs = sum_normalize(probs)
    for cluster in clusters:
        cluster_prob_sum = sum(normalized_probs[i] for i in cluster)
        probs_se.append(cluster_prob_sum)
        cluster_logit_sum = -sum(logits[i] for i in cluster)
        logits_se.append(cluster_logit_sum)
    return probs_se, logits_se

def cal_probs(probs_list):
    return [prod(sublist) for sublist in probs_list]

def fermi_dirac(E, mu=0.0, kT=1.0):
    return E / (math.exp((E - mu) / kT) + 1)

def cal_fermi_dirac_logits(logits_list, mu=0.0):
    return [-mean([fermi_dirac(logit, mu) for logit in sublist]) for sublist in logits_list]

def cal_boltzmann_logits(logits_list):
    return [-mean(sublist) for sublist in logits_list]

def cal_flow(probs_list, logits_list, clusters, fermi_mu=None):
    probs = cal_probs(probs_list)
    if fermi_mu is not None:
        logits = cal_fermi_dirac_logits(logits_list, mu=fermi_mu)
    else:
        logits = cal_boltzmann_logits(logits_list)
    return cal_cluster_ce(probs, logits, clusters)

# Semantic Engine Class
class SemanticEngine:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct", use_8bit=True):
        """Load the model on CUDA using BitsAndBytes for 8-bit quantization. Requires CUDA."""
        print(f"[Engine] Loading model: {model_id}", flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "[Engine] FATAL: No CUDA-capable GPU detected. "
                "This application requires an NVIDIA GPU with CUDA support. "
                "CPU inference is not supported."
            )

        self.device = torch.device("cuda:0")
        print(f"[Engine] Using device: {self.device} ({torch.cuda.get_device_name(0)})", flush=True)
        print(f"[Engine] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)

        print(f"[Engine] Quantization enabled (8-bit: {use_8bit})", flush=True)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            load_in_4bit=not use_8bit,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[Engine] Tokenizer loaded.", flush=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )

        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"[Engine] Model loaded. VRAM used: {vram_used:.1f} GB", flush=True)
        if vram_used < 0.1:
            raise RuntimeError("[Engine] FATAL: VRAM usage near zero after model load — quantization likely failed. Check bitsandbytes CUDA installation.")

        self._sentence_segmenter = pysbd.Segmenter(language='en', clean=False)
        self._claim_filter = ClaimFilter()

    @staticmethod
    def _safe_apply_chat_template(tokenizer, messages, **kwargs):
        """Apply chat template with enable_thinking=False for Qwen3, with fallback for other models."""
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    def generate_responses(self, question, num_samples=5):
        """Generates multiple responses one at a time and extracts tokens, probs, and raw logits."""
        messages = [
            {"role": "system", "content": "Answer the question directly and concisely. Do not explain your reasoning."},
            {"role": "user", "content": question},
        ]
        prompt = self._safe_apply_chat_template(self.tokenizer, messages, tokenize=False, add_generation_prompt=True)
        
        generated_data = []
        
        for i in range(num_samples):
            # Move inputs to the correct device (the model handles its own quantized placement)
            inputs = self.tokenizer(prompt, return_tensors='pt').to("cuda:0")
            
            with torch.no_grad():
                gen_cfg = GenerationConfig(
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,   # raw unfiltered logits for top2 margin
                )
                
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs.sequences[0][input_len:].tolist()
            
            logits_list = []
            probs_list = []
            token_ids = []
            top2_logits_list = []

            for step_idx, score_tensor in enumerate(outputs.scores):
                if step_idx >= len(gen_ids):
                    break
                
                token_id = gen_ids[step_idx]
                if token_id == self.tokenizer.eos_token_id:
                    break
                
                logits = score_tensor[0]
                prob = F.softmax(logits, dim=-1)[token_id].item()
                logit_val = logits[token_id].item()
                
                logits_list.append(logit_val)
                probs_list.append(prob)
                token_ids.append(token_id)

                # Capture top-2 raw logit values for margin features.
                # outputs.logits[step_idx] has unfiltered logits (no -inf from top_k/top_p masking).
                # outputs.scores[step_idx] is post-filtered — top2 is often -inf there.
                raw_logit_vec = outputs.logits[step_idx][0]
                top2_vals = raw_logit_vec.topk(2).values
                top2_logits_list.append((top2_vals[0].item(), top2_vals[1].item()))

            answer_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            generated_data.append({
                "answer": answer_text,
                "logits": logits_list,
                "probs": probs_list,
                "token_ids": token_ids,
                "top2_logits": top2_logits_list,
            })
            print(f"  [Sample {i+1}/{num_samples}] {answer_text[:80]}...", flush=True)
            
        return generated_data

    # ── B1 Sentence-Level Logit Confidence ────────────────────────────────────

    def split_into_sentences(self, text):
        """Split text into sentences using pysbd. Returns list of non-empty stripped strings."""
        sentences = self._sentence_segmenter.segment(text)
        return [s.strip() for s in sentences if s.strip()]

    def align_tokens_to_sentences(self, answer_text, token_ids, sentences):
        """
        Map each token to its parent sentence index via character-span midpoint overlap.
        Uses self.tokenizer for offset mapping.
        Returns: list[int] of length min(len(token_ids), len(offset_mapping)).
        """
        encoding = self.tokenizer(answer_text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = encoding['offset_mapping']

        # Build sentence character spans via sequential find
        sent_spans = []
        pos = 0
        for sent in sentences:
            start = answer_text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            sent_spans.append((start, end))
            pos = end

        token_sentence_idx = []
        n = min(len(token_ids), len(offset_mapping))
        for tok_pos in range(n):
            char_start, char_end = offset_mapping[tok_pos]
            char_mid = (char_start + char_end) // 2
            sent_idx = len(sentences) - 1  # default to last sentence
            for si, (ss, se) in enumerate(sent_spans):
                if ss <= char_mid < se:
                    sent_idx = si
                    break
            token_sentence_idx.append(sent_idx)

        return token_sentence_idx

    def score_sentences(self, answer_text, token_ids, logits, top2_logits):
        """
        B1 sentence-level logit confidence baseline.
        No additional model calls — purely post-processing of existing per-token logit data.

        Returns list[dict] with keys: text, confidence, level, num_tokens,
        mean_chosen_logit, mean_logit_margin.
        Returns empty list if answer has fewer than 2 sentences.
        """
        sentences = self.split_into_sentences(answer_text)
        if len(sentences) < 2:
            return []

        token_sentence_idx = self.align_tokens_to_sentences(answer_text, token_ids, sentences)
        n_sents = len(sentences)
        n_tokens = min(len(token_sentence_idx), len(logits))

        # Group tokens by sentence
        sent_logits = [[] for _ in range(n_sents)]
        sent_margins = [[] for _ in range(n_sents)]

        for i in range(n_tokens):
            si = token_sentence_idx[i]
            if 0 <= si < n_sents:
                sent_logits[si].append(logits[i])
                if top2_logits is not None and i < len(top2_logits):
                    t1, t2 = top2_logits[i]
                    margin = t1 - t2
                    if np.isfinite(margin):
                        sent_margins[si].append(margin)

        # Sigmoid normalization on absolute logit values
        # Calibrated for Llama 3.1 8B (4-bit quantized): raw chosen-token logits ~ 25–45
        LOGIT_SIGMOID_CENTER = 33.0  # logit value that maps to 50% confidence
        LOGIT_SIGMOID_SCALE = 3.0    # steepness of sigmoid curve

        results = []
        for si, sent in enumerate(sentences):
            sl = sent_logits[si]
            sm = sent_margins[si]

            is_claim = self._claim_filter.is_claim(sent)

            if len(sl) == 0:
                results.append({
                    "text": sent, "confidence": None, "level": "none" if not is_claim else "unknown",
                    "num_tokens": 0, "mean_chosen_logit": None, "mean_logit_margin": None,
                    "is_claim": is_claim,
                })
                continue

            mean_logit = float(np.mean(sl))
            mean_margin = float(np.mean(sm)) if sm else None

            # Sigmoid: maps absolute logit to 0-1 confidence
            confidence = 1.0 / (1.0 + math.exp(-(mean_logit - LOGIT_SIGMOID_CENTER) / LOGIT_SIGMOID_SCALE))

            # Boost confidence slightly when logit margin is large (model very certain)
            if mean_margin is not None and mean_margin > 3.0:
                margin_boost = min(0.1, (mean_margin - 3.0) * 0.02)
                confidence = min(1.0, confidence + margin_boost)

            if confidence >= 0.6:
                level = "high"
            elif confidence >= 0.3:
                level = "medium"
            else:
                level = "low"

            # Non-claim sentences get level "none" and no confidence scoring
            if not is_claim:
                level = "none"
                confidence = None

            results.append({
                "text": sent,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "level": level,
                "num_tokens": len(sl),
                "mean_chosen_logit": round(mean_logit, 4),
                "mean_logit_margin": round(mean_margin, 4) if mean_margin is not None else None,
                "is_claim": is_claim,
            })

        return results

    def semantic_analyse(self, question, answer_a, answer_b):
        """Uses the LLM to verify if two answers are semantically equivalent."""
        
        # We construct a clean ChatML-style list so Llama 3 handles it natively via its chat templates
        messages = [
            {"role": "system", "content": "You verify if two answers are semantically equivalent. Output only 'Final Decision: Yes' or 'Final Decision: No'."},
            {"role": "user", "content": (
                f"### Question: {question}\n\n"
                f"### Ground Truth Answer: {answer_a}\n\n"
                f"### Student Answer: {answer_b}\n\n"
                "Verify if the student's answer is equivalent to the ground truth. Do not solve the question yourself. "
                "If correct, simply output \"Final Decision: Yes\". If incorrect, output \"Final Decision: No\"."
            )}
        ]
        
        prompt = self._safe_apply_chat_template(self.tokenizer, messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            
        # Extract only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        result_tokens = outputs[0][input_len:]
        result = self.tokenizer.decode(result_tokens, skip_special_tokens=True)
        
        return "Yes" in result

    def find_semantic_clusters(self, question, answer_list):
        n = len(answer_list)
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
            clusters.append(tuple(cluster))
        return [list(c) for c in clusters]

    def _extract_hidden_states(self, question, answer_text, extra_positions=None):
        """
        Run a separate forward pass on (prompt + answer) to extract hidden states.
        Returns (tbg_hidden, slt_hidden, extra_hiddens):
            tbg_hidden: (num_layers, hidden_dim) at last prompt token.
            slt_hidden: (num_layers, hidden_dim) at second-to-last answer token.
            extra_hiddens: list of (num_layers, hidden_dim) at each extra_positions index,
                           or [] if extra_positions is None.
        extra_positions are token indices relative to the answer start (0-based).
        Returns (None, None, []) if the answer is too short.
        """
        # Use user-only template to match probe training data (no system prompt)
        messages = [
            {"role": "user", "content": question},
        ]
        prompt_only = self._safe_apply_chat_template(self.tokenizer, messages, tokenize=False, add_generation_prompt=True)

        prompt_ids = self.tokenizer(prompt_only, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        full_text = prompt_only + answer_text
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda:0")
        full_len = full_inputs.input_ids.shape[1]

        if full_len <= prompt_len + 1:
            return None, None, []

        with torch.no_grad():
            outputs = self.model(**full_inputs, output_hidden_states=True)

        # hidden_states: tuple of (1, seq_len, hidden_dim), one per layer incl. embedding
        hidden = torch.stack(outputs.hidden_states, dim=0)  # (num_layers, 1, seq_len, hidden_dim)
        hidden = hidden[:, 0, :, :].float().cpu()            # (num_layers, seq_len, hidden_dim)

        tbg_hidden = hidden[:, prompt_len - 1, :].numpy()   # (num_layers, hidden_dim)
        slt_hidden = hidden[:, full_len - 2, :].numpy()     # (num_layers, hidden_dim)

        # Extract hidden states at sentence-boundary positions
        extra_hiddens = []
        if extra_positions:
            for pos in extra_positions:
                abs_pos = prompt_len + pos  # convert answer-relative to absolute
                if 0 <= abs_pos < full_len:
                    extra_hiddens.append(hidden[:, abs_pos, :].numpy())
                else:
                    extra_hiddens.append(None)

        del outputs, hidden
        torch.cuda.empty_cache()

        return tbg_hidden, slt_hidden, extra_hiddens

    def score_with_tbg_probe(self, question, probe_bundle):
        """
        TBG mode: pre-generation risk estimation.
        Runs a single forward pass on the prompt only. No generation. No clustering.

        Returns dict with:
            mode: "tbg_pre_generation"
            energy_risk: float in [0,1] — hallucination risk from energy probe
            entropy_risk: float in [0,1] — hallucination risk from entropy probe
            confidence_level: "high" | "medium" | "low"
        """
        # Use user-only template to match probe training data (no system prompt)
        messages = [
            {"role": "user", "content": question},
        ]
        prompt_text = self._safe_apply_chat_template(self.tokenizer, messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = torch.stack(outputs.hidden_states, dim=0)  # (num_layers, 1, seq_len, hidden_dim)
        hidden = hidden[:, 0, :, :].float().cpu()
        tbg_hidden = hidden[:, prompt_len - 1, :].numpy()   # (num_layers, hidden_dim)
        del outputs, hidden
        torch.cuda.empty_cache()

        l0, l1 = probe_bundle["best_energy_tbg_range"]
        X_tbg = tbg_hidden[l0:l1, :].reshape(1, -1)

        # Energy probe (confidence): invert for risk
        X_e = probe_bundle["tbg_energy_scaler"].transform(X_tbg)
        energy_confidence = probe_bundle["tbg_energy_probe"].predict_proba(X_e)[0, 1]
        energy_risk = 1.0 - energy_confidence

        # Entropy probe (uncertainty = risk directly)
        l0h, l1h = probe_bundle["best_entropy_tbg_range"]
        X_tbg_h = tbg_hidden[l0h:l1h, :].reshape(1, -1)
        X_h = probe_bundle["tbg_entropy_scaler"].transform(X_tbg_h)
        entropy_risk = probe_bundle["tbg_entropy_probe"].predict_proba(X_h)[0, 1]

        combined_risk = (energy_risk + entropy_risk) / 2.0
        if combined_risk < 0.35:
            level = "high"
        elif combined_risk < 0.65:
            level = "medium"
        else:
            level = "low"

        return {
            "mode": "tbg_pre_generation",
            "energy_risk": float(energy_risk),
            "entropy_risk": float(entropy_risk),
            "combined_risk": float(combined_risk),
            "confidence_level": level,
        }

    def score_with_slt_probe(self, question, probe_bundle):
        """
        SLT mode: post-generation confidence score.
        Generates one answer then runs one additional forward pass on (prompt + answer).
        No clustering.

        Returns dict with:
            mode: "slt_post_generation"
            answer: str
            energy_risk: float in [0,1]
            entropy_risk: float in [0,1]
            confidence_level: "high" | "medium" | "low"
        """
        gen_data = self.generate_responses(question, num_samples=1)
        answer_text = gen_data[0]["answer"]
        token_ids = gen_data[0]["token_ids"]

        # B1 sentence-level logit confidence (post-processing, no extra model calls)
        sentence_scores = self.score_sentences(
            answer_text, token_ids,
            gen_data[0]["logits"], gen_data[0].get("top2_logits"),
        )

        # Find the last token position of each sentence (for per-sentence probe scoring)
        sent_end_positions = []
        if sentence_scores:
            sentences = [s["text"] for s in sentence_scores]
            token_sentence_idx = self.align_tokens_to_sentences(answer_text, token_ids, sentences)
            # For each sentence, find the last token that belongs to it
            for si in range(len(sentences)):
                last_pos = -1
                for tok_pos, sent_idx in enumerate(token_sentence_idx):
                    if sent_idx == si:
                        last_pos = tok_pos
                sent_end_positions.append(last_pos if last_pos >= 0 else None)

        # Extract hidden states: SLT, TBG, plus per-sentence-end positions (claims only)
        valid_positions = [
            p for si, p in enumerate(sent_end_positions)
            if p is not None and sentence_scores[si].get("is_claim", True)
        ]
        tbg_hidden, slt_hidden, sent_hiddens = self._extract_hidden_states(
            question, answer_text,
            extra_positions=valid_positions if valid_positions else None,
        )

        if slt_hidden is None:
            valid_confs = [s["confidence"] for s in sentence_scores if s["confidence"] is not None]
            sa_conf = float(np.mean(valid_confs)) if valid_confs else None
            return {
                "mode": "slt_post_generation",
                "answer": answer_text,
                "energy_risk": 0.5,
                "entropy_risk": 0.5,
                "combined_risk": 0.5,
                "confidence_level": "medium",
                "sentence_scores": sentence_scores,
                "sentence_avg_confidence": round(sa_conf, 4) if sa_conf is not None else None,
                "error": "answer too short for SLT extraction",
            }

        # Overall SLT probe scores
        l0, l1 = probe_bundle["best_energy_slt_range"]
        X_slt_e = slt_hidden[l0:l1, :].reshape(1, -1)
        X_e = probe_bundle["slt_energy_scaler"].transform(X_slt_e)
        energy_confidence = probe_bundle["slt_energy_probe"].predict_proba(X_e)[0, 1]
        energy_risk = 1.0 - energy_confidence

        l0h, l1h = probe_bundle["best_entropy_slt_range"]
        X_slt_h = slt_hidden[l0h:l1h, :].reshape(1, -1)
        X_h = probe_bundle["slt_entropy_scaler"].transform(X_slt_h)
        entropy_risk = probe_bundle["slt_entropy_probe"].predict_proba(X_h)[0, 1]

        # Per-sentence dual-probe scoring (energy + entropy) on sentence-end hidden states
        # Only run on claim sentences (non-claims were excluded from valid_positions)
        # AUROC-weighted combination: entropy (0.773) gets 51.5%, energy (0.727) gets 48.5%
        W_ENTROPY = 0.515
        W_ENERGY = 0.485

        if sent_hiddens and sentence_scores:
            valid_idx = 0
            for si, pos in enumerate(sent_end_positions):
                # Skip non-claim sentences entirely
                if not sentence_scores[si].get("is_claim", True):
                    sentence_scores[si]["energy_risk"] = None
                    sentence_scores[si]["entropy_risk"] = None
                    sentence_scores[si]["probe_risk"] = None
                    continue
                if pos is None or valid_idx >= len(sent_hiddens):
                    sentence_scores[si]["energy_risk"] = None
                    sentence_scores[si]["entropy_risk"] = None
                    sentence_scores[si]["probe_risk"] = None
                    continue
                h = sent_hiddens[valid_idx]
                valid_idx += 1
                if h is None:
                    sentence_scores[si]["energy_risk"] = None
                    sentence_scores[si]["entropy_risk"] = None
                    sentence_scores[si]["probe_risk"] = None
                    continue

                # Energy probe (layers l0:l1) — confidence output, invert for risk
                try:
                    X_e = h[l0:l1, :].reshape(1, -1)
                    X_e_scaled = probe_bundle["slt_energy_scaler"].transform(X_e)
                    sent_energy_risk = 1.0 - float(probe_bundle["slt_energy_probe"].predict_proba(X_e_scaled)[0, 1])
                except Exception:
                    sent_energy_risk = None

                # Entropy probe (layers l0h:l1h) — risk directly
                try:
                    X_h = h[l0h:l1h, :].reshape(1, -1)
                    X_h_scaled = probe_bundle["slt_entropy_scaler"].transform(X_h)
                    sent_entropy_risk = float(probe_bundle["slt_entropy_probe"].predict_proba(X_h_scaled)[0, 1])
                except Exception:
                    sent_entropy_risk = None

                # Store individual probe risks
                sentence_scores[si]["energy_risk"] = round(sent_energy_risk, 4) if sent_energy_risk is not None else None
                sentence_scores[si]["entropy_risk"] = round(sent_entropy_risk, 4) if sent_entropy_risk is not None else None

                # Combined risk: AUROC-weighted average
                if sent_energy_risk is not None and sent_entropy_risk is not None:
                    probe_risk = W_ENTROPY * sent_entropy_risk + W_ENERGY * sent_energy_risk
                elif sent_entropy_risk is not None:
                    probe_risk = sent_entropy_risk
                elif sent_energy_risk is not None:
                    probe_risk = sent_energy_risk
                else:
                    probe_risk = None

                sentence_scores[si]["probe_risk"] = round(probe_risk, 4) if probe_risk is not None else None

                # Override level using combined probe risk
                if probe_risk is not None:
                    if probe_risk >= 0.65:
                        sentence_scores[si]["level"] = "low"
                    elif probe_risk >= 0.35:
                        sentence_scores[si]["level"] = "medium"
                    # If probe says low risk, keep existing logit-based level

        # ── Aggregate scoring: token-length conditional ──────────────────────
        TOKEN_THRESHOLD = 100  # matches probe training distribution (TriviaQA short answers)
        answer_token_count = len(token_ids)

        if answer_token_count <= TOKEN_THRESHOLD:
            slt_combined = (energy_risk + entropy_risk) / 2.0

            per_sent_risks = [s["probe_risk"] for s in sentence_scores
                              if s.get("probe_risk") is not None and s.get("is_claim", True)]

            if len(per_sent_risks) >= 2:
                # 2+ claim sentences: SLT token only captures state at the end,
                # so blend with per-sentence probes to represent all claims
                mean_sent_risk = sum(per_sent_risks) / len(per_sent_risks)
                combined_risk = 0.5 * slt_combined + 0.5 * mean_sent_risk
                print(f"[SLT] Short answer ({answer_token_count} tok, {len(per_sent_risks)} claims): "
                      f"blended (slt={slt_combined:.3f}, sent_mean={mean_sent_risk:.3f}) -> {combined_risk:.3f}", flush=True)
            else:
                # 0-1 claim sentences: SLT probe covers the whole answer adequately
                combined_risk = slt_combined
                print(f"[SLT] Short answer ({answer_token_count} tok, {len(per_sent_risks)} claims): "
                      f"SLT-direct -> {combined_risk:.3f}", flush=True)
        else:
            # LONG ANSWER: SLT unreliable, rely heavily on per-sentence probe risks
            per_sent_risks = [s["probe_risk"] for s in sentence_scores
                              if s.get("probe_risk") is not None and s.get("is_claim", True)]

            if per_sent_risks:
                n = len(per_sent_risks)
                max_sent_risk = max(per_sent_risks)
                mean_sent_risk = sum(per_sent_risks) / n

                # Length-adaptive weights
                slt_weight = 0.15
                max_weight = 0.25 / (1.0 + math.log(max(n, 1)))
                mean_weight = 1.0 - slt_weight - max_weight

                combined_risk = (slt_weight * entropy_risk
                               + max_weight * max_sent_risk
                               + mean_weight * mean_sent_risk)
                print(f"[SLT] Long answer ({answer_token_count} tokens > {TOKEN_THRESHOLD}): "
                      f"blended aggregate (n={n} claims, weights: slt={slt_weight:.2f}, "
                      f"max={max_weight:.2f}, mean={mean_weight:.2f})", flush=True)
            else:
                # Fallback if no per-sentence probe data available
                combined_risk = (energy_risk + entropy_risk) / 2.0
                print(f"[SLT] Long answer but no per-sentence probe data: using SLT-direct fallback", flush=True)

        if combined_risk < 0.35:
            level = "high"
        elif combined_risk < 0.65:
            level = "medium"
        else:
            level = "low"

        # Sentence-averaged confidence (claim sentences only)
        valid_confs = [s["confidence"] for s in sentence_scores
                       if s["confidence"] is not None and s.get("is_claim", True)]
        sentence_avg_confidence = float(np.mean(valid_confs)) if valid_confs else None

        return {
            "mode": "slt_post_generation",
            "answer": answer_text,
            "energy_risk": float(energy_risk),
            "entropy_risk": float(entropy_risk),
            "combined_risk": float(combined_risk),
            "confidence_level": level,
            "sentence_scores": sentence_scores,
            "sentence_avg_confidence": round(sentence_avg_confidence, 4) if sentence_avg_confidence is not None else None,
        }
