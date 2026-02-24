import os
import sys
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from statistics import mean
from math import prod

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
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        """Load the model in native fp16 on CUDA."""
        print(f"[Engine] Loading model: {model_id}", flush=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Engine] Using device: {self.device}", flush=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[Engine] Tokenizer loaded.", flush=True)

        # Load model directly to the target device (no device_map to avoid offload complexity)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
        ).to(self.device)
        print("[Engine] Model loaded successfully!", flush=True)

    def generate_responses(self, question, num_samples=5):
        """Generates multiple responses one at a time and extracts tokens, probs, and raw logits."""
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        generated_data = []
        
        for i in range(num_samples):
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs.sequences[0][input_len:].tolist()
            
            logits_list = []
            probs_list = []
            token_ids = []
            
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
            
            answer_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            generated_data.append({
                "answer": answer_text,
                "logits": logits_list,
                "probs": probs_list,
                "token_ids": token_ids
            })
            print(f"  [Sample {i+1}/{num_samples}] {answer_text[:80]}...", flush=True)
            
        return generated_data

    def semantic_analyse(self, question, answer_a, answer_b):
        """Uses the LLM to verify if two answers are semantically equivalent."""
        prompt = (
            f"User: ### Question: {question}\n\n"
            f"### Ground Truth Answer: {answer_a}\n\n"
            f"### Student Answer: {answer_b}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        result = self.tokenizer.decode(outputs[0][-10:], skip_special_tokens=True)
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
