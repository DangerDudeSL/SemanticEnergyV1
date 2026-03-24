import sys
import gc
import os
import pickle
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Semantic Energy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for the engine, the currently loaded model string, and the probe bundle
engine = None
current_model_id = None
probe_bundle = None
loading_model_id = None   # set while a model is being loaded

PROBE_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "models", "probes_llama3-8b_triviaqa.pkl")

@app.on_event("startup")
async def startup_event():
    global engine, current_model_id, probe_bundle, loading_model_id
    try:
        current_model_id = "meta-llama/Llama-3.1-8B-Instruct"
        loading_model_id = current_model_id
        print("[App] Importing SemanticEngine...", flush=True)
        from engine import SemanticEngine, cal_flow, sum_normalize
        print(f"[App] Starting initial model load: {current_model_id}...", flush=True)
        engine = SemanticEngine(model_id=current_model_id)
        loading_model_id = None
        print("[App] Backend is READY to accept requests!", flush=True)

        # Load probe bundle if available (needed for /score_fast_* endpoints)
        if os.path.exists(PROBE_BUNDLE_PATH):
            with open(PROBE_BUNDLE_PATH, "rb") as f:
                probe_bundle = pickle.load(f)
            print(f"[App] Probe bundle loaded from {PROBE_BUNDLE_PATH}", flush=True)
        else:
            print(f"[App] No probe bundle at {PROBE_BUNDLE_PATH} — fast scoring unavailable.", flush=True)

    except Exception as e:
        print(f"[App] FATAL ERROR during startup: {e}", flush=True)
        traceback.print_exc()

@app.get("/status")
async def status_endpoint():
    """Returns the current readiness state of the backend."""
    if engine is not None:
        return {"ready": True, "model_id": current_model_id}
    return {
        "ready": False,
        "loading_model_id": loading_model_id or current_model_id or "unknown",
    }

@app.post("/switch_model")
async def switch_model_endpoint(request: Request):
    """Switch the loaded model. Returns when the new model is ready."""
    from engine import SemanticEngine
    import torch

    global engine, current_model_id, loading_model_id
    try:
        data = await request.json()
        requested_model = data.get("model_id", "")
        if not requested_model:
            return JSONResponse({"error": "model_id is required"}, status_code=400)
        if requested_model == current_model_id and engine is not None:
            return {"status": "already_loaded", "model_id": current_model_id}

        loading_model_id = requested_model
        print(f"\n[App] Model switch requested: {current_model_id} -> {requested_model}", flush=True)

        if engine is not None:
            print("[App] Unloading current model from VRAM...", flush=True)
            del engine.model
            del engine.tokenizer
            del engine
            engine = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"[App] Loading new model: {requested_model}...", flush=True)
        current_model_id = requested_model
        engine = SemanticEngine(model_id=current_model_id)
        loading_model_id = None
        print(f"[App] Model {current_model_id} is READY!", flush=True)
        return {"status": "loaded", "model_id": current_model_id}

    except Exception as e:
        loading_model_id = None
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat")
async def chat_endpoint(request: Request):
    from engine import cal_flow, sum_normalize, SemanticEngine
    import torch
    
    global engine, current_model_id, loading_model_id
    if engine is None:
        return JSONResponse({"error": "Model is still loading, please wait..."}, status_code=503)

    try:
        data = await request.json()
        user_prompt = data.get("prompt", "")
        num_samples = data.get("num_samples", 5)
        requested_model = data.get("model_id", current_model_id)

        # Handle dynamic model switching
        if requested_model != current_model_id:
            print(f"\n[App] Model switch requested: {current_model_id} -> {requested_model}", flush=True)
            print("[App] Unloading current model from VRAM...", flush=True)
            loading_model_id = requested_model
            del engine.model
            del engine.tokenizer
            del engine
            engine = None

            # Force garbage collection and CUDA cache clear to free the VRAM
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            print(f"[App] Loading new model: {requested_model}...", flush=True)
            current_model_id = requested_model
            engine = SemanticEngine(model_id=current_model_id)
            loading_model_id = None
            print(f"[App] Successfully swapped to {current_model_id}!", flush=True)
        
        if not user_prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)
            
        print(f"\n[1/3] Generating {num_samples} responses for: {user_prompt}", flush=True)
        generated_data = engine.generate_responses(user_prompt, num_samples=num_samples)
        
        main_answer = generated_data[0]["answer"]

        # B1 sentence-level logit confidence for the main answer
        sentence_scores = engine.score_sentences(
            main_answer, generated_data[0]["token_ids"],
            generated_data[0]["logits"], generated_data[0].get("top2_logits"),
        )

        print(f"[2/3] Deciding semantic equivalent clusters...", flush=True)
        answer_texts = [d["answer"] for d in generated_data]
        clusters = engine.find_semantic_clusters(user_prompt, answer_texts)
        print(f"Found {len(clusters)} clusters: {clusters}", flush=True)
        
        print(f"[3/3] Calculating Semantic Energy / Uncertainty...", flush=True)
        probs_list = [d['probs'] for d in generated_data]
        logits_list = [d['logits'] for d in generated_data]
        
        probs_se, logits_se = cal_flow(probs_list, logits_list, clusters, fermi_mu=None)
        
        main_cluster_idx = 0
        for idx, cluster in enumerate(clusters):
            if 0 in cluster:
                main_cluster_idx = idx
                break
                
        cluster_energies = sum_normalize(logits_se)
        main_confidence = cluster_energies[main_cluster_idx]
        
        if main_confidence > 0.80:
            confidence_level = "high"
        elif main_confidence > 0.50:
            confidence_level = "medium"
        else:
            confidence_level = "low"
            
        print(f"Main answer confidence: {main_confidence:.2f} ({confidence_level})", flush=True)

        # Sentence-averaged confidence (claim sentences only)
        valid_confs = [s["confidence"] for s in sentence_scores
                       if s["confidence"] is not None and s.get("is_claim", True)]
        sentence_avg_confidence = float(sum(valid_confs) / len(valid_confs)) if valid_confs else None

        return {
            "answer": main_answer,
            "confidence_score": main_confidence,
            "confidence_level": confidence_level,
            "clusters_found": len(clusters),
            "sentence_scores": sentence_scores,
            "sentence_avg_confidence": round(sentence_avg_confidence, 4) if sentence_avg_confidence is not None else None,
            "debug_data": {
                "all_answers": answer_texts,
                "energies_per_cluster": cluster_energies
            }
        }
    except Exception as e:
        print(f"[App] ERROR during inference: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/score_fast_tbg")
async def score_fast_tbg(request: Request):
    """
    TBG mode: pre-generation risk estimation using a trained linear probe.
    Single forward pass on the prompt only — no answer generated, no clustering.
    Requires probe bundle to be loaded (run notebooks/02_train_se_probes.ipynb first).
    """
    global engine, probe_bundle
    if engine is None:
        return JSONResponse({"error": "Model is still loading, please wait..."}, status_code=503)
    if probe_bundle is None:
        return JSONResponse(
            {"error": "Probe bundle not found. Run notebooks/02_train_se_probes.ipynb first."},
            status_code=503
        )
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)

        result = engine.score_with_tbg_probe(prompt, probe_bundle)
        return result

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/score_fast_slt")
async def score_fast_slt(request: Request):
    """
    SLT mode: post-generation confidence score using a trained linear probe.
    Generates one answer then runs one additional forward pass — no clustering.
    Requires probe bundle to be loaded (run notebooks/02_train_se_probes.ipynb first).
    """
    global engine, probe_bundle
    if engine is None:
        return JSONResponse({"error": "Model is still loading, please wait..."}, status_code=503)
    if probe_bundle is None:
        return JSONResponse(
            {"error": "Probe bundle not found. Run notebooks/02_train_se_probes.ipynb first."},
            status_code=503
        )
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)

        result = engine.score_with_slt_probe(prompt, probe_bundle)
        return result

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
