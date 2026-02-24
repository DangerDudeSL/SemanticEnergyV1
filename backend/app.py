import sys
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

# Lazy-load engine on first request to avoid crashing during module import
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        print("[App] Importing SemanticEngine...", flush=True)
        from engine import SemanticEngine, cal_flow, sum_normalize
        print("[App] Starting model load...", flush=True)
        engine = SemanticEngine(model_id="Qwen/Qwen2.5-1.5B-Instruct")
        print("[App] Backend is READY to accept requests!", flush=True)
    except Exception as e:
        print(f"[App] FATAL ERROR during startup: {e}", flush=True)
        traceback.print_exc()

@app.post("/chat")
async def chat_endpoint(request: Request):
    from engine import cal_flow, sum_normalize
    
    global engine
    if engine is None:
        return JSONResponse({"error": "Model is still loading, please wait..."}, status_code=503)
    
    try:
        data = await request.json()
        user_prompt = data.get("prompt", "")
        num_samples = data.get("num_samples", 5)
        
        if not user_prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)
            
        print(f"\n[1/3] Generating {num_samples} responses for: {user_prompt}", flush=True)
        generated_data = engine.generate_responses(user_prompt, num_samples=num_samples)
        
        main_answer = generated_data[0]["answer"]
        
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

        return {
            "answer": main_answer,
            "confidence_score": main_confidence,
            "confidence_level": confidence_level,
            "clusters_found": len(clusters),
            "debug_data": {
                "all_answers": answer_texts,
                "energies_per_cluster": cluster_energies
            }
        }
    except Exception as e:
        print(f"[App] ERROR during inference: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
