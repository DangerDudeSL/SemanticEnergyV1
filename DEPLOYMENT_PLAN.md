# SemanticEnergy — Web Deployment Plan

## Primary Architecture: Vercel (Frontend) + Google Colab (Backend) + ngrok (Tunnel)

This is the recommended approach for development, testing, and supervisor demos.
Unlimited GPU time, zero cost, no credit anxiety.

```
┌──────────────────┐                          ┌─────────────────────────┐
│  Vercel (FREE)   │        HTTPS/JSON        │  Google Colab (FREE)    │
│                  │ ──────────────────────>   │                         │
│  index.html      │                          │  FastAPI (app.py)       │
│  script.js       │   ngrok static domain    │  engine.py              │
│  styles.css      │ <──────────────────────  │  claim_filter.py        │
│                  │                          │  probes.pkl (from HF)   │
│  Permanent URL:  │                          │  Llama 3.1 8B on T4    │
│  your-app.       │                          │                         │
│   vercel.app     │                          │  Temp session (12h max) │
└──────────────────┘                          └─────────────────────────┘
        │                                              │
        │  Static files (permanent)                    │  Downloads from:
        │  Auto-deploys from GitHub                    │
        │                                     ┌───────────────────┐
        │                                     │  HuggingFace Hub  │
        │                                     │  (FREE)           │
        │                                     │                   │
        │                                     │  - Llama 3.1 8B   │
        │                                     │  - probes.pkl     │
        │                                     └───────────────────┘
```

---

## Why This Architecture

| Concern | Answer |
|---------|--------|
| GPU cost | $0 — Colab free tier, unlimited runtime (12h sessions) |
| Credit anxiety | None — no credits to burn, just restart if session expires |
| Frontend URL | Permanent — Vercel gives `your-app.vercel.app` for free |
| Backend URL | Static — ngrok free gives 1 permanent domain |
| Your internet usage | Negligible (~5 KB per request, see breakdown below) |
| Code changes needed | 1 line in script.js (API_BASE URL) |

---

## Component Details

### 1. Frontend: Vercel (Free)

**What it hosts:** The static frontend files (HTML/CSS/JS) — no build step needed.

**Setup:**
1. Sign up at vercel.com (use GitHub login)
2. Import the `SemanticEnergy` GitHub repo
3. Set root directory to `frontend/`
4. Deploy — gives you `semantic-energy.vercel.app`

**Limits (free tier):**
- Unlimited deploys
- Auto HTTPS
- Custom domain support (free)
- 100 GB bandwidth/month (more than enough for static files)
- Auto-deploys when you push to GitHub

**Code change (1 line in script.js):**
```javascript
const API_BASE = 'https://your-domain.ngrok-free.dev';
```

### 2. Backend: Google Colab (Free)

**What it runs:** The full backend — FastAPI server, Llama 3.1 8B, probe scoring.

**Resources provided (free tier):**
- T4 GPU (16 GB VRAM) — sufficient for Llama 8B in 8-bit
- ~12 GB RAM
- 12-hour session limit (just restart the notebook)
- Unlimited sessions per day

**How to run:**
1. Open the Colab deployment notebook
2. Run all cells (installs deps, loads model, starts FastAPI + ngrok)
3. Backend is live until session expires or you close it

**What loads at startup (~60-90 seconds):**
```
Colab session starts
    │
    ├─ 1. pip install requirements (~30s, cached after first run)
    │
    ├─ 2. Download Llama 3.1 8B from HuggingFace Hub (~10 GB)
    │     First time: ~2-3 minutes (Colab has fast download)
    │     Subsequent: cached in Colab's session storage
    │     Quantized to 8-bit → loads into T4 GPU VRAM (~9 GB)
    │
    ├─ 3. Download probes.pkl from HuggingFace Hub (2.1 MB)
    │     4 sklearn LogisticRegression classifiers
    │     Loads into CPU RAM (negligible)
    │
    ├─ 4. Start ngrok tunnel → public URL active
    │
    └─ Ready to serve requests from Vercel frontend
```

### 3. Tunnel: ngrok (Free)

**Why ngrok:** Colab doesn't expose ports to the internet. ngrok creates a public URL that forwards traffic to Colab's FastAPI server.

**Free tier includes:**
- 1 static domain (e.g., `abc123xyz.ngrok-free.dev`) — permanent, never changes
- 1 GB bandwidth/month
- 20,000 requests/month

**Is 1 GB / 20k requests enough?**

Each request to our API:
- Request payload: ~200 bytes (JSON with question text)
- Response payload: ~2-5 KB (JSON with answer + sentence scores + metrics)
- Average round trip: ~5 KB

| Usage scenario | Requests | Bandwidth | Within free limit? |
|---------------|----------|-----------|-------------------|
| 10 demos/day for a month | 300 | 1.5 MB | Yes |
| 50 requests/day heavy testing | 1,500 | 7.5 MB | Yes |
| 100 requests/day extreme | 3,000 | 15 MB | Yes |
| Theoretical max (20k requests) | 20,000 | 100 MB | Yes |

**1 GB bandwidth = ~200,000 requests.** The 20k request limit is what you'd hit first, which is ~650 requests/day. More than enough for demos and testing.

**Known limitation:** Free ngrok shows a browser warning page on first visit that users must click through. This only appears when visiting the ngrok URL directly — API calls from your Vercel frontend are NOT affected (they're programmatic fetch requests, not browser navigation).

### 4. Probe Storage: HuggingFace Hub (Free)

**What it stores:** The trained probe pkl files (2.1 MB each).

**Why HF Hub:**
- Already used by the code to download Llama 3.1 8B — same pattern
- Free, unlimited storage for public repos
- When adding new probe models, just upload — no code change needed

**Setup:**
1. Create HF repo: `DangerDudeSL/semantic-energy-probes`
2. Upload `probes_llama3-8b_triviaqa.pkl` (2.1 MB)
3. Future probe bundles go here too (e.g., `probes_qwen-1.5b_triviaqa.pkl`)

**Download in backend code:**
```python
from huggingface_hub import hf_hub_download

probe_path = hf_hub_download(
    "DangerDudeSL/semantic-energy-probes",
    "probes_llama3-8b_triviaqa.pkl"
)
probe_bundle = pickle.load(open(probe_path, "rb"))
```

---

## Your Internet Data Usage

**Short answer: practically zero.**

All heavy computation happens on Google's servers (Colab). Your internet only carries:
1. The Vercel frontend loads (HTML/CSS/JS) — ~60 KB, cached by browser
2. API requests/responses between Vercel and Colab — ~5 KB each

| Activity | Data used |
|----------|-----------|
| Loading the frontend page | ~60 KB (once, then cached) |
| One chat request + response | ~5 KB |
| 100 requests in a day | ~500 KB |
| A full month of heavy testing | ~15 MB |

**The model download (~10 GB) happens inside Colab, NOT on your internet.** Colab downloads from HuggingFace using Google's data center network. Your browser never touches those model files.

---

## Cost Verification — Everything is Free

| Service | What we use | Free tier limit | Our usage | Verified free? |
|---------|-------------|-----------------|-----------|---------------|
| **Vercel** | Static hosting | 100 GB bandwidth/mo | ~60 KB/visit | Yes |
| **Google Colab** | T4 GPU compute | Unlimited (12h sessions) | ~1-4 hours/session | Yes |
| **ngrok** | Tunnel to Colab | 1 GB bandwidth + 20k req/mo | ~100 MB + 3k req/mo | Yes |
| **HuggingFace Hub** | Probe pkl storage | Unlimited (public repos) | 2.1 MB | Yes |
| **Your internet** | Frontend + API calls | Your existing plan | ~15 MB/month | Yes |

**Total monthly cost: $0**

---

## Files to Create/Modify for Deployment

| File | Change | Effort |
|------|--------|--------|
| `frontend/script.js` | Set `API_BASE` to ngrok static domain | 1 line |
| `backend/app.py` | Add HF Hub download for pkl (optional, can keep local) | ~5 lines |
| UPDATE: `notebooks/colab_deploy.ipynb` | Add ngrok static domain setup | ~10 lines |
| `backend/engine.py` | No changes | 0 |
| `backend/claim_filter.py` | No changes | 0 |
| `frontend/index.html` | No changes | 0 |
| `frontend/styles.css` | No changes | 0 |

**Core logic is untouched.** Only the connection layer changes.

---

## Deployment Steps

### Phase 1: ngrok Setup (one-time)
1. Sign up at ngrok.com (free account)
2. Get your authtoken from the dashboard
3. Note your free static domain (e.g., `abc123xyz.ngrok-free.dev`)
4. This domain is yours permanently

### Phase 2: HuggingFace Hub Setup (one-time)
1. Create HF repo `DangerDudeSL/semantic-energy-probes`
2. Upload `probes_llama3-8b_triviaqa.pkl`
3. (Optional) Update `app.py` to download from HF Hub instead of local path

### Phase 3: Vercel Frontend (one-time)
1. Update `script.js` with ngrok static domain as `API_BASE`
2. Sign up at vercel.com, import GitHub repo
3. Set root directory to `frontend/`, deploy
4. Frontend is now live at `semantic-energy.vercel.app`

### Phase 4: Colab Backend (each session)
1. Open the deployment notebook in Colab
2. Run all cells — installs deps, loads model, starts FastAPI
3. ngrok tunnel connects to your static domain automatically
4. Backend is live — Vercel frontend can now reach it

### Phase 5: Verify
1. Open your Vercel URL in browser
2. Send a test question in each mode (Full SE, SLT, TBG)
3. Verify sentence highlighting, probe scores, and metrics display

---

## Session Workflow (day-to-day usage)

```
You want to demo or test:
    │
    ├─ 1. Open Colab notebook → Run All (~60-90s to load model)
    │
    ├─ 2. Open your-app.vercel.app in browser (already deployed)
    │
    ├─ 3. Use normally — unlimited requests while Colab is running
    │
    ├─ 4. Done? Just close the Colab tab
    │
    └─ Next time? Repeat from step 1 (frontend stays live 24/7)
```

**The frontend is always online.** Only the backend needs manual start per session.
If someone visits while Colab is down, the frontend loads fine but API calls fail gracefully.

---

## Alternative Backend Options

If Colab becomes insufficient in the future:

| Option | GPU | Cost | Persistent URL | When to use |
|--------|-----|------|---------------|-------------|
| **Google Colab + ngrok** | T4 (16GB) | Free | Yes (static domain) | Current: dev + demos |
| **Modal.com** | T4 (16GB) | $30 free/mo | Yes | If you need always-on API |
| **HF ZeroGPU** | H200 (70GB) | Free | Yes | If you rewrite to Gradio |
| **Kaggle + ngrok** | P100 (16GB) | Free | Yes | Backup if Colab is slow |
| **Lightning AI** | Various | 22 hrs/mo | Partial | Alternative notebook env |

---

## Future: Adding New Models

1. Train new probes using the notebooks (e.g., for Qwen or Mistral)
2. Upload new pkl to HF Hub: `probes_qwen-1.5b_triviaqa.pkl`
3. Update backend to pick the right probe bundle based on active model
4. No frontend changes needed — model dropdown already exists
