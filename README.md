# SemanticEnergy

A real-time LLM hallucination detection system based on the **Semantic Energy** framework — an approach that goes beyond traditional entropy-based methods by using **logit-space energy functions** to quantify uncertainty.

> **Paper**: [Semantic Energy: Detecting LLM Hallucination Beyond Entropy](https://arxiv.org/abs/2508.14496)

![Framework](semanticenergy.png)

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or higher |
| **GPU (recommended)** | NVIDIA GPU with CUDA 12.4 — the model runs on CPU too, but will be significantly slower |
| **VRAM** | ~4 GB minimum (Qwen 2.5 1.5B in fp16) |
| **Disk Space** | ~5 GB (model weights are downloaded on first run) |

### Project Structure

```
SemanticEnergy/
├── backend/
│   ├── app.py              # FastAPI server with /chat endpoint
│   └── engine.py           # SemanticEngine: generation, clustering, energy calculation
├── frontend/
│   ├── index.html           # Chat UI
│   ├── script.js            # Frontend logic
│   └── styles.css           # Styling
├── semantic_energy.ipynb     # Research notebook with full pipeline
├── executed_semantic_energy.ipynb  # Pre-executed notebook with outputs
├── start.ps1                # One-command launcher (Windows/PowerShell)
├── setup.bat                # Environment setup (Windows)
├── setup.sh                 # Environment setup (Linux/macOS)
├── requirements.txt         # Python dependencies
├── test_endpoint.py         # Quick API test script
└── README.md
```

### Quick Setup (Automated)

**Windows:**
```cmd
git clone https://github.com/DangerDudeSL/SemanticEnergyV1.git
cd SemanticEnergyV1
setup.bat
```

**Linux / macOS:**
```bash
git clone https://github.com/DangerDudeSL/SemanticEnergyV1.git
cd SemanticEnergyV1
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Create a Python virtual environment (`.venv`)
2. Install PyTorch with CUDA 12.4 support (falls back to CPU if no GPU)
3. Install all remaining dependencies from `requirements.txt`

### Manual Setup

If you prefer to set up manually:

```bash
# 1. Clone the repository
git clone https://github.com/DangerDudeSL/SemanticEnergyV1.git
cd SemanticEnergyV1

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 3. Install PyTorch (pick ONE)
# With CUDA 12.4 (recommended for GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only:
pip install torch torchvision torchaudio

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Running the Application

**Windows (PowerShell) — One Command:**
```powershell
.\start.ps1
```
This launches both the backend (port 8000) and frontend (port 3000) simultaneously.

**Manual Start (any OS):**
```bash
# Terminal 1 — Backend
cd backend
python app.py
# API will be available at http://127.0.0.1:8000
# Swagger docs at http://127.0.0.1:8000/docs

# Terminal 2 — Frontend
cd frontend
python -m http.server 3000
# Open http://127.0.0.1:3000 in your browser
```

> **Note:** On the first run, the Qwen 2.5 1.5B model (~3 GB) will be automatically downloaded from Hugging Face. Subsequent runs will use the cached model.

### Testing the API

Once the backend is running, you can test it with:
```bash
python test_endpoint.py
```

Or via curl:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "num_samples": 5}'
```

---

## ☁️ Deploying to the Web

### Why Self-Hosted LLM is Required

Semantic Energy requires **per-token logits** from the language model — not just generated text. Most LLM API providers (OpenAI, Anthropic, etc.) don't expose raw logits, which makes them incompatible. The model must be self-hosted with full access to output scores.

### 🥇 Easiest: Google Colab + ngrok (Free)

The fastest way to get a public URL — no setup, no Docker, no credit card.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DangerDudeSL/SemanticEnergyV1/blob/master/SemanticEnergy_Colab.ipynb)

**What you need:**
1. A Google account (for Colab)
2. A free [ngrok auth token](https://dashboard.ngrok.com/get-started/your-authtoken)

**Steps:**
1. Click the **Open in Colab** badge above
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run all cells — paste your ngrok token when prompted
4. Get a public URL like `https://abc123.ngrok-free.app` and share it!

> ⚠️ Free Colab sessions last ~12 hours and the URL changes on restart. For a permanent deployment, use Hugging Face Spaces below.

### Alternative: Hugging Face Spaces (Free T4 GPU)

This repo includes a ready-to-deploy `Dockerfile` for [Hugging Face Spaces](https://huggingface.co/spaces). The Qwen 2.5 1.5B model fits comfortably on a free T4 GPU (16 GB VRAM).

**Step-by-step:**

1. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select **Docker** as the SDK
   - Select **T4 small** as hardware (free tier)

2. **Clone your Space and copy the files:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME

   # Copy the project files
   cp -r /path/to/SemanticEnergyV1/* .
   ```

3. **Push to deploy:**
   ```bash
   git add .
   git commit -m "Deploy Semantic Energy"
   git push
   ```

4. Your app will be live at `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`

> **Note:** First deployment takes ~5 minutes as it builds the Docker image and downloads the model weights (~3 GB).

### Other GPU Hosting Options

| Platform | GPU | Cost | Notes |
|---|---|---|---|
| **[Hugging Face Spaces](https://huggingface.co/spaces)** | T4 (16 GB) | Free | ✅ Recommended — includes `Dockerfile` |
| **[Modal](https://modal.com)** | Any | ~$0.10/hr | Serverless, pay-per-second, cold starts |
| **[RunPod](https://runpod.io)** | Wide selection | ~$0.20/hr | Full control, always-on |
| **[Vast.ai](https://vast.ai)** | Wide selection | ~$0.10/hr | Cheapest GPU rentals |
| **[Google Colab + ngrok](https://colab.research.google.com)** | T4 | Free | Good for demos, not permanent |

### What About LLM APIs?

| Provider | Logits Available? | Compatible? |
|---|---|---|
| **OpenAI** | Top-5 logprobs only | ❌ Insufficient |
| **Anthropic** | No | ❌ |
| **Google Gemini** | No | ❌ |
| **Together AI** | Full logprobs | ⚠️ Possible with code changes |
| **Self-hosted vLLM** | Full logits | ✅ Fully compatible |

---

## 📖 How It Works

For specific implementations, please refer to the code in the notebook. We have uploaded all the intermediate results generated by the models in the `cache_data` [Google Drive](https://drive.google.com/file/d/16ykjWpLV1bY82IRFpvMhzHyKIq9Me02J/view?usp=sharing) directory to facilitate the reproduction of experiments.

### Step 1: Sampling Response

Similar to semantic entropy, for a given question, it is necessary to first sample multiple responses. You can refer to the following code to save the required content:

```
messages = [
    {"role": "user", "content": question}
    ]
generated_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) 
inputs = tokenizer(generated_prompt, return_tensors='pt')
inputs = {k: v.to(model.device) for k, v in inputs.items()}
generated_output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    do_sample=True,
    return_dict_in_generate=True,
    output_scores=True,
)

# Extract the generated token ids (excluding the prompt)
generated_ids = generated_output.sequences[0][len(inputs["input_ids"][0]):].tolist()
scores = generated_output.scores  # List[tensor: (batch, vocab_size)], len == num_generated_tokens
logits_list = []
probs_list = []
token_ids = []

for step_idx, score_tensor in enumerate(scores):

    """Example: the next-token probability distribution is {'token_id_1': '0.75', 'token_id_2': '0.22', ...},
    the next-token logit distribution is {'token_id_1': '35', 'token_id_2': '28', ...},
    suppose the sampled token is 'token_id_2' """

    logits = score_tensor[0].tolist()  # (vocab_size,)
    token_id = generated_ids[step_idx]
    prob = F.softmax(score_tensor[0], dim=-1)[token_id].item()
    logits_list.append(logits[token_id])  # Save the logit value corresponding to 'token_id_2': 28,
    probs_list.append(prob)              # Save the probability value corresponding to 'token_id_2': 0.22,
    token_ids.append(token_id)           # Save the value :'token_id_2'

```
### Step 2: Semantic Clustering

You can refer to the following code to cluster the different generated responses. Here, we take ```TIGER-Lab/general-verifier``` as an example to analyze semantics, though you can also use other models:

```
class SemanticAnalyser:
    def __init__(self, model_path="TIGER-Lab/general-verifier"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).cuda()

    def semantic_analyse(self, question, answer_a, answer_b):
        prompt = (
            f"User: ### Question: {question}\n\n"
            f"### Ground Truth Answer: {answer_a}\n\n"
            f"### Student Answer: {answer_b}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2025,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        print(self.tokenizer.decode(outputs[0][-3:], skip_special_tokens=True))
        return self.tokenizer.decode(outputs[0][-3:], skip_special_tokens=True)

def find_semantic_clusters(question, answer_list, analyser):
    def is_semantic_same(i, j):
        return "Yes" in analyser.semantic_analyse(question, answer_list[i], answer_list[j])
    
    n = len(answer_list)
    clusters = []
    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        
        for j in range(i + 1, n):
            if not visited[j] and is_semantic_same(i, j):
                cluster.append(j)
                visited[j] = True
        clusters.append(tuple(cluster))
    print(len(clusters), f"Clusters: {clusters}")
    return clusters

analyser = SemanticAnalyser()

```

### Step 3: Uncertainty Estimation
The final step is to calculate the Semantic Energy. This paradigm is generally consistent with Semantic Entropy, with the primary difference being that the probability used for estimating uncertainty is replaced with logits. For the code, please refer to ```semantic_energy.ipynb```.

**Reliability of a single response:**
The reliability of a single response is equal to the reliability of the cluster it belongs to. For example, if a question is answered 5 times, and the answers are semantically clustered as ```(answer1, answer2, answer3)``` and ```(answer4, answer5)```, then we can compute the energies of the two clusters, namely ```energy_cluster1``` and ```energy_cluster2```. Consequently, the reliability of ```answer1, answer2, answer3``` is given by the value computed from ```energy_cluster1```, while the reliability of ```answer4, answer5``` is given by the value computed from ```energy_cluster2```.


### Contact Us
You can get in touch with us by sending an email to the corresponding author. If the corresponding author receives the email, they will convey its contents to me. For a faster response, you can directly raise an issue in this project, and I will do my best to reply to your question on the same day.


### Citation

```
@article{ma2025semantic,
  title={Semantic Energy: Detecting LLM Hallucination Beyond Entropy},
  author={Ma, Huan and Pan, Jiadong and Liu, Jing and Chen, Yan and Joey Tianyi Zhou and Wang, Guangyu and Hu, Qinghua and Wu, Hua and Zhang, Changqing and Wang, Haifeng},
  journal={arXiv preprint arXiv:2508.14496},
  year={2025}
}
```
