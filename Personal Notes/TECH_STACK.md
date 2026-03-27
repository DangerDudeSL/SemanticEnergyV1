# 7.2 Technology Stack — SEPEnergy

## 7.2.1 Technology Stack Diagram
(See wireframes/07_tech_stack.html)

---

## 7.2.2 Programming Languages

| Language | Version | Usage | Justification |
|----------|---------|-------|---------------|
| Python | 3.12 | Backend server, ML inference, probe training, data processing | De facto standard for ML/AI. Native support for PyTorch, HuggingFace, scikit-learn. Required for bitsandbytes GPU quantization. |
| JavaScript | ES6+ | Frontend chat UI, DOM manipulation, API calls, state management | Vanilla JS chosen over frameworks (React/Vue) to minimize bundle size and complexity for a single-page chat interface. No build step required. |
| HTML5 | 5 | Frontend page structure, semantic markup | Standard web markup language. |
| CSS3 | 3 | UI styling, glassmorphism theme, responsive layout, sentence highlighting | Native CSS with custom properties. No preprocessor needed for this scope. |
| LaTeX | - | Thesis documentation, algorithm pseudocode | University requirement for formal documentation. |

---

## 7.2.3 Development Frameworks

| Framework | Version | Layer | Justification |
|-----------|---------|-------|---------------|
| FastAPI | ≥0.129.0 | Backend REST API | Async-native Python web framework. Built-in request validation, auto-generated OpenAPI docs. Significantly faster than Flask for concurrent requests. Native async support critical for long-running LLM inference calls (~60-120s). |
| Uvicorn | ≥0.41.0 | ASGI Server | High-performance ASGI server for FastAPI. Handles async request lifecycle. |
| HuggingFace Transformers | ≥5.0.0 | ML Inference | Industry standard for loading, configuring, and running LLM inference. Provides `AutoModelForCausalLM`, `AutoTokenizer`, chat templates, and `output_hidden_states` support required for probe extraction. |
| PyTorch | ≥2.6.0 | ML Backend | Deep learning framework powering Transformers. Required for GPU tensor operations, `torch.no_grad()` inference, and CUDA memory management. |

---

## 7.2.4 Libraries / Toolkits

| Library | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| bitsandbytes | ≥0.43.0 | 8-bit model quantization | Reduces Llama 3.1 8B from ~16 GB to ~9 GB VRAM, enabling inference on consumer GPUs (12 GB). |
| scikit-learn | ≥1.5.0 | Linear probe training & inference | Provides `LogisticRegression` and `StandardScaler` for the 4 hallucination detection probes. Lightweight, no GPU needed for probe inference. |
| NumPy | ≥2.0.0 | Numerical operations | Array manipulation for hidden state extraction, feature flattening, and statistical aggregation. |
| SciPy | ≥1.14.0 | Scientific computing | Mathematical functions used in energy flow calculations. |
| pySBD | ≥0.3.4 | Sentence segmentation | Rule-based sentence boundary detection. More accurate than regex splitting for complex sentences. Used for per-sentence hallucination scoring. |
| accelerate | ≥1.10.0 | Model loading | HuggingFace library for efficient model placement across devices. Handles `device_map` for quantized models. |
| datasets | ≥3.0.0 | Training data loading | Loads TriviaQA dataset from HuggingFace Hub for probe training pipeline (notebooks). |
| ngrok | Free tier | HTTPS tunneling | Exposes Colab backend to the internet via a static domain. Enables Vercel frontend to reach Colab GPU backend. |
| Pickle | stdlib | Model serialization | Serializes trained probe bundles (4 probes + 4 scalers + layer ranges) into `.pkl` files for fast loading at startup. |

---

## 7.2.5 Integrated Development Environments (IDEs)

| IDE / Tool | Usage | Justification |
|------------|-------|---------------|
| Visual Studio Code | Primary IDE for frontend (HTML/CSS/JS) and backend (Python) development | Lightweight, extensive extension ecosystem. Python and JavaScript IntelliSense. Integrated terminal for running servers. Git integration. |
| Google Colab | Notebook IDE for probe training (Jupyter notebooks) and cloud GPU inference | Free T4 GPU access. Pre-installed PyTorch and CUDA. Used for both training pipeline (notebooks 00-04) and production backend hosting. |
| Jupyter Notebook | Probe training and experimentation | Interactive cell-based execution for iterative ML experimentation. Visualization of training metrics, AUROC curves, and feature ablation studies. |
| Git + GitHub | Version control and collaboration | Standard version control. Repository hosting, commit history, branch management. |
| Chrome DevTools | Frontend debugging | Network tab for API monitoring, Console for JS debugging, Elements for CSS inspection. |
