# SemanticEnergy — System Architecture Diagrams

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend (Vanilla JS — Port 3000)"]
        UI["Chat UI<br/>index.html"]
        ModeSelector["Mode Selector<br/>Full SE | Fast SLT | Fast TBG"]
        StateManager["State Manager<br/>localStorage + sessionStorage"]
        Renderer["Response Renderer<br/>Sentence Highlighting<br/>Confidence Badges<br/>Metrics Panels"]
        UI --> ModeSelector
        UI --> Renderer
        UI --> StateManager
    end

    subgraph Backend["Backend (FastAPI — Port 8000)"]
        API["API Layer<br/>app.py"]
        Engine["SemanticEngine<br/>engine.py"]
        ClaimFilter["Claim Filter<br/>claim_filter.py<br/>(Regex-based)"]
        ProbeBundle["Probe Bundle<br/>4 LogisticRegression<br/>+ 4 StandardScalers"]

        API --> Engine
        API --> ProbeBundle
        Engine --> ClaimFilter
    end

    subgraph ML["ML / Model Layer"]
        LLM["Llama 3.1-8B-Instruct<br/>8-bit Quantized (bitsandbytes)<br/>~9 GB VRAM"]
        HiddenStates["Hidden State Extractor<br/>33 Transformer Layers<br/>4096-dim per layer"]
        LLM --> HiddenStates
    end

    subgraph Training["Training Pipeline (Notebooks)"]
        NB0["00_preflight<br/>Math Verification"]
        NB1["01_generate_dataset<br/>500 TriviaQA Records<br/>Hidden States + Labels"]
        NB2["02_train_se_probes<br/>Layer Sweep + Training<br/>4 Probes → .pkl"]
        NB4["04_sentence_baseline<br/>Per-Sentence Validation"]
        NB0 --> NB1 --> NB2 --> NB4
    end

    subgraph Storage["Data & Models"]
        Dataset["probe_dataset<br/>541 MB .pkl"]
        Probes["probes_llama3-8b<br/>2.1 MB .pkl"]
        NB2 --> Probes
        NB1 --> Dataset
    end

    Frontend -->|"REST API (JSON)<br/>CORS Enabled"| Backend
    Engine --> LLM
    ProbeBundle -.->|"Loads at startup"| Probes
```

## 2. Request Flow — Three Scoring Modes

```mermaid
flowchart LR
    User["👤 User Input"]

    subgraph FullSE["Full SE Mode (60-120s)"]
        direction TB
        G5["Generate 5 Samples<br/>temperature=0.7"]
        SC["Semantic Clustering<br/>Pairwise LLM Verification"]
        EF["Energy Flow Calc<br/>Fermi-Dirac Transform"]
        SS1["Score Sentences<br/>Sigmoid Normalization"]
        CF1["Claim Filter"]
        G5 --> SC --> EF --> SS1 --> CF1
    end

    subgraph FastSLT["Fast SLT Mode (5-15s)"]
        direction TB
        G1["Generate 1 Sample"]
        FP1["Forward Pass<br/>prompt + answer"]
        SLT["Extract SLT Hidden State<br/>(2nd-to-last token)"]
        P1["Probe Prediction<br/>energy + entropy risk"]
        SS2["Score Sentences"]
        G1 --> FP1 --> SLT --> P1 --> SS2
    end

    subgraph FastTBG["Fast TBG Mode (0.5-2s → 5-15s)"]
        direction TB
        FP2["Forward Pass<br/>prompt only"]
        TBG["Extract TBG Hidden State<br/>(last prompt token)"]
        P2["Pre-Gen Probe<br/>⚡ Instant Risk"]
        Then["Then → SLT Pipeline"]
        FP2 --> TBG --> P2 --> Then
    end

    User -->|"/chat"| FullSE
    User -->|"/score_fast_slt"| FastSLT
    User -->|"/score_fast_tbg"| FastTBG

    FullSE --> Response["📊 Response<br/>confidence + clusters<br/>+ sentence scores"]
    FastSLT --> Response
    FastTBG --> Response
```

## 3. Data Flow — Hidden State Extraction & Probe Scoring

```mermaid
flowchart TB
    Input["Input: question + answer"]

    subgraph Tokenization["Tokenization"]
        T1["Tokenize prompt → prompt_len"]
        T2["Tokenize prompt+answer → full_len"]
    end

    subgraph ForwardPass["Forward Pass (output_hidden_states=True)"]
        FP["Run through 33 Transformer Layers"]
        L["Layer outputs:<br/>[batch, seq_len, 4096] × 33"]
    end

    subgraph Extraction["Hidden State Extraction"]
        TBG_H["TBG: hidden[:, prompt_len-1, :]<br/>All 33 layers"]
        SLT_H["SLT: hidden[:, full_len-2, :]<br/>All 33 layers"]
        EXTRA["Extra: sentence boundary positions"]
    end

    subgraph ProbeScoring["Probe Scoring"]
        direction LR
        subgraph TBG_Probes["TBG Probes (Layers 28-32)"]
            TE["Energy Probe<br/>AUROC: 0.748"]
            TN["Entropy Probe<br/>AUROC: 0.786"]
        end
        subgraph SLT_Probes["SLT Probes (Layers 17-24)"]
            SE["Energy Probe<br/>AUROC: 0.667"]
            SN["Entropy Probe<br/>AUROC: 0.788"]
        end
    end

    subgraph Output["Risk Calculation"]
        Scale["StandardScaler.transform()"]
        Predict["probe.predict_proba()"]
        Risk["combined_risk =<br/>(energy_risk + entropy_risk) / 2"]
        Level["Map → high | medium | low"]
    end

    Input --> Tokenization --> ForwardPass
    FP --> L --> Extraction
    TBG_H --> TBG_Probes
    SLT_H --> SLT_Probes
    TBG_Probes --> Scale
    SLT_Probes --> Scale
    Scale --> Predict --> Risk --> Level
```

## 4. Deployment Architecture

```mermaid
graph TB
    subgraph Local["Local Development"]
        SetupBat["setup.bat / setup.sh<br/>Python 3.12 + CUDA 12.4"]
        StartPS["start.ps1<br/>Launches both servers"]
        FE_Local["Frontend :3000<br/>Python http.server"]
        BE_Local["Backend :8000<br/>Uvicorn + FastAPI"]
        GPU_Local["NVIDIA GPU<br/>≥12 GB VRAM"]
        SetupBat --> StartPS
        StartPS --> FE_Local
        StartPS --> BE_Local
        BE_Local --> GPU_Local
    end

    subgraph HFSpaces["HuggingFace Spaces (Docker)"]
        Dockerfile["Dockerfile<br/>python:3.10-slim"]
        DeployStart["deploy_start.py<br/>Combined Server :7860"]
        StaticFiles["Static Files<br/>index.html, script.js, styles.css"]
        Dockerfile --> DeployStart
        DeployStart --> StaticFiles
    end

    subgraph FreeTier["Free Tier Architecture"]
        Vercel["Vercel<br/>Frontend (FREE)"]
        Colab["Google Colab<br/>Backend (FREE T4 GPU)"]
        Ngrok["ngrok<br/>Static Domain Tunnel"]
        HF_Hub["HuggingFace Hub<br/>Probe Storage"]
        Vercel -->|HTTPS| Ngrok
        Ngrok --> Colab
        Colab --> HF_Hub
    end
```

## 5. Frontend Component Architecture

```mermaid
graph TB
    subgraph Navbar["Navbar (Fixed 60px)"]
        Brand["Brand: Semantic Energy + BETA"]
        MetricsGuide["ℹ️ Metrics Guide"]
        ClearChat["🗑️ Clear Chat"]
        ModelDropdown["Model Dropdown<br/>Llama 3.1 8B | Qwen 2.5 1.5B"]
    end

    subgraph ChatContainer["Chat Container (Scrollable)"]
        Welcome["Welcome Message"]
        Messages["Message Bubbles"]
        subgraph MessageComponents["Per-Message Components"]
            Bubble["Text Bubble<br/>(HTML-escaped)"]
            Badge["Confidence Badge<br/>🟢 🟡 🔴 + %"]
            Timer["Response Timer<br/>⏱ X.Xs"]
            Metrics["Metrics Panel<br/>(Collapsible)"]
            SentenceTable["Sentence Analysis<br/>S1, S2... with bars"]
        end
    end

    subgraph InputArea["Input Area (Fixed Bottom)"]
        ModeBtn["Mode Selector<br/>Full SE | Fast SLT | Fast TBG ⚡"]
        TextArea["Auto-resize Textarea<br/>Shift+Enter = newline"]
        SendBtn["Send Button"]
    end

    subgraph State["State Management"]
        LS["localStorage<br/>mode, model_id, model_label"]
        SS["sessionStorage<br/>chat history HTML"]
    end

    Navbar --> ChatContainer --> InputArea
    InputArea --> State
    Messages --> MessageComponents
```

## 6. ML Training Pipeline

```mermaid
flowchart LR
    subgraph Data["Data Collection"]
        TQA["TriviaQA Dataset<br/>500 Questions"]
        Gen["5 Generations/Question<br/>with Hidden States"]
        Labels["Teacher Signals<br/>energy_score_raw<br/>entropy_score_raw"]
        TQA --> Gen --> Labels
    end

    subgraph Preprocessing["Preprocessing"]
        Binarize["Binarize Labels<br/>MSE Minimization<br/>(No Ground Truth)"]
        LayerSweep["Layer Sweep<br/>0-33 layers<br/>Window size=5"]
        Labels --> Binarize
        Binarize --> LayerSweep
    end

    subgraph Training["Training"]
        LR_E["LogisticRegression<br/>Energy Probes ×2"]
        LR_N["LogisticRegression<br/>Entropy Probes ×2"]
        LayerSweep --> LR_E
        LayerSweep --> LR_N
    end

    subgraph Evaluation["Evaluation"]
        AUROC["AUROC Scores<br/>Val: 0.74-0.87<br/>Test: 0.67-0.79"]
        Bootstrap["Bootstrap CI<br/>Feature Ablation"]
        LR_E --> AUROC
        LR_N --> AUROC
        AUROC --> Bootstrap
    end

    subgraph Output["Output"]
        PKL["probes_llama3-8b_triviaqa.pkl<br/>4 probes + 4 scalers<br/>2.1 MB"]
        Bootstrap --> PKL
    end
```
