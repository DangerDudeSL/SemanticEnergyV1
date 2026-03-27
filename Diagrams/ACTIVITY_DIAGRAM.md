# SemanticEnergy — Activity Diagram (OOADM)

## SLT Scoring Activity Diagram

```mermaid
flowchart TB
    START(("●"))

    subgraph ChatUI["Chat UI"]
        A1["Submit Question"]
        A_END["Display Answer +<br/>Overall Risk +<br/>Sentence-level Highlights"]
    end

    subgraph API["SEPenergy API"]
        B1["Receive POST /score_fast_slt"]
        B2["Return JSON Response"]
    end

    subgraph Engine["SEPEnergy Engine"]
        C1["Pass 1: model.generate()"]
        C2["Split into Sentences"]
        C3{"Is Claim?"}
        C4["Score Sentence<br/>(sigmoid on logits)"]
        C5["Skip Sentence"]
        C6["Pass 2: model.forward()<br/>extract hidden states"]
        C7{"Answer ≤ 100<br/>tokens?"}
        C8["Use SLT probe directly"]
        C9["Blend SLT +<br/>per-sentence risks"]
    end

    subgraph Probes["Probe Bundle"]
        D1["scaler.transform()"]
        D2["predict_proba()"]
        D3["Return energy_risk +<br/>entropy_risk"]
    end

    STOP(("◉"))

    START --> A1
    A1 --> B1
    B1 --> C1
    C1 --> C2
    C2 --> C3
    C3 -->|"Yes"| C4
    C3 -->|"No"| C5
    C4 --> C6
    C5 --> C6
    C6 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> C7
    C7 -->|"Yes"| C8
    C7 -->|"No"| C9
    C8 --> B2
    C9 --> B2
    B2 --> A_END
    A_END --> STOP

    style START fill:#000,stroke:#000,color:#fff
    style STOP fill:#000,stroke:#000,color:#fff
    style C3 fill:#FFF3CD,stroke:#CC9A06,color:#000
    style C7 fill:#FFF3CD,stroke:#CC9A06,color:#000
```

## Draw.io Instructions (Recommended for Thesis)

Since Mermaid cannot render proper UML activity diagram swim lanes, use draw.io:

1. Open **app.diagrams.net** → New Diagram
2. Choose template: **UML → Activity Diagram** (has all the right shapes)
3. Create **5 vertical swim lane partitions**:

| Swim Lane | Actions |
|-----------|---------|
| **Chat UI** | Submit Question, Display Results |
| **SEPenergy API** | Receive POST, Return JSON |
| **SEPEnergy Engine** | generate(), split_sentences(), score_sentences(), forward(), extract_hidden_states(), aggregate risk |
| **LLM** | Return answer + logits (Pass 1), Return hidden_states (Pass 2) |
| **Probe Bundle** | transform(), predict_proba(), return risks |

4. **Decision nodes** (diamonds):
   - "Is Claim?" → Yes: Score Sentence / No: Skip
   - "Answer ≤ 100 tokens?" → Yes: SLT direct / No: Blend risks

5. **Start**: Filled black circle (●) at top of Chat UI lane
6. **End**: Bull's-eye circle (◉) at bottom of Chat UI lane
7. **Control flow**: Solid arrows between actions
8. **Guard conditions**: Labels on decision branches [Yes] / [No]
