# SemanticEnergy — Class Diagram (OOADM)

## Compact Class Diagram (Thesis-friendly)

```mermaid
classDiagram
    direction LR

    class SEPenergyAPI {
        -SemanticEngine engine
        -dict probe_bundle
        -str current_model_id
        +startup_event()
        +chat_endpoint(Request)
        +score_fast_tbg(Request)
        +score_fast_slt(Request)
        +switch_model_endpoint(Request)
        +status_endpoint()
    }

    class SemanticEngine {
        -AutoModelForCausalLM model
        -AutoTokenizer tokenizer
        -ClaimFilter _claim_filter
        -Segmenter _sentence_segmenter
        +generate_responses(str, int) list
        +score_sentences(str, list, list, list) list
        +semantic_analyse(str, str, str) bool
        +find_semantic_clusters(str, list) list
        -_extract_hidden_states(str, str, list) tuple
        +score_with_tbg_probe(str, dict) dict
        +score_with_slt_probe(str, dict) dict
    }

    class ClaimFilter {
        +set FILLER_PHRASES
        +list NON_CLAIM_PATTERNS
        -list _compiled
        +is_claim(str) bool
    }

    class ProbeBundle {
        +LogisticRegression tbg_energy_probe
        +LogisticRegression tbg_entropy_probe
        +LogisticRegression slt_energy_probe
        +LogisticRegression slt_entropy_probe
        +StandardScaler[4] scalers
        +tuple[4] layer_ranges
    }

    class SentenceScore {
        +str text
        +float confidence
        +str level
        +bool is_claim
        +float energy_risk
        +float entropy_risk
        +float probe_risk
    }

    class GeneratedResponse {
        +str answer
        +list logits
        +list probs
        +list token_ids
        +list top2_logits
    }

    SEPenergyAPI "1" *-- "1" SemanticEngine
    SEPenergyAPI "1" *-- "0..1" ProbeBundle
    SemanticEngine "1" *-- "1" ClaimFilter
    SemanticEngine "1" ..> "0..*" GeneratedResponse
    SemanticEngine "1" ..> "0..*" SentenceScore
    ProbeBundle "1" o-- "4" LogisticRegression
```

## Rendering Instructions

**Option A — Mermaid Live (fastest)**
1. Go to **mermaid.live**
2. Paste the code above
3. Export as SVG/PNG → insert into thesis

**Option B — draw.io manual build**
1. Open **app.diagrams.net**
2. Search "UML" in left sidebar shapes
3. Drag UML Class shapes and fill using the details above
