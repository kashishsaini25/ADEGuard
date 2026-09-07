# ADEGuard — AI-Powered Adverse Drug Event Detection and Severity Mapping

ADEGuard analyzes vaccine/drug adverse event reports (VAERS-style free text) and:

- **Classifies severity** (mild / moderate / severe) with a fine-tuned BioBERT model
- **Extracts medical entities** (symptoms, drugs, diseases, dosage, etc.) with a biomedical NER model
- **Clusters symptoms semantically** (Sentence-BERT embeddings + UMAP + HDBSCAN) and links clusters to historical severe-outcome rates
- **Explains predictions** with LIME and SHAP
- **Generates a plain-language narrative summary** of each report using an LLM (Groq/Llama/GPT-OSS), with explicit safety guardrails against diagnosis or causality claims

> ⚠️ **Disclaimer**: This is a research/portfolio project, not a medical or regulatory tool. Severity labels are derived from a rule-based heuristic over VAERS outcome fields (death, hospitalization, life-threatening event, disability, ER visit) — they are a **proxy**, not clinical ground truth. The model does not diagnose patients or establish that a vaccine/drug caused an event.

## Demo

```bash
streamlit run app.py
```

![screenshot placeholder](docs/screenshot.png)

## Architecture

```
Raw VAERS CSVs (DATA / SYMPTOMS / VAX)
        │
        ▼
Data merge + cleaning + rule-based severity labeling
        │
        ▼
   ┌────────────────────┬─────────────────────┐
   ▼                    ▼                     ▼
BioBERT fine-tune   Biomedical NER      Sentence-BERT embeddings
(severity classifier)  (entity extraction)   → UMAP → HDBSCAN
        │                    │              (symptom clustering)
        │                    └───────┬──────────┘
        │                            ▼
        │                 Symptom-cluster ↔ severity
        │                    intelligence table
        ▼                            │
   LIME / SHAP                       │
   explanations                      │
        │                            │
        └─────────────┬──────────────┘
                       ▼
         LLM narrative summary (Groq)
                       │
                       ▼
              Streamlit demo app
```

## Tech stack

| Component | Tool |
|---|---|
| Severity classification | `dmis-lab/biobert-v1.1` fine-tuned via 🤗 Transformers |
| NER | `d4data/biomedical-ner-all` |
| Symptom embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensionality reduction | UMAP |
| Clustering | HDBSCAN |
| Explainability | SHAP, LIME |
| GenAI narrative | Groq API (OpenAI-compatible), `openai/gpt-oss-20b` |
| Demo UI | Streamlit |

## Repository structure

```
.
├── app.py                     # Streamlit demo app
├── requirements.txt
├── README.md
├── notebooks/
│   └── ADEGuard_GenAI.ipynb   # full training + analysis pipeline
├── models/
│   └── severity/biobert/      # fine-tuned model (not committed — see below)
└── results/
    ├── symptom_intelligence.csv
    ├── severity_model_results.json
    └── ...
```

## Getting the trained model

The fine-tuned BioBERT severity model (~430MB) is too large for a normal git push and is **not committed** to this repo (see `.gitignore`). Choose one:

**Option A — Hugging Face Hub (recommended)**
1. From Colab, push the model: `trainer.push_to_hub("yourusername/adeguard-severity-biobert")` (requires a free HF account + `huggingface-cli login`)
2. Set an environment variable before running the app: `export SEVERITY_MODEL_ID=yourusername/adeguard-severity-biobert`

**Option B — Local copy**
1. Download `models/severity/biobert/` from your Colab session (zip it, then use the Colab file browser to download, or save directly to Google Drive)
2. Place it at `models/severity/biobert/` in this repo locally (it will be git-ignored, which is intentional)

## Setup

```bash
git clone https://github.com/<your-username>/ADEGuard.git
cd ADEGuard
pip install -r requirements.txt

# Get a free Groq API key at https://console.groq.com/keys
export GROQ_API_KEY=your_key_here      # Windows (cmd): set GROQ_API_KEY=your_key_here

streamlit run app.py
```

## Reproducing training

The full pipeline (data loading, cleaning, severity labeling, BioBERT fine-tuning, NER, clustering, XAI, GenAI) is in `notebooks/ADEGuard_GenAI.ipynb`. It's designed to run on Google Colab with a T4 GPU. Data source: [VAERS](https://vaers.hhs.gov/data/datasets.html) (2024–2026 CSV exports).

## Results

| Metric | Score |
|---|---|
| Accuracy | 92.5% |
| Weighted F1 | 0.922 |
| Macro F1 | 0.76 |

Per-class performance (severity classifier):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| mild | 0.95 | 0.97 | 0.96 |
| moderate | 0.63 | 0.50 | 0.56 |
| severe | 0.82 | 0.73 | 0.77 |

**Known limitation**: the dataset is heavily imbalanced (~86% mild), so accuracy is dominated by the majority class. The `moderate` class in particular is under-served — see the "Future work" section below.

## Limitations

- Severity labels are a rule-based proxy over self-reported VAERS outcome fields, not clinician-verified ground truth
- VAERS data itself is subject to well-known self-reporting and ascertainment biases
- The GenAI narrative is generated by an LLM and explicitly instructed not to diagnose or claim causality — it summarizes model outputs, not medical facts beyond the report text
- No entity linking to standardized vocabularies (e.g., MedDRA) yet — extracted symptoms are free text

## Future work

- Class-weighted / focal loss training to improve `moderate`-class recall
- MedDRA-based entity normalization
- Drug–symptom pair (relation) extraction
- FastAPI endpoint for programmatic access

## License

MIT
