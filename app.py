
import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
)
from openai import OpenAI

CLASS_NAMES = ["mild", "moderate", "severe"]
LOCAL_SEVERITY_MODEL_PATH = "models/severity/biobert"
HF_SEVERITY_MODEL_ID = os.environ.get("SEVERITY_MODEL_ID", "")
NER_MODEL_NAME = "d4data/biomedical-ner-all"
SYMPTOM_INTELLIGENCE_PATH = "results/symptom_intelligence.csv"

ENTITY_MAPPING = {
    "CHEMICAL": "DRUG",
    "CHEMICAL_SUBSTANCE": "DRUG",
    "DRUG": "DRUG",
    "MEDICATION": "DRUG",
    "DISEASE": "DISEASE",
    "DISEASE_DISORDER": "DISEASE",
    "SIGN_SYMPTOM": "SYMPTOM",
    "SYMPTOM": "SYMPTOM",
    "DOSAGE": "DOSAGE",
    "ROUTE": "ROUTE",
    "FREQUENCY": "FREQUENCY",
    "DURATION": "DURATION",
    "SEVERITY": "SEVERITY",
}

GENAI_MODEL = "openai/gpt-oss-20b"
@st.cache_resource(show_spinner="Loading severity model...")
def load_severity_model():
    model_source = (
        LOCAL_SEVERITY_MODEL_PATH
        if os.path.isdir(LOCAL_SEVERITY_MODEL_PATH)
        else HF_SEVERITY_MODEL_ID
    )
    if not model_source:
        return None, None, None

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Loading biomedical NER model...")
def load_ner_pipeline():
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "ner",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple",
        device=device,
    )


@st.cache_data(show_spinner=False)
def load_symptom_intelligence():
    if os.path.exists(SYMPTOM_INTELLIGENCE_PATH):
        return pd.read_csv(SYMPTOM_INTELLIGENCE_PATH)
    return None


def get_genai_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
def normalize_entity_label(label):
    label = str(label).upper().strip()
    return ENTITY_MAPPING.get(label, label)


def predict_severity(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    predicted_id = int(np.argmax(probabilities))
    return CLASS_NAMES[predicted_id], float(probabilities[predicted_id]), probabilities


def extract_entities(text, ner_pipeline):
    raw_entities = ner_pipeline(text)
    extracted = []
    for entity in raw_entities:
        original_label = entity["entity_group"]
        normalized_label = normalize_entity_label(original_label)
        extracted.append(
            {
                "text": entity["word"],
                "original_label": original_label,
                "label": normalized_label,
                "confidence": round(float(entity["score"]), 4),
            }
        )
    return extracted


def clean_json_output(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def build_genai_prompt(medical_text, entities, predicted_severity, confidence):
    return f"""
You are the GenAI explanation component of ADEGuard,
an AI system for adverse-event analysis using VAERS-style reports.

Analyze the following adverse-event report and the outputs
from the machine-learning pipeline.

IMPORTANT:
- Do not diagnose the patient.
- Do not claim that a vaccine or drug caused the event.
- Do not recommend treatment.
- Do not invent medical facts that are not present.
- Clearly distinguish reported information from AI-generated interpretation.
- Treat the predicted severity as a model prediction, not a clinical diagnosis.

REPORT:
{medical_text}

DETECTED MEDICAL ENTITIES:
{json.dumps(entities, indent=2)}

MODEL-PREDICTED SEVERITY:
{predicted_severity}

MODEL CONFIDENCE:
{confidence:.2%}

Generate a concise structured analysis containing:
1. Event Summary
2. Detected Symptoms
3. Detected Drugs/Vaccines
4. Other Important Entities
5. Predicted Severity
6. Key Evidence
7. Important Uncertainty
8. Safety/Causality Disclaimer

Return the answer in clear JSON format only, with no markdown fences.
"""


def get_genai_analysis(client, prompt):
    response = client.chat.completions.create(
        model=GENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful biomedical NLP assistant "
                    "for adverse-event analysis."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    raw_output = response.choices[0].message.content
    try:
        return json.loads(clean_json_output(raw_output))
    except Exception:
        return {"raw_output": raw_output}

st.set_page_config(page_title="ADEGuard", page_icon="💊", layout="wide")

st.title("💊 ADEGuard")
st.caption(
    "AI-powered adverse drug event detection and severity mapping — "
    "BioBERT severity classification, biomedical NER, symptom clustering, "
    "and an LLM-generated plain-language summary."
)

st.warning(
    "⚠️ Research and portfolio demo only. This tool does not diagnose, "
    "does not establish causality between a vaccine/drug and an event, "
    "and is not a substitute for professional medical or regulatory review.",
    icon="⚠️",
)

severity_tokenizer, severity_model, device = load_severity_model()
ner_pipeline_obj = load_ner_pipeline()
symptom_intelligence = load_symptom_intelligence()
genai_client = get_genai_client()

if severity_tokenizer is None:
    st.error(
        "Severity model not found. Place your fine-tuned BioBERT model in "
        f"`{LOCAL_SEVERITY_MODEL_PATH}`, or set the `SEVERITY_MODEL_ID` "
        "environment variable to a Hugging Face Hub repo id. "
        "See the README for details."
    )
    st.stop()

default_text = (
    "After receiving the vaccine, the patient developed severe headache, "
    "dizziness and vomiting. The patient was subsequently taken to the "
    "emergency department for evaluation."
)

text_input = st.text_area(
    "Adverse event report text",
    value=default_text,
    height=150,
)

analyze_clicked = st.button("Analyze report", type="primary")

if analyze_clicked and text_input.strip():
    with st.spinner("Running severity model and entity extraction..."):
        severity_label, confidence, probabilities = predict_severity(
            text_input, severity_tokenizer, severity_model, device
        )
        entities = extract_entities(text_input, ner_pipeline_obj)
        symptoms = [e for e in entities if e["label"] == "SYMPTOM"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Predicted severity")
        severity_color = {"mild": "🟢", "moderate": "🟡", "severe": "🔴"}
        st.markdown(f"## {severity_color.get(severity_label, '')} {severity_label.upper()}")
        st.metric("Model confidence", f"{confidence:.1%}")

        prob_df = pd.DataFrame(
            {"severity": CLASS_NAMES, "probability": probabilities}
        ).set_index("severity")
        st.bar_chart(prob_df)

    with col2:
        st.subheader("Detected medical entities")
        if entities:
            entity_df = pd.DataFrame(entities)[["text", "label", "confidence"]]
            st.dataframe(entity_df, use_container_width=True, hide_index=True)
        else:
            st.write("No entities detected.")

        if symptoms and symptom_intelligence is not None:
            st.subheader("Symptom cluster context")
            symptom_texts_lower = {s["text"].lower().strip() for s in symptoms}
            matches = symptom_intelligence[
                symptom_intelligence["symptom"].isin(symptom_texts_lower)
            ]
            if not matches.empty:
                st.dataframe(
                    matches[["symptom", "cluster", "frequency", "severe_percentage"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No matching symptom-cluster data for these terms.")

    st.subheader("GenAI narrative summary")
    if genai_client is None:
        st.info(
            "Set a `GROQ_API_KEY` environment variable (or Streamlit secret) "
            "to enable the GenAI narrative summary. Get a free key at "
            "console.groq.com/keys."
        )
    else:
        with st.spinner("Generating narrative summary..."):
            prompt = build_genai_prompt(text_input, entities, severity_label, confidence)
            try:
                analysis = get_genai_analysis(genai_client, prompt)
                if "raw_output" in analysis:
                    st.code(analysis["raw_output"])
                else:
                    for key, value in analysis.items():
                        st.markdown(f"**{key}**")
                        st.write(value)
            except Exception as exc:
                st.error(f"GenAI request failed: {exc}")

elif analyze_clicked:
    st.warning("Please enter some report text first.")

st.divider()
st.caption(
    "Built on VAERS-style data. Severity labels are derived from a rule-based "
    "heuristic over reported outcome fields (death, hospitalization, "
    "life-threatening event, disability, ER visit) and are a proxy, not a "
    "clinical ground truth."
)
