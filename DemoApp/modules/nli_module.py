import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

class NLIInferenceModule:
    def __init__(self, model_path="assets/XLM-R_base_nli_100k_2e"):
        self.tokenizer, self.model = load_model_and_tokenizer(model_path)
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def predict(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        return self.label_map[pred], probs.squeeze().tolist()
