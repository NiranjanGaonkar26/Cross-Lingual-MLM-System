import streamlit as st
from modules.embedding_module import WordEmbeddingSimilarityModule
from modules.nli_module import NLIInferenceModule

st.set_page_config(page_title="Multilingual NLP App", layout="centered")
st.title("🌍 Multilingual Language Model Explorer")

# Dropdown to select module
module = st.selectbox("Select a module:", [
    "Word Embedding Similarity",
    "NLI (Natural Language Inference)"
])

# Instantiate modules
embedding_module = WordEmbeddingSimilarityModule()
nli_module = NLIInferenceModule()

if module == "Word Embedding Similarity":
    st.header("🔤 Word Embedding Similarity")

    word = st.text_input("Enter an English word:", "book")

    if st.button("Compute Similarity"):
        with st.spinner("Translating and encoding..."):
            translations = embedding_module.translate(word)
            st.write("**Translations:**")
            st.json(translations)

            embeddings = embedding_module.get_embeddings(translations)
            sim_matrix, labels = embedding_module.compute_cosine_similarity(embeddings)

        st.subheader("📊 Cosine Similarity Heatmap")
        st.pyplot(embedding_module.plot_heatmap(sim_matrix, labels))

        st.subheader("🌀 Embeddings Visualization")
        st.pyplot(embedding_module.plot_pca(embeddings))

elif module == "NLI (Natural Language Inference)":
    lang = st.selectbox("Select a Language:", ["English", "French", "Kannada"])

    if lang == "English":
        st.header("🤔 Natural Language Inference (English)")

        premise = st.text_area("Enter the Premise (English):", "He was reading a book.")
        hypothesis = st.text_area("Enter the Hypothesis (English):", "He was studying.")

    elif lang == "French":
        st.header("🤔 Natural Language Inference (French)")

        premise = st.text_area("Enter the Premise (French):", "Il lisait un livre.")
        hypothesis = st.text_area("Enter the Hypothesis (French):", "Il était en train d'étudier.")

    elif lang == "Kannada":
        st.header("🤔 Natural Language Inference (Kannada)")

        premise = st.text_area("Enter the Premise (Kannada):", "ಅವನು ಪುಸ್ತಕವನ್ನು ಓದುತ್ತಿದ್ದನು.")
        hypothesis = st.text_area("Enter the Hypothesis (Kannada):", "ಅವನು ಅಧ್ಯಯನ ಮಾಡುತ್ತಿದ್ದನು.")

    if st.button("Predict Relationship"):
        with st.spinner("Predicting using XLM-R model..."):
            label, probs = nli_module.predict(premise, hypothesis)

        st.success(f"**Prediction:** {label}")
        st.write("**Confidence Scores:**")
        st.json({"entailment": round(probs[0], 4), "neutral": round(probs[1], 4), "contradiction": round(probs[2], 4)})

        st.subheader("📊 Bar Plot of Accuracy and F1 score for XLM-R")
        st.image('assets/XLM-R Acc F1.png', output_format="PNG")

