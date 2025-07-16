import streamlit as st
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@st.cache_resource
def load_labse_model():
    return SentenceTransformer('sentence-transformers/LaBSE')


class WordEmbeddingSimilarityModule:
    def __init__(self):
        self.translator = GoogleTranslator(source="en", target="auto")
        self.model = load_labse_model()

    def translate(self, word):
        return {
            "en": word,
            "fr": GoogleTranslator(source="en", target="fr").translate(word),
            "hi": GoogleTranslator(source="en", target="hi").translate(word),
            "kn": GoogleTranslator(source="en", target="kn").translate(word)
        }

    def get_embeddings(self, translations):
        return {lang: self.model.encode([text])[0] for lang, text in translations.items()}

    def compute_cosine_similarity(self, embeddings):
        langs = list(embeddings.keys())
        heatmap = np.zeros((len(langs), len(langs)))
        for i, l1 in enumerate(langs):
            for j, l2 in enumerate(langs):
                heatmap[i][j] = cosine_similarity([embeddings[l1]], [embeddings[l2]])[0][0]
        return heatmap, langs

    def plot_heatmap(self, sim_matrix, labels):
        fig, ax = plt.subplots()
        sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="coolwarm", ax=ax)
        return fig

    # def plot_tsne(self, embeddings):
    #     tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    #     vectors = list(embeddings.values())
    #     labels = list(embeddings.keys())
    #     reduced = tsne.fit_transform(np.array(vectors))
    #
    #     fig, ax = plt.subplots()
    #     sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", ax=ax)
    #     ax.set_title("t-SNE Visualization of Word Embeddings")
    #     return fig

    def plot_pca(self,embeddings):
        vectors = list(embeddings.values())
        labels = list(embeddings.keys())

        color_map = {
            "en": "green",
            "fr": "orange",
            "hi": "blue",
            "kn": "purple"
        }

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.array(vectors))

        fig, ax = plt.subplots()
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2", s=100)
        ax.set_title("2-D Visualization of Word Embeddings")
        ax.axhline(0, linestyle='--', color='gray', linewidth=0.5)
        ax.axvline(0, linestyle='--', color='gray', linewidth=0.5)
        return fig

        # for i, (x, y) in enumerate(reduced):
        #     lang = labels[i]
        #     color = color_map.get(lang, "black")
        #
        #     # Draw arrow
        #     ax.arrow(
        #         0, 0, x, y,
        #         head_width=0.02, head_length=0.03,
        #         fc=color, ec=color, alpha=0.8, length_includes_head=True
        #     )
        #
        #     # Draw point
        #     ax.scatter(x, y, s=100, color=color, label=lang if lang not in labels[:i] else "", zorder=3)
        #     ax.text(x + 0.01, y + 0.01, lang, fontsize=10)
        #
        # # Axis formatting
        # buffer = 0.1
        # ax.set_xlim(reduced[:, 0].min() - buffer, reduced[:, 0].max() + buffer)
        # ax.set_ylim(reduced[:, 1].min() - buffer, reduced[:, 1].max() + buffer)
        # ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        # ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        # ax.set_title("PCA Visualization of Word Embeddings")
        # ax.grid(True, linestyle='--', alpha=0.5)
        # return fig
