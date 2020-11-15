import streamlit as st
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

RPD_EMBEDDINGS_PATH = "../data/processed/tbd_embeddings.pkl"
PROF_EMBEDDINGS_PATH = "../data/processed/prof_embeddings.pkl"
RPD_KEYWORDS = "../data/processed/tbd_keywords.json"
PROF_KEYWORDS = "../data/processed/prof_keywords.json"


@st.cache(allow_output_mutation=True)
def load_rpd_embeddings():
    with open(RPD_EMBEDDINGS_PATH, "rb") as f:
        rpd_embeddings = pickle.load(f)
    return rpd_embeddings


@st.cache(allow_output_mutation=True)
def load_prof_embeddings():
    with open(PROF_EMBEDDINGS_PATH, "rb") as f:
        prof_embeddings = pickle.load(f)
    return prof_embeddings


@st.cache(allow_output_mutation=True)
def load_rpd_keywords():
    with open(RPD_KEYWORDS) as f:
        rpd_embeddings = json.load(f)
    return rpd_embeddings


@st.cache(allow_output_mutation=True)
def load_prof_keywords():
    with open(PROF_KEYWORDS) as f:
        rpd_embeddings = json.load(f)
    return rpd_embeddings


def compute_similarity(rpd_embeddings, prof_embeddings, top_n):
    rpd_names, rpd_embeddings = zip(*rpd_embeddings.items())
    prof_names, prof_embeddings = zip(*prof_embeddings.items())
    similarities = cosine_similarity(rpd_embeddings, prof_embeddings)
    result = {}
    for i, rpd_name in enumerate(rpd_names):
        idxs = argsort(similarities[i], top_n=top_n, reverse=True)
        result[rpd_name] = [(prof_names[idx], similarities[i, idx]) for idx in idxs]
    return result


def argsort(x, top_n=None, reverse=False):
    x = np.asarray(x)
    if top_n is None:
        top_n = x.size
    if top_n <= 0:
        return []
    if reverse:
        x = -x
    most_extreme = np.argpartition(x, top_n)[:top_n]
    return most_extreme.take(np.argsort(x.take(most_extreme)))


def main():
    rpd_embeddings = load_rpd_embeddings()
    prof_embeddings = load_prof_embeddings()

    rpd_keywords = load_rpd_keywords()
    prof_keywords = load_prof_keywords()

    st.sidebar.title("Compute similarity")
    top_n = st.sidebar.number_input("TOP-N", value=10)
    find_button = st.sidebar.button("Find")

    st.sidebar.title("Keywords")
    selected_rpd = st.sidebar.selectbox("RPD", list(rpd_embeddings.keys()))
    rpd_keywords_button = st.sidebar.button("Display", key="rpd")

    selected_prof_standard = st.sidebar.selectbox(
        "Prof standard", list(prof_embeddings.keys())
    )
    prof_keywords_button = st.sidebar.button("Display", key="prof")

    if find_button:
        st.title("RPD to prof. standard matching")
        result = compute_similarity(rpd_embeddings, prof_embeddings, top_n)
        st.write(result)

    if rpd_keywords_button:
        st.title("Keywords")
        st.header(f"RPD: {selected_rpd}")
        st.write(rpd_keywords[selected_rpd])

    if prof_keywords_button:
        st.title("Keywords")
        st.header(f"Prof standard: {selected_prof_standard}")
        st.write(prof_keywords[selected_prof_standard])


if __name__ == "__main__":
    main()
