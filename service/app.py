import streamlit as st
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import zipfile
import tempfile
import shutil
from pathlib import Path
import re
import pke
import spacy
from dataclasses import dataclass

import sys
sys.path.append('..')

from src.pdf_parser import Parser

import pickle

@dataclass
class ProfStandard:
    name: str
    rank: int
    rpds: dict

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        return super().find_class(module, 'ProfStandard')


RPD_EMBEDDINGS_PATH = "../data/processed/tbd_embeddings.pkl"
PROF_EMBEDDINGS_PATH = "../data/processed/prof_embeddings.pkl"
RPD_KEYWORDS = "../data/processed/tbd_keywords.json"
PROF_KEYWORDS = "../data/processed/prof_keywords.json"
SORTED_PROF_RANKS = "../data/sorted_prof_ranks.pkl"


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


@st.cache(allow_output_mutation=True)
def parse_zip(zip_file):
    with zipfile.ZipFile(zip_file) as zip_archive:
        temp_dir = Path(tempfile.mkdtemp())
        zip_archive.extractall(temp_dir)
        parser = Parser()
        docs = {}
        unzip_folder_name = zip_file.name.rsplit('.', 1)[0]
        for file_path in temp_dir.joinpath(unzip_folder_name).iterdir():
            if file_path.suffix in [".docx", '.pdf']:
                text = parser.parse(file_path, ext=file_path.suffix[1:])
                parts = []
                part1_span = re.search(r'1\.1\.?', text)
                part2_span = re.search(r'1\.2\.?', text)
                part3_span = re.search(r'1\.3\.?', text)
                part4_span = re.search(r'1\.4\.?', text)
                if not (part1_span and part2_span and part3_span and part4_span):
                    continue
                parts.append(text[part1_span.end():part2_span.start()])
                parts.append(text[part3_span.end():part4_span.start()])

                docs[file_path.stem] = '\n'.join(parts)

        shutil.rmtree(temp_dir)

    return docs


@st.cache(allow_output_mutation=True)
def get_keywords(docs):
    model = spacy.load("../ru2/")

    rpd_keywords = {}

    for x in docs:
        extractor = pke.unsupervised.SingleRank()
        extractor.load_document(input=docs[x], spacy_model=model)
        extractor.candidate_selection()
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=30)
        rpd_keywords[x] = keyphrases
    return rpd_keywords


@st.cache(allow_output_mutation=True)
def load_prof_embeddings():
    with open(PROF_EMBEDDINGS_PATH, "rb") as f:
        prof_embeddings = pickle.load(f)
    return prof_embeddings

@st.cache(allow_output_mutation=True)
def load_prof_ranks():
    return CustomUnpickler(open(SORTED_PROF_RANKS, 'rb')).load()


def main():
    # rpd_embeddings = load_rpd_embeddings()
    prof_embeddings = load_prof_embeddings()

    # rpd_keywords = load_rpd_keywords()
    # prof_keywords = load_prof_keywords()

    st.sidebar.title("Пакет документов для образовательной программы")
    zip_file = st.sidebar.file_uploader('', type='zip')
    # upload_button = st.sidebar.button("Загрузить")

    # rpd_keywords = {}
    # if zip_file and upload_button:
    #     docs = parse_zip(zip_file)
    #     rpd_keywords = get_keywords(docs)

    st.sidebar.title("Профстандарты")
    top_n = st.sidebar.number_input("TOP-N", min_value=1, value=20)
    display_keywords = st.sidebar.checkbox('Display keywords')
    find_button = st.sidebar.button("Найти")
    if zip_file and find_button:
        docs = parse_zip(zip_file)
        rpd_keywords = get_keywords(docs)
        prof_ranks = load_prof_ranks()

        rows = []
        for i, prof_standard in enumerate(prof_ranks[:top_n]):
            keywords = set()
            for values in prof_standard.rpds.values():
                keywords.update(list(values))
            row = f"| {i} | `{prof_standard.name}` | {prof_standard.rank} |"
            rows.append(row)

        table_rows = "\n".join(rows)
        table = f"""
        |  | Профстандарт | Score |
        | --- | ---: | ---: |
        {table_rows}
        """
        st.markdown(table)

        if display_keywords:
            st.title('Keywords')
            keywords = {}
            for prof_standard in prof_ranks[:top_n]:
                keywords[prof_standard.name] = set()
                for kw in prof_standard.rpds.values():
                    keywords[prof_standard.name].update(list(kw))
                keywords[prof_standard.name] = list(keywords[prof_standard.name])
            st.write(keywords)



if __name__ == "__main__":
    main()
