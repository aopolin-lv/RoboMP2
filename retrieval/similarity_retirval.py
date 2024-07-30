from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random
import json
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

SENTENCEBERT_PATH = ""  # The path of SentenceBert
random.seed(3072)

retrival_mode_dict = {"instruction":"instruction",
                      "template":"template",
                      "annotation":"annotation",
                      "all":"all",
                      "instruction+annotation":"instruction+annotation",
                      }

retrival_mode = "instruction"
method_list = ["sentence_bert","BM25","TF_IDF"]

k = 3           
device_id = 0

def get_retrieval_list(retrival_mode,codebase):
    retrieval_list = []
    if retrival_mode == "instruction":
        for i in range(len(codebase)):
            retrieval_list.append(codebase[i]["instruction"])
    elif retrival_mode == "template":
        for i in range(len(codebase)):
            retrieval_list.append(codebase[i]["template"])
    elif retrival_mode == "annotation":
        for i in range(len(codebase)):
            retrieval_list.append(codebase[i]["annotation"])
    elif retrival_mode == "all":
        for i in range(len(codebase)):
            retrieval_list.append(codebase[i]["instruction"]+codebase[i]["annotation"]+codebase[i]["template"])
    elif retrival_mode == "instruction+annotation":
        for i in range(len(codebase)):
            retrieval_list.append(codebase[i]["instruction"]+codebase[i]["annotation"])

    return retrieval_list

def cal_BM25(query,codebase,k):
    # Calculate similarity using BM25
    bm25 = BM25Okapi(codebase)
    bm25_scores = bm25.get_scores(query)
    # Print the similarity after sorting
    sorted_indices_bm25 = np.argsort(bm25_scores)[::-1]
    return [codebase[i] for i in sorted_indices_bm25[:k]]

def cal_sentence_bert_cls(query_instruction,candidate_list):
    score = []
    model = SentenceTransformer(SENTENCEBERT_PATH)
    model.to(torch.device(f"cuda:{device_id}"))
    for candidate in candidate_list:
        embeddings = model.encode(query_instruction+candidate)
        score.append(sigmoid(embeddings[0]))
    return score

def cal_sentence_bert(query_instruction,codebase,k):
    assert SENTENCEBERT_PATH is not None, "The path of SentenceBert is NULL"
    model = SentenceTransformer(SENTENCEBERT_PATH)
    model.to(torch.device(f"cuda:{device_id}"))
    embeddings = model.encode(codebase)
    # Create Faiss Index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    # Calculate the embedding vector of the query instruction
    query_embedding = np.array(model.encode([query_instruction]))
    # Using Faiss for similarity retrieval
    D, I = index.search(query_embedding, k)
    # The three instructions with the highest similarity in printing
    most_similar_instructions = [codebase[i] for i in I[0]]
    return most_similar_instructions

def cal_TF_IDF(query_instruction,codebase,k):
    # Calculate similarity using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    corpus_tfidf = tfidf_vectorizer.fit_transform(codebase)
    query_tfidf = tfidf_vectorizer.transform([query_instruction])
    # Calculate cosine similarity
    cosine_similarity = np.dot(corpus_tfidf, query_tfidf.T).toarray().flatten()
    # Print the similarity after sorting
    sorted_indices_tfidf = np.argsort(cosine_similarity)[::-1]
    return [codebase[i] for i in sorted_indices_tfidf[:k]],[cosine_similarity[i] for i in sorted_indices_tfidf[:k]]




    


