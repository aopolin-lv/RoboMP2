from data_process.utils import read_json, get_instruction
from retrieval.similarity_retrieval import cal_TF_IDF
from .similarity_retrieval import cal_sentence_bert_cls

all_codebase_path = "data/codebase/all_codebase.json"

def coarse_grained_rank(instruction, topk):
    data = read_json(all_codebase_path)
    instruction_list = get_instruction(data)
    rough_rank_list,cosine_similarity = cal_TF_IDF(instruction,instruction_list,topk)
    return rough_rank_list, cosine_similarity

def fine_grained_rank(instruction,candidate_list):
    score_list = cal_sentence_bert_cls(instruction,candidate_list)
    return score_list