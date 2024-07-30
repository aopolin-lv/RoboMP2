import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from transformers.trainer_utils import set_seed
import torch
from .similarity_retirval import *

FULL_PROMPT_FILE_DICT = {
    "full_examples": "visual_programming_prompt/full_prompt.ini",
    "five_examples": "visual_programming_prompt/five_examples.ini",
    "disturb_examples": "visual_programming_prompt/disturb_example.ini",
    "one_examples": "visual_programming_prompt/most_relevant.ini",
    "no_example":"visual_programming_prompt/no_examples.ini"
}
PROMPT_FILE_DICT = {
    # "full_examples": "visual_programming_prompt/full_prompt.ini",
    "five_examples": "visual_programming_prompt/i2a_prompt/five_examples.ini",
    # "disturb_examples": "visual_programming_prompt/disturb_example.ini",
    "one_examples": "visual_programming_prompt/i2a_prompt/most_relevant.ini",
    # "no_example":"visual_programming_prompt/no_examples.ini"
}

FULL_PROMPT_CUSTOM_FILE_DICT = {
    "topk":"visual_programming_prompt/no_examples.ini",
    "one_example": "visual_programming_prompt/most_relevant.ini",
    "full_examples": "visual_programming_prompt/full_prompt.ini",
    
}
RESOLVED_PROMPT_CUSTOM_FILE_DICT = {
    "tools": "visual_programming_prompt/new_template/tools.ini",
    "examples": "visual_programming_prompt/new_template/examples.ini",
    "instruction":"visual_programming_prompt/new_template/instruction.ini"
}


def cosine_similarity(a, b):
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    similarity_matrix = dot_product / np.outer(norm_a, norm_b)
    return similarity_matrix

def scaled_dot_product_attention(Q, K):
    # attention_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(torch.tensor(Q.shape[0]))
    attention_scores = cosine_similarity(Q.cpu().numpy(),K.cpu().numpy())   
    return attention_scores


def draw_attention(hidden_state,method_list,round_num,size,model_name,prompt_name,task,template_owner):

    attention_scores = scaled_dot_product_attention(hidden_state,hidden_state)
    data = attention_scores.tolist()
    average_score = np.mean(attention_scores)
    variance = np.var(attention_scores)
    norm_data = (attention_scores-average_score)/np.sqrt(variance)   
    plt.imshow(norm_data, cmap='RdPu', interpolation='nearest')
    plt.xticks(list(range(len(attention_scores.tolist()[0]))), list(method_list.values()),fontsize=5,rotation=90)
    plt.yticks(list(range(len(attention_scores.tolist()[0]))), list(method_list.values()),fontsize=5)
    for i in range(len(attention_scores.tolist()[0])):
        for j in range(len(attention_scores.tolist()[0])):
            color = "white" if  attention_scores.tolist()[i][j] > average_score else "black"
            plt.text(j, i, round(norm_data.tolist()[i][j], round_num), color=color,fontsize=size, ha="center", va="center", weight='bold')

    plt.colorbar()

    if template_owner == "i2a":
        folder_path = ""
    elif template_owner == "custom":
        folder_path = ""
    path = os.path.join(folder_path,model_name.split("/")[-1])
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path,prompt_name)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,f'{task}.png'), dpi=400)

    plt.clf()
    print(1)

def three_parts_index(prompt):

    def_index = {}
    index_get_obs_image = prompt.find("IMPLEMENTED TOOLS")
    index_example_1 = prompt.find("## Example 1")
    def_index[1] = "THIRD PARTY TOOLS"
    def_index[index_get_obs_image] = "IMPLEMENTED TOOLS"
    task_index = prompt.find("Begin to execute the task")
    i = 1
    while(True):
        method_name = "def main_"+ str(i)
        index_example = prompt.find(method_name)
        if i > 17:
            break
        if index_example == -1:
            i = i+1 
            continue
        if index_example > task_index:
            break
        
        def_index[index_example] = method_name
        i=i+1
        
    def_index[task_index] = "Begin to execute the task"
    output_index = prompt.find("output fewer lines") + 40
    def_index[output_index] = "model_output"
    index_answer = prompt.rfind("def main()")
    if index_answer!=-1:
        def_index[index_answer] = "def main()"
    return def_index

def get_token_index(index_dict: dict, target_indices: list):
    result_keys = []
    for target_index in target_indices:
        closest_key = min(index_dict.keys(), key=lambda key: abs(target_index - index_dict[key][0]))
        result_keys.append(closest_key)
    return result_keys

def change_form(prompt_index_dic,token_list):
    new_dict = {value: token_list[i] for i, value in enumerate(prompt_index_dic.values())}
    return new_dict

def change_key_value(dict):
    return {value: key for key, value in dict.items()}

def attention_map(prompt,token_list):
    def_index = three_parts_index(prompt)
    def_index_list = list(def_index.keys())
    token_list = get_token_index(token_list,def_index_list)
    final_index = change_form(def_index,token_list)
    return change_key_value(final_index)

def get_full_i2a_prompt_file(prompt_type,task):
    full_prompt_file = PROMPT_FILE_DICT[prompt_type]
    if prompt_type == "one_examples":
        if "rotate" in task:
            full_prompt_file = ""
        elif "pick" in task:
            full_prompt_file = ""
        elif "rearrange_then" in task:
            full_prompt_file = ""
        elif "scene" in task:
            full_prompt_file = ""
    with open(full_prompt_file, "r") as f:
        full_prompt_i2a = f.readlines()
    full_prompt_i2a = "".join(full_prompt_i2a)
    return full_prompt_i2a

def get_codebase(codebase_path):
    with open(codebase_path,"r") as f:
        codebase = json.load(f)
    return codebase
    
def get_topk_examples_and_relevant_api_prompt(rough_rank_list,sorted_indices):
    task_index_dict = {
        "Put the {object1} into the {object2}" : "visual_manipulation",
        "Put the {texture1} object in {scene} into the {texture2} object" : "scene_understanding",
        "Rotate the {dragged_obj}" : "rotate",
        "Rearrange to this {scene}." : "rearrange",
        "Rearrange to this {scene} then restore." : "rearrange_then_restore",
        "{demo object1} is {novel adj} than {demo object2}. {demo object3} is {novel adj} than {demo object4}." : "novel_adj",
        "This is a {novel name1} {object1} . This is a {novel name2} {object2}." : "novel_noun",
        "This is a {novel name1} {object1}. This is a {novel name2} {object2}. {demo object1} is {adj} than {demo object2}." : "novel_adj_and_noun",
        "defined as rotating object a specific angle" : "twist",
        "Follow this motion" : "follow_motion",
        "Stack objects" : "follow_order",
        "Sweep" : "sweep_without_exceeding",
        "Put all objects with the same texture" : "same_texture",
        "Put all objects with the same shape" : "same_shape",
        "First put" : "manipulate_old_neighbor",
        "Finally restore" : "pick_in_order_then_restore",
    }
    # Retrieve the coarse-grained taskname
    task_name_list = []
    for rough_rank_item in rough_rank_list:
        task_index_list = list(task_index_dict.keys())
        for task_index in task_index_list:
            if task_index in rough_rank_item:
                task_name_list.append(task_index_dict[task_index])
                break

    # Retrieve the code_name list corresponding to task_name
    union_method_list = set()
    with open("data/codebase/method_apibase.json", "r") as f:
        method_apibase = json.load(f)
        for task_name in task_name_list:
            method_api = method_apibase[task_name]
            union_method_list = union_method_list.union(set(method_api))
    union_method_list = list(union_method_list)

    # Get the API description corresponding to each method
    with open("data/codebase/apibase.json", "r") as f:
        method_apibase = json.load(f)


    # Splicing three levels of tools in sequence
    first_level_method_list = ["GetObsImage", "GetPromptImages", "GetSceneImage", "GetPromptAssets", "GetWholeTask"]
    with open("data/prompt_base/first_level.ini","r") as f:
        first_level = f.readlines()
    first_level.append("\n")
    first_level.append("\n")
    for first_level_method in first_level_method_list:
        if first_level_method in union_method_list:
            first_level = first_level + method_apibase[first_level_method]
            first_level.append("\n")
    # first_level.append("\n")

    second_level_method_list = ['MMPROBO.generate', 'GetAllObjectsFromImage', 'GetAllObjectsFromPromptImage', 'GetAllSameTextureObjects', 'GetAllSameProfileObjects', 'GetSameShapeObjectsAsObject', 'GetObjectsWtihGivenTexture', 'GetDragObjName', 'GetBaseObjName', 'PickPlace', 'DistractorActions', 'RearrangeActions', 'SelectFromScene', 'GetStepsLocForObject', 'GetAllObjextsStepsLoc', 'SelectObj', 'RotateAll', 'RecDegree', 'RecAdj', 'PlanOrder', 'CalculateValidArea', 'MultiPPWithConstrain', 'MultiPPWithSame', 'GetReverse', 'Sorted', 'GetSweptAndBoundaryName', 'GetTimes']
    with open("data/prompt_base/second_level.ini","r") as f:
        secont_level = f.readlines()
    secont_level.append("\n")
    for second_level_method in second_level_method_list:
        if second_level_method in union_method_list:
            secont_level = secont_level + method_apibase[second_level_method]
    secont_level.append("\n")

    with open("data/prompt_base/third_level.ini","r") as f:
        third_level = f.readlines()
    third_level.append("\n")

    tools_prompt_custom = first_level + secont_level + third_level
    tools_prompt_custom = "".join(tools_prompt_custom)

    # The original code
    example_index=1
    # Obtain the codebase
    codebase = get_codebase("data/codebase/all_codebase.json")

    # with open(tools_path, "r") as f:
    #     tools_prompt_custom = f.readlines()
    # tools_prompt_custom = "".join(tools_prompt_custom)
    instruction_path = "data/prompt_base/instruction.ini"
    with open(instruction_path, "r") as f:
        instruction_prompt_custom = f.readlines()
    instruction_prompt_custom = "".join(instruction_prompt_custom)
    sorted_list = []
    for i in sorted_indices:
        sorted_list.append(rough_rank_list[i])
    examples_custom = ""
    for i in range(len(sorted_list)):
        for j in range(len(codebase)):
            if codebase[j]["instruction"] in sorted_list[i]:
                examples_custom = examples_custom + "## Example "+str(example_index)+"\n"+"# Instruction: "+codebase[j]["instruction"]+"\n"+'"""'+codebase[j]["annotation"]+'"""'+"\n"+''.join(codebase[j]["template"])+"\n"
                example_index = example_index + 1
    return tools_prompt_custom + examples_custom + instruction_prompt_custom



def load_model(model_path):
    # load model
    set_seed(42)
    config_kwargs = {
        "output_hidden_states": True,
        # "output_attentions":True
    }
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    config=config,
    # device_map="auto",
    )
    model.to("cuda")
    model.eval()
    return model,tokenizer