from data_process.gptutils import gpt4

rewrite_prompt_path = "data/rewrite.txt"

rewrite_dic = {}

def read_rewrite_prompt(path):
    with open(path, 'r') as file:
        content = file.read()
    return content

def insert_instruction(rewrite_prompt,instruction):
    return rewrite_prompt.replace("Insert your instruction",instruction)

def rewrite(instructrion):
    prompt = read_rewrite_prompt(rewrite_prompt_path)
    prompt = insert_instruction(prompt,instructrion)
    result = gpt4(prompt)
    return result



