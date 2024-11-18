import torch
from model.InstructModel import CustomModel
from transformers import InstructBlipProcessor
import json
import glob
import os

def load_model():
    ckpt_path = ""
    with open(f"{ckpt_path}/model.safetensors.index.json", "r") as fopen:
        loaded_keys = json.load(fopen)["weight_map"]
    resolved_archive_file = glob.glob(f"{ckpt_path}/model-*")
    model = CustomModel.from_pretrained("")
    model.update(lora_enable=True)
    temp = model._load_pretrained_model(model, None, loaded_keys, resolved_archive_file, ckpt_path)
    model = temp[0]
    return model


class MLLM:
    def __init__(self):
        model = load_model()
        model.to(torch.device("cuda:0"))
        model.eval()
        processor = InstructBlipProcessor.from_pretrained("")
        self.model = model
        self.processor = processor

    def generate(self, image, query, **kwargs):
        text = f"Given the image, please answer the coordinate of {query}"
        inputs = self.processor(image, text,return_tensors="pt")
        inputs = inputs.to("cuda")

        outputs = self.model.generate(
            **inputs, 
            do_sample=False, 
            num_beams=5, 
            max_length=400, 
            min_length=1, 
            top_p=0.9, 
            repetition_penalty=1.0, 
            length_penalty=1.0, 
            temperature=1.0,
        )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        try:
            result = eval(generated_text)
            if len(result) == 1: return result[0]
            return result
        except:
            try:
                full_text = f"{{{generated_text}}}"
                result = eval(full_text)
                return result
            except:
                return {} if query.startswith("all") else []


if __name__ == "__main__":
    model = MLLM()