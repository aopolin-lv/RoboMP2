import openai
from openai import OpenAI
from data_process.utils import *
import base64
from PIL import Image
import io

# Get the prompt for the enhanced template
CONTENT = "Now you are given a template,\
rewrite the template in ten different ways to keep the meaning intact, such as reformatting the sentence appropriately, \
or replacing the verb with the corresponding synonym or synonymous phrase, Do not change {{}} and its contents. Given template: {template}"
SAVE_FOLDER = "data"

def encode_array(image_array):
    image = Image.fromarray(image_array)
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format='JPEG')
    image_bytes = image_byte_array.getvalue()

    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
    return base64_encoded

def gpt4v(content,array_image):
    image = encode_array(array_image)
    client = OpenAI(
    base_url="", # Input your own key.
    api_key="",
    )

    completion = client.chat.completions.create(
    model = "gpt-4-vision-preview",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": content},
            {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            },
        ],
        },
    ],
    max_tokens=4096,
    )
    return completion.choices[0].message.content

def gpt4(content):

    client = OpenAI(
    base_url="",
    api_key="",
    )
    completion = client.chat.completions.create(
    model = "gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": content},
        ],
    )
    return completion.choices[0].message.content