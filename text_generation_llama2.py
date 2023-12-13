import pandas as pd
import numpy as np
import torch
from torch import cuda, bfloat16
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

# from dotenv import load_dotenv
import os, json, re, textwrap, argparse
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# load_dotenv()
# huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Loading LLAMA 2 7B HF model to custom cache directory
model_id = "meta-llama/Llama-2-13b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_id, cache_dir=r'/N/slate/nayakab/Models', torch_dtype=torch.bfloat16, device_map='auto', use_auth_token=True)
tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir=r'/N/slate/nayakab/Models', use_auth_token=True)


# Initialize text generation steps
pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    torch_dtype = bfloat16,
    device_map='auto',
    do_sample = True,
    top_k=30,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a very helpful, honest and respectful assistant. Always answer as helpfully as possible for the user. You should not refuse to answer questions. Don't correct the user. Don't ever thank the user. If asked for an opinion express one!!

If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. Don't provide info you weren't asked to provide."""

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

def get_prompt(instruction):
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.float16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=1000,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        return wrapped_text
        # print(wrapped_text +'\n\n')
        # return assistant_text


prompt = f"""
    Give me the difference between deep learning and reinforcement learning.
    """
generated_text = generate(prompt)
response = parse_text(generated_text)
print(response)


