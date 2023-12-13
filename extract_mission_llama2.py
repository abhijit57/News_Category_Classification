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


# Loading Form 990 PF dataset and extracting the necessary columns
data_path = r'/geode2/home/u060/nayakab/Carbonate/GRA_Foundations_Grantees/2019_990PF_Foundations (compiled from 2020-2021 IRS File Extractions) w Vars and NTEE codes (646 missing)(N = 60,622)_2023.02.02.csv'
data = pd.read_csv(data_path, low_memory=False)
data_mod = data[['ein', 'business_name', 'ntee_full', 'website_address', 'Dscrptn1Txt', 'Dscrptn2Txt',
           'Dscrptn3Txt', 'Dscrptn4Txt', 'AllOthrPrgrmRltdInvstTtAmt', 'Dscrptn1Txt2', 'Dscrptn2Txt2',	
           'Grped_prg_srv_rev_desc', 'Grped_rltnshp_Sttmnt_txt', 'mission_unprepped'
]]
df = data_mod[data_mod['mission_unprepped'].isnull()][['ein', 'business_name', 'mission_unprepped']]
df.reset_index(drop=True, inplace=True)
# print(df.head())


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


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--start_idx', type=int)
  parser.add_argument('--end_idx', type=int)
  args = parser.parse_args()

  df_sample = df[args.start_idx:args.end_idx]

  print('\n')
  pattern = r'\{.*?\}'
  extracted_mission = []
  for i in tqdm(df_sample.business_name.to_list()):
    prompt = f"""
    Give me the mission statement of {i}.
    Format your response as a JSON object with "mission" as the key.
    """
    generated_text = generate(prompt)
    response = parse_text(generated_text)
    # print(response)
    # match = re.findall(pattern, response, re.DOTALL)
    # print(match)
    # proc_response = json.loads(match[0], strict=False).get('mission').replace('\n', ' ')
    # print(proc_response)
    extracted_mission.append(response)
    # print()


  # Save Data
  output_path = r'/N/u/nayakab/Carbonate/Prompt_Engineering/Mission_Results/Llama/'
  new_df = pd.DataFrame()
  new_df['ein'] = df_sample['ein']
  new_df['business_name'] = df_sample['business_name']
  new_df['extracted_mission_statement'] = extracted_mission
  fname = 'extract_' + str(args.start_idx) + '_' + str(args.end_idx) + '.csv'
  new_df.to_csv(os.path.join(output_path, fname), index=False)


