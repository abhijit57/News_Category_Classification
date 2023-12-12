import pandas as pd
import numpy as np
import torch
import flask

from torch.utils.data import TensorDataset, DataLoader, random_split, SequentialSampler, RandomSampler
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import Trainer
from datasets import load_dataset
from time import time
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


device = "cuda"
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model_path = "output/checkpoint-20952"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=42).to(device)
test_df = pd.read_csv('news_test_data.csv')

start = time()
attention_masks, input_ids = None, None
predictions = []
for i in tqdm(range(len(test_df))):
    sent = test_df.iloc[i]['news']
    encoded_dict = tokenizer.encode_plus(
                        sent,                                # sentences to encode
                        add_special_tokens=True,             # adding [CLS] and [SEP] tokens respectively at the start and end of the sentences
                        truncation=True,
                        return_attention_mask=True,          # construct attention masks
                        return_tensors='pt'                  # Return pytorch tensors
                    )

    # Add the encoded sentences to the list
    input_ids = encoded_dict['input_ids']
    input_ids = input_ids.to(device)
    attention_masks = encoded_dict['attention_mask']
    attention_masks = attention_masks.to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
        _, preds = torch.max(outputs[0], dim=1)
        predictions.append(preds.tolist()[0])
end = time()
print(f'Total time taken for total number of {len(predictions)} predictions: {round((end-start)/60,2)} minutes.')



