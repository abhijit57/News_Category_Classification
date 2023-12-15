import pandas as pd
import numpy as np
import torch
import flask

from torch.utils.data import TensorDataset, DataLoader, random_split, SequentialSampler, RandomSampler
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import Trainer
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from time import time
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


device = "cuda"
## Load dataset
news_test = load_dataset(
                          'csv',
                          data_files=r'/N/u/nayakab/Carbonate/News Category Classification_NLP/NCRI/news_test_data.csv'
)
news_test['test'] = news_test['train']
news_test.pop('train')
test_label_ids = LabelEncoder().fit_transform(news_test['test']['category']).tolist()
news_test = news_test['test'].add_column(name='labels', column=test_label_ids)


# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def tokenize_func(sents):
    return tokenizer(
        sents['news'], 
        padding="max_length", 
        truncation=True,
        max_length=512
    )
tokenized_test_ds = news_test.map(tokenize_func, batched=True)
tokenized_test_ds = tokenized_test_ds.remove_columns(['news', 'category'])
tokenized_test_ds = tokenized_test_ds.with_format("torch")


# Get number of GPU device
n_gpus = torch.cuda.device_count()
model_path = "output/checkpoint-20952"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=42).to(device)

# Make predictions
start = time()
test_trainer = Trainer(model)
raw_pred, _, _ = test_trainer.predict(tokenized_test_ds)
y_pred = np.argmax(raw_pred, axis=1)
end = time()
print('\n')
print(f'Total Time taken for {len(y_pred)} predictions: {round((end-start)/60, 2)} minutes.')