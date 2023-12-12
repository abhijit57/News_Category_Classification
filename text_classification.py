import pandas as pd
import numpy as np
import evaluate
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, EarlyStoppingCallback
from datasets import load_dataset, load_metric
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


device = "cuda"
news_train = load_dataset(
                          'csv', 
                          data_files=r'/N/u/nayakab/Carbonate/News Category Classification_NLP/NCRI/news_train_data.csv'
)
train_label_ids = LabelEncoder().fit_transform(news_train['train']['category']).tolist()
news_train = news_train['train'].add_column(name='labels', column=train_label_ids)
news_val = load_dataset(
                          'csv',
                          data_files=r'/N/u/nayakab/Carbonate/News Category Classification_NLP/NCRI/news_val_data.csv'
)
news_val['val'] = news_val['train']
news_val.pop('train')
val_label_ids = LabelEncoder().fit_transform(news_val['val']['category']).tolist()
news_val = news_val['val'].add_column(name='labels', column=val_label_ids)
print(news_train, news_val)
print('Datasets Loaded.........')
print('\n')


# Tokenizing the datasets
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def tokenize_func(sents):
    return tokenizer(
        sents['news'], 
        padding="max_length", 
        truncation=True,
        max_length=512
    )

def format_tokenized_dataset(dataset):
    tokenized = dataset.map(tokenize_func, batched=True)
    tokenized = tokenized.remove_columns(['news', 'category'])
    # tokenized = tokenized.rename_column("category", "labels")
    tokenized = tokenized.with_format("torch")
    return tokenized

print("Tokenizing Datasets ------------")
tokenized_train_ds = format_tokenized_dataset(news_train)
tokenized_val_ds = format_tokenized_dataset(news_val)
print("Tokenization Complete.........")
print('\n')


## Fine-tune pretrained model
# Get number of GPU device
n_gpus = torch.cuda.device_count()
# Initialize Model
model = BertForSequenceClassification.from_pretrained(
# model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-cased', 
     num_labels=42,
).to(device)

# Evaluation Metrics
metric = evaluate.combine(["accuracy", "f1"])
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions, average='macro')
    precision = precision_score(y_true=labels, y_pred=predictions, average='macro')
    f1 = f1_score(y_true=labels, y_pred=predictions, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    seed=0,
    learning_rate=2e-5,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    load_best_model_at_end=True,
    use_cpu=False
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
model_output_dir = "./news_model"
trainer.train()
trainer.save_model(model_output_dir)
trainer.evaluate()

