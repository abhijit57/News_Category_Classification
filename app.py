import pandas as pd
import numpy as np
import flask
import torch
from sklearn.preprocessing import LabelEncoder
from flask import request, render_template, jsonify
from transformers import BertForSequenceClassification, AutoTokenizer

device = "cuda"
app = flask.Flask(__name__)
model_path = "output/checkpoint-20952"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=42).to(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

test_df = pd.read_csv('news_test_data.csv')


@app.route('/', methods=['GET', 'POST'])  
def predict():
    if request.method == 'POST':
        text = request.form['text']

        if not text:
            return render_template('index.html', prediction="Please enter text.")

        try:
            # Tokenize text
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                # max_length=128,
                truncation=True,
                # pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"]
            input_ids = input_ids.to(device)
            
            # Make prediction
            output = model(input_ids)
            prediction = output.logits.argmax(-1).item()
        
            return render_template('index.html', prediction=f"The predicted class is: {prediction}")

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html', prediction=None)



if __name__ == '__main__':
    app.run(debug=True)


