# News_Category_Classification



### The News_Category_Dataset.zip file contains news_train, news_test, news_val csv files derived from the parent news category dataset (in .txt format) downloaded from: https://www.kaggle.com/datasets/rmisra/news-category-dataset. The .txt parent dataset is larger than 25mb, so couldn't be uploaded to GitHub, hence provided the link.
### data_preparation.py:  This program does the above, reads the news_category.txt file and breaks it down into news train, test and val sets. This program performs data consistency checks and train test val splits. To run this program use this command: python data_preparation.py

### text_classification.py: This program reads the 3 csv files using hugging face datasets library, processes them and performs tokenization. Then, loads BertForSequenceClassification pre-trained model, using huggingface trainer sets training arguments. Finally, fine-tunes the BERT model using trainer object with training arguments defined in the previous step and performs all training and evaluations against the val set. The logs, graphs, system charts, artifacts were logged into Weights and Biases. To run this program: python text_classification.py
Link to Weights and Biases Report: 
