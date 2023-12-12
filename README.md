# News_Category_Classification

## Create a virtual environment and load the following packages mentioned in the requirements.txt file. Install the cuda version of torch 2.0.1 or higher by referring the pytorch documentation.

### The News_Category_Dataset.zip file contains news_train, news_test, news_val csv files derived from the parent news category dataset (in .txt format) downloaded from: https://www.kaggle.com/datasets/rmisra/news-category-dataset. The .txt parent dataset is larger than 25mb, so couldn't be uploaded to GitHub, hence provided the link.
### data_preparation.py:  This program does the above, reads the news_category.txt file and breaks it down into news train, test and val sets. This program performs data consistency checks and train test val splits. To run this program use this command: python data_preparation.py

### text_classification.py: This program reads the 3 csv files using hugging face datasets library, processes them and performs tokenization. Then, loads BertForSequenceClassification pre-trained model, using huggingface trainer sets training arguments. Finally, fine-tunes the BERT model using trainer object with training arguments defined in the previous step and performs all training and evaluations against the val set. The logs, graphs, system charts, artifacts were logged into Weights and Biases. The best model was observed at the 4th epoch (checkpoint-20952) with an f1 score of 61.7% on the validation set. To run this program: python text_classification.py
Weights and Biases Report Uploaded in the main branch with the filename "W&B Report".

### inference.py: This program runs the predictions using the trained BERT Model (4th epoch: checkpoint-20952) on the complete dataset on a single GPU for a test dataset of 25141 records. The total time taken was 3.12 minutes.
![image](https://github.com/abhijit57/News_Category_Classification/assets/44730823/4e14f03c-c489-4222-a3f5-737d8d182302)

### optimized_inference.py: This program runs the predictions using the trained BERT model on the complete test dataset on 4 GPUs and the total time taken was 2.06 minutes. It used multi-gpu inferencing compared to the previous inference process. This highlights a speedup factor of 33%.
![image](https://github.com/abhijit57/News_Category_Classification/assets/44730823/624c540a-5cfa-44b5-9720-eebc64bcce35)

### Run both the programs using respective commands: python inference.py, python optimized_inference.py

### Flask App: The templates folder contains the index.html file. app.py python script uses the index.html file to render a simple UI to perform news category text classification by taking a text box input and providing the category class as the prediction. Run the flask app: python app.py and click on the URL provided in the console.


#### Application Execution Order:
1) Clone the repo, create virtual environment preferably conda, load all the libraries mentioned in the requirements.txt file. Activate the virtual environment and execute the below mentioned files.
2) data_preparation.py (can be skipped, if you want to run the text classification script directly from the csv files. Run this file only if you want to read it from the txt file, perform data consistency checks and train, test splits).
3) text_classification.py
4) inference.py
5) optimized_inference.py
6) app.py




