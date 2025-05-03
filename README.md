# Detection of AI-Generated Tweets

## Models

The pipeline developed in this final coursework consists of the following models:

- **Baseline model:** This is provides the baseline performance for the NLP pipeline that intends to improve upon by fine-tuning other variants of the pre-trained BERT model. The code for this model can be found in the `baseline.ipynb` file, and running it will yield the baseline results.
- **BERT model:** This model is a pre-trained model, and it is used to compare the performance of the baseline model. The code for this model can be found in the `bert.ipynb` file, and running it will fine-tune it based on the data and export the model once the fine-tuning is complete.
- **DistilBERT model:** This model is a distilled version of the BERT model, and it is used to compare the performance of the baseline model. The code for this model can be found in the `distilbert.ipynb` file, and running it will fine-tune the pre-trained model based on the data and export the model at the end.

## Data

The data used in this coursework is the TweepFake dataset, which is publicly available on [Kaggle](https://www.kaggle.com/datasets/danofer/tweepfake-dataset). However, the size of the dataset is not too large, so it is provided in the submission and in the GitHub repository within the `data` folder.
