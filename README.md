# DSC232-Group_Project_Yelp

### Create Environment and Install Dependencies:
- The "requirements.txt" file can be used to install all the necessary packages for all the notebooks in this course.
- Navigate to the folder location where you have downloaded this repository. Then, run "pip install -r requirements.txt". This should create your environment with all necessary packages.

### Link for data download: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

### Steps Taken for Pre-Processing:
Our data cleaning and preprocessing phase includes the following steps: handling null values, tokenization, stop words removal, stemming, and TF-IDF. The dataset consists of two columns: text and stars. If the text is missing, it means no review was written, so we simply removed these null values. Tokenization and stop words removal are essential preprocessing steps in Natural Language Processing and text mining. Stop words add little semantic value. Removing these words helps reduce the dimensionality of the text data. We also performed stemming to group words with the same base meaning together. This approach can improve the performance of the model by treating different forms of a word as the same term. 

### Training and evaluation of the first model:
Our dataset was trained on a multinomial logistic regression model. We performed an 80/20 train test split. Our features included review length, sentiment polarity, hashing TF, and TF-IDF. After training we predicted on the test set to obtain the test accuracy.

### Where does your model fit in the fitting graph? 
Our first initial model has a test error rate of 37.7%. 

### What are the next models you are thinking of and why?
We are considering multinomial naive bayes and SVMs as our next models. Multinomial naive bayes are well suited for text classification tasks and can handle multi-class classification. SVMs are also effective for text classification.

### Conclusion: What is the conclusion of your 1st model? What can be done to possibly improve it?
Our initial model has an error rate of 37.7%. To enhance its performance, we plan to remove some potentially irrelevant features. Specifically, we aim to retrain the model using only the TF-IDF and sentiment polarity features. Additionally, we will experiment with different types of models to see if accuracy improves.


