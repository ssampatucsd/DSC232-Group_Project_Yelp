# DSC232: Yelp Reviews

## Overview
This repository contains the final project for DSC232, featuring a thorough analysis and the creation of predictive models aimed at predicting 'stars' from user review text. Within, you'll find datasets, analytical scripts, model training algorithms, and results designed to offer companies deeper insights into their customers' experiences. Comprehensive documentation is available to help you navigate each aspect of the project.

Below is the writeup for this project. Here is the link to our Jupyter Notebook.

## 1. Introduction
Yelp Inc. is a widely-used online platform where users can search for and review local businesses. Founded in 2004, it acts as a directory for various businesses, including restaurants, retail stores, services, and more. Yelp provides details about the business such as business hours, contact information, menus, photos, and user-generated reviews. Businesses can claim their Yelp profiles to interact with customers, post updates, and respond to reviews.

Yelp has several positive impacts on businesses, such as increasing visibility, establishing credibility and trust, engaging with customers, gaining market insights, and providing advertising opportunities. For instance, Yelp can help businesses get discovered by new customers, driving foot traffic and online inquiries. However, there are also negative impacts, such as the potential for negative reviews to harm a business's reputation and revenue.

This project aims to predict a business's rating out of five stars based on user reviews. We use machine learning techniques to address the challenge of text classification. This predictive goal can provide businesses with insights into what customer experiences are associated with positive and negative reviews. The project is interesting because of its potential positive impact on both businesses and customers. It can enhance the success of businesses and improve customer decision-making, as customers often rely on reviews for their purchasing decisions.

The dataset used in this project is notable for its size and comprehensive information. It is a subset of Yelp's data, containing information about businesses, reviews, and users across eight metropolitan areas in the USA and Canada. The data, sourced directly from Yelp.com, includes 7 million customer reviews and features 150,000 businesses. The credibility, size, and richness of the data were the main reasons for choosing this dataset.

## 2. Figures

## 3. Methods
### 3.1 Data Exploration
The Yelp API contains a variety of different datasets about the businesses, reviews, and users. For this analysis, we specifically used the review dataset. The dataset is obtained from Kaggle, available at this link https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset. To ensure ease of access, it is also replicated within this repository.

The review dataset consistents of the following columns:

`review id`: string, 22 character unique review id

`user_id`: string, 22 character unique user id

`business_id`: string, 22 character unique business id

`stars`: integer, star rating (1-5)

`date`: string, date formatted YYYY-MM-DD

`text`: string, the user's review

`useful`: integer, number of useful votes received

`funny`: integer, number of funny votes received

`cool`: integer, number of cool votes received

The data exploration consists of serveral parts:


### 3.2 Data Preprocessing
1. Selecting features: Since we are trying to predict 'stars' based on 'text' we only select these features. 
2. Addressing Missing Values: Eliminate rows containing missing values. In most cases, missing values occured when a user left a star rating on a business, but failed to include a written review. These rows were irrelevant to our specific analysis.
3. Add features: We added feature 'review_length', which is the length of a user's text.
4. Tokenization: Split text into individual words. This is useful in simplifying text processing and enhancing machine learning models. It also improves text analysis and helps in understanding the syntax and semantics of the text.
5. Stop Words Removal: Removal of common words from a text that have little to no semantic value in the context of the specific text processing task. This reduces dimensionality, decreases noise and therefore improves model performance, enhances text analysis, and facilitates efficient storage.
6. Stemming:  Reduce words to theirr base or root form. The goal was to treat different forms of a word as the same word in order to simplify text processing and analysis. This reduces dimensionality, improves search and information retrieval, and enhances text analysis.
7. Sentiment Polarity Scores: Generate numerical values that indicate the sentiment or emotional tone of a piece of text. This contributes to insights into customer sentiments and is helpful in predictive analytics.
8. Term Frequency-Inverse Document Frequency (TF-IDF): Generate a numeric value that reflects the importance of a word in a document relative to a collection of documents. This is useful for feature selection, reducing noise, and information retrieval.
   
### 3.3 Model 1: Logistic Regression
### 3.4 Model 2: Support Vector Machine

## Results
### 4.1 Data Exploration
### 4.2 Data Preprocessing
### 4.3 Model 1: Logistic Regression
### 4.4 Model 2: Support Vector Machine
### 4.5 Compare Model Performances

## 5. Discussion
### Data Exploration: Exploratory Data Analysis
### Data Preprocessing
### Model 1: Logistic Regression
### Model 2: Support Vector Machine
### General Discussion
#### Success
#### Limitations

## 6. Conclusion

## 7. Collaboation

