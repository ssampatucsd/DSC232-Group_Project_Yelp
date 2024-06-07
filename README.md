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

The Business DataFrame consists of the following columns:
- `business_id`: string, 22 character unique business id
- `name`: string, the business's name
- `address`: string, the full address of the business
- `city`: string, the name of the city the business is located in
- `state`: string, 2 character state code, if applicable
- `postal_code`: string, the postal code the business is located in
- `latitude`: float, latitude
- `longitude`: float, longitude
- `stars`: float, star rating, rounded to half stars
- `review_count`: integer, number of reviews
- `is_open`: integer, 0 if business is closed, 1 if business is open
- `attributes`: object, business attributes to values 
- `categories`: array of strings of business categories
- `hours`: object of key day to value hours, hours use a 24 hour clock

The User DataFrame consists of the following columns:
- `user_id`: string, 22 character unique user id, maps to the user in user.json
- `name`: string, the user's first name
- `review_count`: integer, the number of reviews each user has written
- `yelping_since`: string, when the user joined Yelp, formatted as YYYY-MM-DD
- `friends`: array of strings, an array of the user's friends as user_ids
- `useful`: integer, number of useful votes sent by the user
- `funny`: integer, number of funny votes sent by the user
- `cool`: integer, number of cool votes sent by the user
- `fans`: integer, number of fans the user has
- `elite`: array of integers, the years the user was elite
- `average_stars`: float, average rating of all reviews
- `compliment_hot`: integer, number of hot compliments received by the user
- `compliment_more`: integer, number of more compliments received by the user
- `compliment_profile`: integer, number of profile compliments received by the user
- `compliment_cute`: integer, number of cute compliments received by the user
- `compliment_list`: integer, number of list compliments received by the user
- `compliment_note`: integer, number of note compliments received by the user
- `compliment_plain`: integer, number of plain compliments received by the user
- `compliment_cool`: integer, number of cool compliments received by the user
- `compliment_funny`: integer, number of funny compliments received by the user
- `compliment_writer`: integer, number of writer compliments received by the user
- `compliment_photos`: integer, number of photo compliments received by the user

The Review DataFrame consists of the following columns:
- `review id`: string, 22 character unique review id
- `user_id`: string, 22 character unique user id
- `business_id`: string, 22 character unique business id
- `stars`: integer, star rating (1-5)
- `date`: string, date formatted YYYY-MM-DD
- `text`: string, the user's review
- `useful`: integer, number of useful votes received
- `funny`: integer, number of funny votes received
- `cool`: integer, number of cool votes received

To arrive at our conclusion of using the review dataset, we had to explore what information was contained in each dataset. The data exploration consists of serveral parts:

Review DataFrame: Explore text statistics
- Observation count: Identify the total number of reviews.
- Column examination: Extract the column names and their respective types.
- Missing data analysis: Detect missing values in the dataset. Missing data should not just be ignored. Often, there are underlying reasons for why this is occuring, so this section focuses on exploring those potential reasons.
- Distribution of categorical variables: Explore the distribution of `stars`, `cool`, `funny`, `useful`.

Business DataFrame: Explore business demographics
- Observation count: Identify the total number of businesses.
- Column examination: Extract the column names and their respective types.
- Missing data analysis: Detect missing values in the dataset.
- Value counts for categorical data: Explore value counts for variables `state`, `city`, `stars`, `category` to better understand the demographics in our data.
  
User DataFrame: Explore user demographics
- Observation count: Identify the total number of users.
- Column examination: Extract the column names and their respective types.
- Missing data analysis: Detect missing values in the dataset. Specifically, how many users had no reviews.
- Summary statistics: Extract summary statistics regarding the review count.
- Correlations: Explore correlation between `star` and attributes `cool`, `funny`, `useful`.

### 3.2 Data Preprocessing
1. Selecting features: Select only relevant features for our model. Since we are trying to predict `stars` based on `text`, we only select these features. 
2. Addressing Missing Values: Eliminate rows containing missing values. 
3. Add features: We added feature 'review_length', which is the length of a user's text.
4. Tokenization: Split text into individual words. 
5. Stop Words Removal: Removal of common words from a text that have little to no semantic value in the context of the specific text processing task. 
6. Stemming:  Reduce words to their base or root form. The goal was to treat different forms of a word as the same word.
7. Sentiment Polarity Scores: Generate numerical values that indicate the sentiment or emotional tone of a piece of text. 
8. Term Frequency-Inverse Document Frequency (TF-IDF): Generate a numeric value that reflects the importance of a word in a document relative to a collection of documents. 
   
### 3.3 Model 1: Multinomial Logistic Regression
Multinomial Logistic Regression: Predicts the rating (multi-class) based on user text. This extends binary logistic regression to handle our multi-class target variable.
The logits for each category j are calculated relative to a reference category k:

<img width="487" alt="Screenshot 2024-06-06 at 10 19 00 PM" src="https://github.com/ssampatucsd/DSC232-Group_Project_Yelp/assets/168300575/c4be8900-4f0e-4e40-aa55-34195edeccc3">

The probabilities for each category are computed using the softmax function:

<img width="434" alt="Screenshot 2024-06-06 at 10 19 45 PM" src="https://github.com/ssampatucsd/DSC232-Group_Project_Yelp/assets/168300575/88f1960c-0949-42b5-8f75-b15f3803c7d8">

- Train Test Split: Split the dataset into training and testing sets to assess predictive performance and to avoid overfitting/underfitting.
- Baseline model:
- Feature Extension:
- Accuracy Metric: Establish an assessment of model accuracy and generalizability.
- Comparative Analysis:
  
### 3.4 Model 2: Support Vector Machine
Support Vector Machine: Predicts the rating (multi-class) based on user text. Since our target variable contains multiple classes, OneVsRest was also utilized here.

- Train Test Split: Split the dataset into training and testing sets to assess predictive performance and to avoid overfitting/underfitting.
- Baseline model:
- Feature Extension:
- Accuracy Metric: Establish an assessment of model accuracy and generalizability.
- Comparative Analysis:

## Results
### 4.1 Data Exploration
### 4.2 Data Preprocessing
1. Selecting features: Since we are trying to predict `stars` based on `text`, we only select these features. 
2. Addressing Missing Values: In most cases, missing values occured when a user left a star rating on a business, but failed to include a written review. These rows were irrelevant to our specific analysis.
3. Add features: We added feature 'review_length', which is the length of a user's text.
4. Tokenization: This is useful in simplifying text processing and enhancing machine learning models. It also improves text analysis and helps in understanding the syntax and semantics of the text.
5. Stop Words Removal: This reduces dimensionality, decreases noise and therefore improves model performance, enhances text analysis, and facilitates efficient storage.
6. Stemming: The goal of stemming was to simplify text processing and analysis. This reduces dimensionality, improves search and information retrieval, and enhances text analysis.
7. Sentiment Polarity Scores: This contributes to insights into customer sentiments and is helpful in predictive analytics.
8. Term Frequency-Inverse Document Frequency (TF-IDF): This is useful for feature selection, reducing noise, and information retrieval.
   
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

