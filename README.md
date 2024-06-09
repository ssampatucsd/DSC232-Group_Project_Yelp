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
- Schema and column examination: Extract the column names and their respective types.
- Missing data analysis: Detect missing values in the dataset. Missing data should not just be ignored. Often, there are underlying reasons for why this is occuring, so this section focuses on exploring those potential reasons.
- Duplicate values: Identify observations with duplicate values.
- Summary statistics: Extract summary statistics for numerical features.
- Distribution of categorical variables: Explore the distribution of `stars`, `cool`, `funny`, `useful`.

Business DataFrame: Explore business demographics
- Observation count: Identify the total number of businesses.
- Schema and column examination: Extract the column names and their respective types.
- Missing data analysis: Detect missing values in the dataset.
- Value counts for categorical data: Explore value counts for variables `state`, `city`, `stars`, `category` to better understand the demographics in our data.
  
User DataFrame: Explore user demographics
- Observation count: Identify the total number of users.
- Schema and column examination: Extract the column names and their respective types.
- Missing data analysis: Detect missing values in the dataset. Specifically, how many users had no reviews.
- Summary statistics: Extract summary statistics regarding the review count.
- Correlations: Explore correlation between `star` and attributes `cool`, `funny`, `useful`.
- Distribution of number of reviews: Extract the number of reviews by user.

### 3.2 Data Preprocessing
1. Selecting features: Select only relevant features for our model. Since we are trying to predict `stars` based on `text`, we only select these features. 
2. Addressing Missing Values: Eliminate rows containing missing values. 
3. Add features: We added feature `review_length`, which is the length of a user's text.
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
  
### 3.4 Model 2: Support Vector Machine with OneVsRest
Support Vector Machine: Predicts the rating (multi-class) based on user text. Since our target variable contains multiple classes, OneVsRest was also utilized here.

- Train Test Split: Split the dataset into training and testing sets to assess predictive performance and to avoid overfitting/underfitting.
- Baseline model:
- Feature Extension:
- Accuracy Metric: Establish an assessment of model accuracy and generalizability.
- Comparative Analysis:

## Results
### 4.1 Data Exploration
Review DataFrame: 
- Observation count: Total number of reviews = 6,990,280.
- Schema and column examination: To avoid repetition of information, please reference section 3.1.
- Missing data analysis: Missing values in `text` column.
- Duplicate values: Multiple duplicate values in `text`, specifically containing "DO NOT PARK HERE!", "I had a terrible experience", "Great place to be."
- Summary statistics: Average review length is 567.76, however, this variable has a large standard deviation of 527.25. Min review length = 1. Max review length = 5000.
- Distribution of categorical variables:
    1. Stars: 5 had the highest value counts by far, followed by 4, then 1. This means that people are far more likely to leave positive reviews for a positive experience over a negative review for a negative experience.
  2. Cool, Funny, Useful: `useful` had the highest value counts compared to `cool` and `funny`, which intuitively makes sense given the nature of the platform.

Business DataFrame: 
- Observation count: Total number of businesses = 150,346.
- Schema and column examination: To avoid repetition of information, please reference section 3.1.
- Value counts for categorical data: Explore value counts for variables `state`, `city`, `stars`, `category` to better understand the demographics in our data.
  1. State: Top 5 states are PA, FL, TN, IN, MO.
  2. Stars: 4.0 had the highest count, followed by 4.5. Distribution resembles a bell curve that is skewed left.
  3. Category: Top 5 categories are: resturaunts, food, shopping, home services, beauty and spas. Resturaunts had an exceedingly high amount of reviews.
  4. City: Top 5 cities are: Philadelphia, Tuscon, Tampa, Indianapolis, Nashville. This is consistent with the top 5 states. Further, there were a handful of cities with less than 3 businesses.

User DataFrame:
- Observation count: Total number of users = 1,987,897.
- Schema and column examination: To avoid repetition of information, please reference section 3.1.
- Missing data analysis: No missing values.
- Summary statistics: Mean review count was 23.4, but has a high standard devation of 82.6. Min = 0. Max = 17473.
- Correlations: No correlations present.
- Distribution of number of reviews: This part detected a handful of outliers. Specifically, there were a handful of users with more than 5,000 reviews. 

From the data exploration process, our team concluded that our project requires use of the Review DataFrame only. The rest of this section only focuses on the Review DataFrame.
  
### 4.2 Data Preprocessing
1. Selecting features: Since we are trying to predict `stars` based on `text`, we only select these feature. All other features simply add unnecessary noise to our model.
2. Addressing Missing Values: Missing values occured when a user left a star rating on a business, but failed to include a written review. 
3. Add features: We added feature `review_length`, which is the length of a user's text.
4. Tokenization: This was found to be useful in simplifying text processing and enhancing machine learning models. It also improves text analysis and helps in understanding the syntax and semantics of the text.
5. Stop Words Removal: This reduces dimensionality, decreases noise and therefore improves model performance, enhances text analysis, and facilitates efficient storage, which was a challenge given how large the dataset was.
6. Stemming: The goal of stemming was to simplify text processing and analysis. This reduces dimensionality, improves search and information retrieval, and enhances text analysis.
7. Sentiment Polarity Scores: This contributes to insights into customer sentiments and is helpful in predictive analytics.
8. Term Frequency-Inverse Document Frequency (TF-IDF): This is useful for feature selection, reducing noise, and information retrieval. 
   
### 4.3 Model 1: Multinomial Logistic Regression 
A multinomial logistic regression was performed on the columns combined features and stars. It utilizes the combined features which was composed of review length and sentiment polarity done in previous steps to predict the star ratings on businesses. A grid search was done to get the best hyperparmeters for the model and also cross validation was performed to further optimized the model. The best regParam was 0.01 and best elasticNetParam was 0.3. The accuracy of this model is 61.94%.

### 4.4 Model 2: Support Vector Machine with OneVsRest

### 4.5 Compare Model Performances

## 5. Discussion
### Data Exploration: Exploratory Data Analysis
During the Exploratory Data Analysis (EDA) phase, we began by examining the number of observations in the Review DataFrame (6,990,280 unique reviews), Business DataFrame (150,346 unique businesses), and User DataFrame (1,987,897 unique users). Given the substantial sample size of reviews, we can be confident that our findings are generalizable. Further, the observation counts of the Business DataFrame and the User DataFrame show diversity in our modelling efforts. That is, the reviews come from a large number of businesses and many different users. Considering the Review DataFrame, we then explored missing values. Often, missing values can actually contain important information because in many cases, they are not just missing by chance. In this case, there were several observations with missing values in the `text` field, which indicated that a user left a rating on a business, but did not provide a text response. These values were dropped because they were irrelevant given our problem statement, so we can be confident that simply dropping missing values would not impact the integrity of our model. Although the Business DataFrame and User DataFrame were not used in modelling, it was still important that we explored these dataframes since all data frames in this API are related. In analyzing the demographics of the Business DataFrame, we can be confident that the reviews contain information on a variety of different businesses. This adds diversity and generalizability to the model since the reviews capture many different types of businesses in different areas of the United States. The User DataFrame contains the source of the reviews, or the actual user that wrote the review. There were only about ten users with no reviews, meaning that most users are active or were active users when they left the review. This is important in checking that the reviews were truly writtten by a human and are not "fake" reviews.

### Data Preprocessing
During the Data Preprocessing phase of the study, various steps were taken to refine the dataset for subsequent modeling. Our problem consisted of a text classification problem, where we had only a `text` column (the actual review) to predict `stars` (the rating). Data preprocessing was essential to achieve this. Specifically, we performed tokenization, stop words removal, stemming, extracting sentiment polarity scores, and term frequency-inverse document frequency (TF-IDF). Tokenization was crucial for removing stop words, which allowed our models to focus on words that carry significant information about the sentiment and content of the review. Although stemming is a common preprocessing step, we did not utilize it in our final models.  We found that stemming did not significantly affect our results, and it was unnecessary to include it. It is good practice to build the simplest model possible withtout sacrificing accuracy. 

Further, we extracted sentiment polarity scores and TF-IDF. These were arguably the most important data preprocessing methods for our project. Sentiment polarity scores provided an additional feature that captured the emotional tone of the review, which is often strongly correlated with the rating. This likely improved our predictive power, as sentiment is a key indicator of user satisfaction or dissatisfaction. Similarly, TF-IDF was crucial for modeling. It assigns higher weights to unique words in a review, improving the model's ability to differentiate between reviews. This helps identify key terms that significantly impact the classification, leading to better model performance.

These preprocessing steps were essential for extracting meaningful information from text data, reducing noise, and improving the performance of our models. Each step played a cruicial role in ensuring that the data was both informative and manageable.

### Model 1: Logistic Regression

### Model 2: Support Vector Machine

### General Discussion
Our project has highlighted the power of text classification models to extract consumer insights. This sectionn outlines the successes achieved, the limitations encountered, and our future plans for extending this work.

#### Success

#### Limitations

## 6. Conclusion

## 7. Collaboation
Lian Martin, Christie Ma, Emily Zhuang, Johnson Tso, and Sanjay Sampat worked well together and collaborated as a team throughout this project.

Lian Martin participated by creating visualizations for categorical variables and contributed to the Logistic Regression and SVM models. She also took the initiative and contributed to an extensive amount of the final write up (introduction, methods, results, and discussion).
