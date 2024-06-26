# DSC232: Yelp Reviews

## Overview
This repository contains the final project for DSC232, featuring a thorough analysis and the creation of predictive models aimed at predicting 'stars' from user review text. Within, you'll find datasets, analytical scripts, model training algorithms, and results designed to offer companies deeper insights into their customers' experiences. Comprehensive documentation is available to help you navigate each aspect of the project.

Below is the written report for this project. Here is the link to our Jupyter Notebook.

## 1. Introduction
Yelp Inc. is a widely-used online platform in the United States where users can search for and review local businesses. Founded in 2004, it acts as a directory for various businesses, including restaurants, retail stores, services, and more. Yelp provides information about the business such as business hours, contact information, photos, and user-generated reviews. Businesses can claim their Yelp profiles to interact with customers, post updates, and respond to reviews.

Yelp has several positive impacts on businesses, such as increasing visibility, establishing credibility and trust, engaging with customers, gaining market insights, and providing advertising opportunities. For instance, Yelp can help businesses get discovered by new customers, driving foot traffic and online inquiries. Yelp's convenient category searches, such as for restaurants, doctors, or plumbers, allows businesses to become more accessible to consumers, and thus, increases not only consumer reach but consumer acquisition. However, there are also negative impacts, such as the potential for negative reviews to harm a business's reputation and revenue.

This project aims to predict a business's rating out of five stars based on user reviews. We use machine learning techniques and natural language processing to address the challenge of text classification. In addition, we investigate whether there is a correlation between language and rating. This predictive goal can provide businesses with insights into what customer experiences are associated with positive and negative reviews. The project is interesting because of its potential positive impact on both businesses and customers. It can enhance the success of businesses and improve customer decision-making, as customers often rely on reviews for their purchasing decisions.

The dataset used in this project is notable for its size and comprehensive information. It is a subset of Yelp's data, containing information about businesses, reviews, and users across eight metropolitan areas in the USA and Canada. The data, sourced directly from Yelp.com, includes 7 million customer reviews and features 150,000 businesses. The credibility, size, and richness of the data were the main reasons for choosing this dataset.

## 2. Figures
Green line = Error on the test dataset

Red line = Error on the train dataset

![image](https://github.com/ssampatucsd/DSC232-Group_Project_Yelp/assets/150002146/01109385-edc8-4dce-aa09-98031f8742b6)
Figure 1.

![image](https://github.com/ssampatucsd/DSC232-Group_Project_Yelp/assets/150002146/5ca11384-229d-4167-ba0b-d11c10edd63e)
Figure 2.


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
- Baseline Model/Model Complexities: Begin with a baseline model/simpler model for comparative analysis and increase complexity to assess how this influences the model's training and testing error.
- Accuracy Metric: Establish an assessment of model accuracy and generalizability.
- Comparative Analysis: Compare training and testing errors for differing model complexities.
  
### 3.4 Model 2: Support Vector Machine with OneVsRest
Support Vector Machine: Predicts the rating (multi-class) based on user text. Since our target variable contains multiple classes, OneVsRest was also utilized here to account for our multi-class problem.

- Train Test Split: Split the dataset into training and testing sets to assess predictive performance and to avoid overfitting/underfitting.
- Baseline Model/Model Complexities: Begin with a baseline model/simpler model for comparative analysis and increase complexity to assess how this influences the model's training and testing error.
- Accuracy Metric: Establish an assessment of model accuracy and generalizability.
- Comparative Analysis: Compare training and testing errors for differing model complexities.

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
  3. Category: Top 5 categories are: restaurants, food, shopping, home services, beauty and spas. Resturaunts had an exceedingly high amount of reviews.
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
A multinomial logistic regression was performed on the columns `combined features` and `stars`. It utilizes the combined features, which was composed of `review length`, `sentiment polarity`, and `TD-IDF` done in previous steps to predict the star ratings on businesses. A grid search was done to get the best hyperparmeter values for regParam and elasticNetParam for the model and also k-fold cross validation (k = 5) was performed to further optimize the model. The best regParam was 0.01, which matched the ideal complexity in Figure 1., and the best elasticNetParam was 0.3. The best CV test accuracy of this model is 61.94%.

The underfitting/overfitting graph is displayed below:
![image](https://github.com/ssampatucsd/DSC232-Group_Project_Yelp/assets/150002146/01109385-edc8-4dce-aa09-98031f8742b6)

### 4.4 Model 2: Support Vector Machine with OneVsRest
A support vector machine was performed using the same `combined features` column as input and `stars` as ouput. The combined features again composed of `review length`, `sentiment polarity`, and `TF-IDF`. After running multiple values of hyperparameters and visualizing the results with a fitting graph, the best value of regParam was 0.01. This value resulted in fairly consistent training and test errors and minimized both the training and test error, as shown in Figure 2. This shows that the model is neither underfitting nor overfitting the data.

The underfitting/overfitting graph is displayed below:

![image](https://github.com/ssampatucsd/DSC232-Group_Project_Yelp/assets/150002146/5ca11384-229d-4167-ba0b-d11c10edd63e)

### 4.5 Compare Model Performances
Both models were utilized to predict the star ratings of businesses. The SVM model demonstrated superior accuracy compared to the logistic regression model because it is capable of handling more complex and capturing the non-linear relationships in the data. Additionally, SVM's approach of maximizing the margin between classes makes it robust to overfitting, which likely contributed to its higher accuracy. Logistic regression finds linear relationships between the features and the star ratings, which causes a limitation when the actual relationships in the data are more complicated and not capturing all the nuances, which led to the lower accuracy. 

## 5. Discussion
### Data Exploration: Exploratory Data Analysis
During the Exploratory Data Analysis (EDA) phase, we began by examining the number of observations in the Review DataFrame (6,990,280 unique reviews), Business DataFrame (150,346 unique businesses), and User DataFrame (1,987,897 unique users). Given the substantial sample size of reviews, we can be confident that our findings are generalizable. Further, the observation counts of the Business DataFrame and the User DataFrame show diversity in our modeling efforts. That is, the reviews come from a large number of businesses and many different users.

Considering the Review DataFrame, we explored missing values. Missing values can often actually contain important information because they are not just missing by chance in many cases. In this specific case, there were several observations with missing values in the text field, which indicated that a user left a rating on a business but did not provide a text response. These values were dropped because they were irrelevant given our problem statement, so we can confidently state that simply dropping missing values would not impact the integrity of our model.

Although the Business DataFrame and User DataFrame were not used in modeling, it was still important that we explored these dataframes since all data frames in this API are related. In analyzing the demographics of the Business DataFrame, we find that these reviews contain information on a variety of different businesses. This helps adding diversity and generalizability to the model since the reviews capture various types of businesses in different areas of the United States.

The User DataFrame contains the source of the reviews, or the actual user that wrote the review. There were only about ten users with no reviews, meaning that most users are active or were active users when they left the review. This allows us to ensure that reviews were truly written by a human and are not "fake" reviews.

### Data Preprocessing
During the Data Preprocessing phase of the study, various steps were taken to refine the dataset for subsequent modeling. Our problem consisted of a text classification problem, where we had only a `text` column (the actual review) to predict `stars` (the rating). Data preprocessing was essential to achieve this. Specifically, we performed tokenization, stop words removal, stemming, extracting sentiment polarity scores, and term frequency-inverse document frequency (TF-IDF). Tokenization was crucial for removing stop words, which allowed our models to focus on words that carry significant information about the sentiment and content of the review. Although stemming is a common preprocessing step, we did not utilize it in our final models.  We found that stemming did not significantly affect our results, and it was unnecessary to include it. It is good practice to build the simplest model possible withtout sacrificing accuracy. 

Further, we extracted sentiment polarity scores and TF-IDF. These were arguably the most important data preprocessing methods for our project. Sentiment polarity scores provided an additional feature that captured the emotional tone of the review, which is often strongly correlated with the rating. This likely improved our predictive power, as sentiment is a key indicator of user satisfaction or dissatisfaction. Similarly, TF-IDF was crucial for modeling. It assigns higher weights to unique words in a review, improving the model's ability to differentiate between reviews. This helps identify key terms that significantly impact the classification, leading to better model performance.

These preprocessing steps were essential for extracting meaningful information from text data, reducing noise, and improving the performance of our models. Each step played a cruicial role in ensuring that the data was both informative and manageable.

### Model 1 and 2: Logistic Regression and Support Vector Machine 
Initially, we thoroughly explored the dataset's structure, schema, and identify relevant features for model training. We selected the features that have influence on the the prediction of star ratings. Then, we split the data into training and testing set to evaluate the model's performance, and we trained and optimized the model through hyperparameter tuning using grid search and cross-validation. Lastly, we generated overfitting and underfitting graphs to help us better understand and select the ideal complexity for each model.

Multinomial Logistic Regression was chosen for our predictive modeling because it extends binary logistic regression to handle multiple classes, which is essential for our multi-class target variable, the star ratings is from 1 to 5 stars. Logistic regression provides a probabilistic framework that is interpretable and can be valuable for understanding the certainty of predictions. Support vector machine (SVMs) was selected for our predictive modeling because of its robustness in handling high-dimensional data and effectiveness in mulit-classification tasks. SVMs can create complex decision boundaries and manage large feature space deriving from the review text data. 

Our logistic regression model and support vector machine performed 61.94% and 58.38% accuracy in prediction, respectively, which indicate a reasonable performance, given the complexity in the review text. While the models provided valueable insights, there are two shortcomings. First, our models may be too simple that it may not capture complex relationship in the data. Second, other relevant features may be excluded during the feature engineering step. These shortcomings could impact the accuracy of the prediction. Our work is rarely a perfect solution, and future research is indeed needed. We could explore in a more advanced model, such as recurrent neural networks (RNNs) or a better technique for natural language prcessing (NLP). The quality of the data can be further examined to handle duplicates reviews and biases. Overall, our model has rooms for improvement and needs to be updated as new data becomes available or new methods are developed.

Both logistic regression and support vector machine models are effective for predicting Yelp star ratings. The SVM's ability to handle non-linear relationships within the data and logistic regression's interpretability and efficiency allowed to explore the data in different ways. Future work could possibility include other kernels for SVM and other hyperparameters for improving the prediction accuracy.

### General Discussion
Our project has highlighted the power of text classification models to extract consumer insights. This section outlines the successes achieved, the limitations encountered, and our future plans for extending this work.

#### Success
Our extensive exploratory data preprocessing on the review text allowed us to effectively refine the dataset. These reprocessing steps significantly improved our model performance by reducing noise and extracting meaningful features, such as TF-IDF and sentiment polarity scores. Also, we were able to successfully apply multinomial logistic regression model to demonstrate its effectiveness in handling multi-class classification tasks.

#### Limitations
One of the limitations we ecountered was the variability in review length and the duplicate reviews, which could introduce bias into the model. Although we addressed the mising values and performed data cleaning, there is always the potential for unseen biases in such large datasets. These limitations could explain the resuts of low accuracy in the predictions from our two models.

## 6. Conclusion
This project demonstrated the value of machine learning techniques, specifically in logistic regression and support vector machine, in deriving actionable insights from the user's review texts. By predicting star ratings based on review texts, businesses can better understand customer sentiments and improve their services to increase customer satisfaction. While our models have shown promising results, there is room for improvement. We could explore more complex text processing techniques, such as deep learning models, to further enhance predictive accuracy. In addition, we could incorporate other features from the Yelp dataset, such as business attributes, category, longitude, and latitude, to provide a more holistic view and to potentially improve the model performance. Despite the limitation of mitigating potential biases in the dataset, the project underscores the potential of applying machine learning to extract valuable insights from textual data, enabling business to gain a deeper understanding of their customers' need and the companys' services that need further improvement.


## 7. Collaboation
Lian Martin, Christie Ma, Emily Zhuang, Johnson Tso, and Sanjay Sampat worked well together and collaborated as a team throughout this project.

Lian Martin participated by creating visualizations for categorical variables and contributed to the Logistic Regression and SVM models. She also took the initiative and contributed to an extensive amount of the final write up (introduction, methods, results, and discussion).
Christie contributed through preparing the initial dataset and exploring the data in different ways. She helped to improve the model and finding parameters, although not used ultimately. She also wrote the comparison between models and parts of the conclusion.
Emily Zhuang contributed by sharing potential project datasets, creating graphical visualizations for the initial data exploration, and helping her team troubleshoot Expanse issues. She helped her team successfully create the fitting graph and contributed to the report.
Johnson Tso participated in the discussions throughout the project. He contributed by exploring the dataset, preprocessing the data in feature engineerings to select the important features for training the model, and writing up parts of introduction, model results, and general discussion.
Sanjay Sampat helped with he intial eda and project ideas. He assisted on the feature enginering in chossing the relevant features and removing the unnecessary ones. He then helped develop the Logistic regression model through gridsearch cv and k fold validation. Finally he created the overfitting and underfitting graphs for both models. 
