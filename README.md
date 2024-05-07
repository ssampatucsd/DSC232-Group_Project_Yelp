# DSC232-Group_Project_Yelp

### Create Environment and Install Dependencies:
- The "requirements.txt" file can be used to install all the necessary packages for all the notebooks in this course.
- Navigate to the folder location where you have downloaded this repository. Then, run "pip install -r requirements.txt". This should create your environment with all necessary packages.

### Link for data download: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

### Steps Taken for Pre-Processing:
1. Understand the data
   - Read and load the dataset.
   - Explore the dataset's features, datatype, and schema.
2. Data cleaning
   - Check for missing values (null values), duplicates, and outliers.
3. Feature engineering
   - Create new features and/or apply principal component analysis (PCA) to reduce dimensionality.
4. Data transformation
   - Clean text by converting to lowercase to ensure uniformity and removing punctuation and special     
     characters.
   - Tokenize words and sentences.
   - Remove "stop words".
   - Reduce words to their base or root form by applyin gStemming and lemmatization technique.
   - Transform text into numerical format using bag-of-word, term frequency-inverse document frequency         (TF-IDF), and word embeddings techniques.


