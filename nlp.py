import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import re

dataset = pd.read_csv('datasets/train.csv')

processed_tweet = []

dataset['tweet'][0]

for i  in range(31962):
    tweet = re.sub('@[\w]*',' ',dataset['tweet'][i])
    tweet = re.sub('[^a-zA-Z#]',' ',tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(token) for token in tweet if not token in stopwords.words('english')]
    tweet = ' '.join(tweet)
    processed_tweet.append(tweet)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(processed_tweet)
X = X.toarray()
y = dataset['label'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)

nb.score(X_train,y_train)
y_pred = nb.predict()                  