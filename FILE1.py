#importing all required libraries
import pandas as pd
import numpy as np
import nltk
import re
data = pd.read_csv('stress.txt')

#checking the head
print(data.head())
#checking wether or not the data contains null values and either dropping or replacing them
print(data.isnull().sum())

#preparing the text column of this dataset to clean the text column with stopwords, links, special symbols and language error
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["text"] = data["text"].apply(clean)

#The label column in this dataset contains labels as 0 and 1. 0 means no stress, and 1 means stress. I will use Stress and No stress labels instead of 1 and 0.
#Preparing the text column accordingly and selecting the text and label columns for the process of building and training the machine learning model:
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]
print(data.head())

# Splitting the dataset into training and test sets:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

#As this task is based on the problem of binary classification, I will be using the Bernoulli Naive Bayes algorithm, which is one of the best algorithms for binary classification problems. So let’s train the stress detection model:
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(xtrain, ytrain)

#Testing the model's performance on some random statements based on mental health
#presence of stress
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output) 
#absence of stress
user = input("enter a text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
