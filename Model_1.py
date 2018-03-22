from sklearn.datasets import fetch_20newsgroups
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import numpy as np


categories_train = ['comp.graphics', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x']
train_data = fetch_20newsgroups(subset='train', categories=categories_train)['data']

categories_test = ['alt.atheism', 'soc.religion.christian', 'sci.med']
test_data = fetch_20newsgroups(subset='test', categories=categories_test)['data']

'''
Data is preprocessed to convert into sentences conprising of only alphabets
'''
def preprocess_data(text):
    text=text.replace('\n',' ')
    letters_only=re.sub('[^a-zA-Z]', ' ', text)
    lower_case=letters_only.lower().split()
    words = [w for w in lower_case if not w in stopwords.words('english')]
    return ' '.join(words)

def preprocess_corpus(data):
    processed_data=[]
    for i in range(len(data)):
        processed_data.append(preprocess_data(data[i]))
    return processed_data

data = preprocess_corpus(train_data)
preprocessed_test_data = preprocess_corpus(test_data)

##Tf-Idf vectorizer is trained on all of vocabulary available
train_data_vocab = fetch_20newsgroups(subset='train')['data']
preprocessed_train_data_vocab = preprocess_corpus(train_data_vocab)
vectorized_vocab = vectorizer.fit_transform(preprocessed_train_data_vocab)

vectorized_data = vectorizer.transform(data)
vectorized_test_data = vectorizer.transform(preprocessed_test_data)

rng = np.random.RandomState(42)
model = IsolationForest(max_samples=100, random_state=rng)
model.fit(vectorized_data)

prediction = model.predict(vectorized_test_data)
print(prediction)
