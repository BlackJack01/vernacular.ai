from sklearn.datasets import fetch_20newsgroups
categories_train = ['comp.graphics', 'comp.sys.ibm.pc.hardware']
train_data = fetch_20newsgroups(subset='train', categories=categories_train)['data']

import tensorflow as tf
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

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
    
preprocessed_data = preprocess_corpus(train_data)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

vectorized_data = vectorizer.fit_transform(preprocessed_data)
vectorized_data=np.array(vectorized_data.todense())

n_input_output = vectorized_data.shape[1]
n_hidden = 100

#Here an autoencoder neural network architecture is used to detect outliers

X = tf.placeholder(shape = [None, n_input_output], dtype = tf.float64)
Y = tf.placeholder(shape = [None, n_input_output], dtype = tf.float64)

W = {
    'h':tf.Variable(tf.random_normal(shape = [n_input_output, n_hidden], dtype = tf.float64), dtype = tf.float64),
    'out':tf.Variable(tf.random_normal(shape = [n_hidden, n_input_output], dtype = tf.float64), dtype = tf.float64)
}

b = {
    'h':tf.Variable(tf.random_normal(shape = [n_hidden], dtype = tf.float64), dtype = tf.float64),
    'out':tf.Variable(tf.random_normal(shape = [n_input_output], dtype = tf.float64), dtype = tf.float64)
}

def model(x):
    h = tf.add(tf.matmul(x, W['h']), b['h'])
    out = tf.add(tf.matmul(h, W['out']), b['out'])
    return out
    
prediction = model(X)
loss = tf.reduce_mean(tf.square(prediction - Y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

batch_size = 24
n_iterations = (int)(vectorized_data.shape[0]/batch_size)+1

for epoch in range(50):
    print("Epoch ",epoch)
    for i in range(n_iterations):
        if i < n_iterations-1:
            batch_x = vectorized_data[i*batch_size: (i+1)*batch_size]
        else:
            batch_x = vectorized_data[i*batch_size: vectorized_data.shape[0]]
        sess.run(train_step, feed_dict = { X:batch_x,Y: batch_x})
        if (i+1)%10 == 0:
            print("Iteration ",(i+1),"    ",sess.run(loss, feed_dict = {X: batch_x, Y: batch_x}))
            

categories_test = ['alt.atheism']
test_data = fetch_20newsgroups(subset='test', categories=categories_test)['data']

preprocessed_test_data = preprocess_corpus(test_data)
vectorized_test_data = vectorizer.transform(preprocessed_test_data)
vectorized_test_data = np.array(vectorized_test_data.todense())

loss1 = tf.reduce_mean(tf.square(prediction - Y), axis = 0)
pred = sess.run(loss1, feed_dict = {X: vectorized_test_data,Y: vectorized_test_data})
pred1 = sess.run(loss1, feed_dict = {X: vectorized_data,Y: vectorized_data})

import matplotlib.pyplot as plt

plt.figure(figsize = (4,8))
plt.hist(pred)
plt.show()

plt.figure(figsize = (4,8))
plt.hist(pred1)
plt.show()

## Looking at the plots here I decide 90 as the threshold limit
result =[]
for i in range(len(pred)):
    if pred[i]>=90:
        result.append(-1)
    else:
        result.append(1)

print(result)
