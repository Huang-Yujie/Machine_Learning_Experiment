from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np

trainData = pd.read_csv('trainCut.csv')
testData = pd.read_csv('testCut.csv')

x_train = trainData['textCut'].values.astype('U')
y_train = trainData['label'].values.astype('U')
x_test = testData['textCut'].values.astype('U')

vec = CountVectorizer()
transformer = TfidfTransformer()
x_train = transformer.fit_transform(vec.fit_transform(x_train))
x_test = transformer.transform(vec.transform(x_test))

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)

np.savetxt('output.txt', y_predict, fmt='%s')