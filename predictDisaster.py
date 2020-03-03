# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:28:38 2020

@author: Moumita Kamal
"""

import pandas as pd
train_dataset = pd.read_csv('Datasets/train.csv')
test_dataset = pd.read_csv('Datasets/test.csv')

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
count_vectorizer = feature_extraction.text.CountVectorizer()

X_train = count_vectorizer.fit_transform(train_dataset['text'])
X_test = count_vectorizer.transform(test_dataset['text'])

classifier = linear_model.RidgeClassifier()
classifier.fit(X_train, train_dataset['target'])

pred = classifier.predict(X_test)

sub = pd.read_csv('sample_submission.csv')
sub['target'] = classifier.predict(X_test)
sub.to_csv('submission.csv', index = False)