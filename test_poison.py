#!/usr/bin/env python 
from art.attacks.poisoning import PoisoningAttackSVM
from art.estimators.classification import SklearnClassifier
from sklearn.svm import SVC
from art.estimators.classification.scikitlearn import ScikitlearnSVC

import pandas as pd
import numpy as np 
import tensorflow as tf 

df = pd.read_csv(''.join(['data/', 'parkinsons', '.csv']), header=None)
# X is a number array that is n x d and y is d-dimensional vector of [0,1]
X, y = df.values[:,:-1], df.values[:,-1]

# split the data up:: not really needed
i_perm = np.random.permutation(len(y))
X, y = X[i_perm], y[i_perm]
n = len(y)
m = int(np.floor(n/2))

X1, y1, X2, y2 = X[:m], y[:m], X[m:], y[m:]
X2, y2 = X2[:10], y2[:10]
m = len(y2)


# convert to one-hot. so, [[1,0],[0,1],[1,0],...]
y1_ohe = tf.keras.utils.to_categorical(y1, 2)
y2_ohe = tf.keras.utils.to_categorical(y2, 2)

x_train = X1
y_train = y1_ohe
x_test = X2
y_test = y2_ohe


poison = SklearnClassifier(model=SVC(kernel='rbf'), clip_values=(-5,5))
poison.fit(x_train, y_train)

attack = PoisoningAttackSVM(poison, 0.01, 1.0, x_train, y_train, x_test, y_test, 25)
attack_y = np.ones((15,2)) - y_train[:15]
attack_point, _ = attack.poison(np.array(x_train[:15]), y=np.array(attack_y))

print(attack_point)
