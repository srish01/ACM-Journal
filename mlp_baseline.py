import os 
import sys
import numpy as np 
import pandas as pd 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import models
# from tensorflow.keras import optimizers

from utils import jaccard, kuncheva, total_consistency
from art.attacks.evasion import FastGradientMethod, DeepFool
from art.attacks.poisoning import PoisoningAttackSVM
from art.estimators.classification import SklearnClassifier
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.estimators.classification import KerasClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from skfeature.function.information_theoretical_based import MIM, MRMR, DISR
from skfeature.function.similarity_based.fisher_score import fisher_score, feature_ranking
from skfeature.function.similarity_based.reliefF import reliefF

if tf.__version__[0] != '2':
    raise ImportError('This notebook requires TensorFlow v2.')

SEED = 1
# classifier used to generate the attack [svc, rfc]
CLFR = 'mlp'
# number of cross fold runs to perform 
TRIALS = 5
# type of attack to generate [fgsm, deepfool, svc]
ATTACK_TYPE = 'fgsm'
# svc attack parameters if the algorithm is used in ATTACK_TYPE
SVC_ATTACK_PARAM ={'T':25, 'step':0.1, 'eps':.5}
# adversarial sample ratio
RATIO = 0.2
# number of features to select 
NFEATURES = 10 
# number of classes 
NUM_CLASSES = 2
# MLP parameters
MLP_epochs = 20
MLP_batch_size = 50
MLP_learning_rate = 0.01

# path the the data folder [must be downloaded from Gitlab]
BASE_PATH = 'data/'
# binary classification dataset that we are going to use from the uci data sets
DATA_SETZ = ['breast-cancer-wisc-diag',
             'breast-cancer-wisc-prog',
             'conn-bench-sonar-mines-rocks',
             'cylinder-bands',
             'ionosphere',
             'molec-biol-promoter',
             'musk-1',
             'oocytes_merluccius_nucleus_4d',
             'oocytes_trisopterus_nucleus_2f',
             'parkinsons',
             'spectf_train'
             ]

def create_network(Xtr, NUM_CLASSES):   
    mlp = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape = Xtr.shape[1]),        
        tf.keras.layers.Dense(20, activation = 'relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation = 'sigmoid')
    ])
    mlp.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    mlp.summary()
    return(mlp)

#optimizer = rmsprop

param = ['error', 'accuracy']
baseline = np.zeros((len(DATA_SETZ), len(param)))

for i in range(len(DATA_SETZ)): 
    # load the binary classification dataset then 
    data_set = DATA_SETZ[i]
    
    df = pd.read_csv(''.join([BASE_PATH, data_set, '.csv']), header=None)
    X, y = df.values[:,:-1], df.values[:,-1]

    n, nf = X.shape
    i_perm = np.random.permutation(n)
    X, y = X[i_perm], y[i_perm]

    # we need to set up the k-fold evaluator 
    kf = KFold(n_splits=TRIALS)
    kf.get_n_splits(X)
    
    print(''.join([data_set, ' (', str(i+1), ' of ', str(len(DATA_SETZ)), ')']))

    k = 0
    mean_loss = 0
    mean_accuracy = 0
      
    for train_index, test_index in kf.split(X):
        # split the original data into training / testing datasets. we are not going to 
        # use the testing data since we are not going to learn a classifier. 
        Xtr, ytr, Xte, yte = X[train_index,:], y[train_index], X[test_index,:], y[test_index]

        # convert the labels to a one-hot encoding for the attack model then fit the classifier 
        ytr_ohe = tf.keras.utils.to_categorical(ytr, NUM_CLASSES)
        yte_ohe = tf.keras.utils.to_categorical(yte, NUM_CLASSES)
        model = create_network(Xtr, NUM_CLASSES)
        
        model.fit(Xtr, ytr_ohe, epochs = MLP_epochs, batch_size = MLP_batch_size, verbose = 0)
        
        # Test the model after training
        test_results = model.evaluate(Xte, yte_ohe, verbose=0)
        k+=1
        mean_loss += test_results[0]
        mean_accuracy += test_results[1]
        
        print("Iteration: ", k)
        print('Loss:', test_results[0], "Accuracy:", test_results[1])
    
    #print("Mean Loss: ", mean_loss/TRIALS)
    #print("Mean_accuracy: ", mean_accuracy/TRIALS)
    
    baseline[i, 0] = mean_loss/TRIALS
    baseline[i, 1] = mean_accuracy/TRIALS
    print(baseline)
    
    np.savez('outputs/MLP/baseline_scores.npz')
