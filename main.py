#!/usr/bin/env python 

# Copyright 2021 Gregory Ditzler 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import os 
import sys
import numpy as np 
import pandas as pd 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from utils import jaccard, kuncheva, total_consistency
from art.attacks.evasion import FastGradientMethod, DeepFool, SaliencyMapMethod
from art.attacks.poisoning import PoisoningAttackSVM, PoisoningAttackCleanLabelBackdoor, PoisoningAttackBackdoor
from art.attacks.poisoning.adversarial_embedding_attack import PoisoningAttackAdversarialEmbedding
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import SklearnClassifier
#from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.estimators.classification import KerasClassifier
from tests.utils import get_tabular_classifier_tf

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from skfeature.function.information_theoretical_based import MIM, MRMR, DISR
from skfeature.function.similarity_based.fisher_score import fisher_score, feature_ranking
from skfeature.function.similarity_based.reliefF import reliefF

if tf.__version__[0] != '2':
    raise ImportError('This notebook requires TensorFlow v2.')

# PROGRAM CONSTANTS 
# seed for reproducibility   
SEED = 1
# classifier used to generate the attack [svc, rfc, mlp, nn, lr]
CLFR = 'nn'
# number of cross fold runs to perform 
TRIALS = 5 
# type of attack to generate [fgsm, deepfool, svc, jsma, cleanlabel_single, cleanlabel_pattern, embedding]
ATTACK_TYPE = 'deepfool'
# svc attack parameters if the algorithm is used in ATTACK_TYPE
SVC_ATTACK_PARAM ={'T':25, 'step':0.1, 'eps':.5}
# adversarial sample ratio
RATIO = 0.2
# number of features to select 
NFEATURES = 10
# number of classes 
NUM_CLASSES = 2
# MLP parameters
epochs = 20
batch_size = 50
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
# location of the file output 
OUTPUT_FILE = ''.join(['outputs/Classifier_NN/deepfool/experiment_', ATTACK_TYPE, '_', CLFR, '.npz'])

# BASIC MLP structure
def create_network(Xtr, NUM_CLASSES):   
    mlp = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape = Xtr.shape[1]),        
        tf.keras.layers.Dense(20, activation = 'relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation = 'sigmoid')
    ])
    mlp.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    mlp.summary()
    return(mlp)

# 5-nn
def create_nn(Xtr, NUM_CLASSES):
    nn = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape = Xtr.shape[1]), 
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(0.2, input_shape=(128,)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(20, activation = 'relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation = 'sigmoid')
    ])
    nn.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    nn.summary()
    return(nn)

# set the random seed for reproducibility 
np.random.seed(SEED)

# initialize all of the variables where we are going to save our data. 
mim_jaccard_dist_clean = np.zeros((len(DATA_SETZ,)))
mrmr_jaccard_dist_clean = np.zeros((len(DATA_SETZ,)))
relief_jaccard_dist_clean = np.zeros((len(DATA_SETZ,)))
fisher_jaccard_dist_clean = np.zeros((len(DATA_SETZ,)))
disr_jaccard_dist_clean = np.zeros((len(DATA_SETZ,)))

mim_jaccard_dist_adv = np.zeros((len(DATA_SETZ,)))
mrmr_jaccard_dist_adv = np.zeros((len(DATA_SETZ,)))
relief_jaccard_dist_adv = np.zeros((len(DATA_SETZ,)))
fisher_jaccard_dist_adv = np.zeros((len(DATA_SETZ,)))
disr_jaccard_dist_adv = np.zeros((len(DATA_SETZ,)))

mim_kuncheva_dist_clean = np.zeros((len(DATA_SETZ,)))
mrmr_kuncheva_dist_clean = np.zeros((len(DATA_SETZ,)))
relief_kuncheva_dist_clean = np.zeros((len(DATA_SETZ,)))
fisher_kuncheva_dist_clean = np.zeros((len(DATA_SETZ,)))
disr_kuncheva_dist_clean = np.zeros((len(DATA_SETZ,)))

mim_kuncheva_dist_adv = np.zeros((len(DATA_SETZ,)))
mrmr_kuncheva_dist_adv = np.zeros((len(DATA_SETZ,)))
relief_kuncheva_dist_adv = np.zeros((len(DATA_SETZ,)))
fisher_kuncheva_dist_adv = np.zeros((len(DATA_SETZ,)))
disr_kuncheva_dist_adv = np.zeros((len(DATA_SETZ,)))

mim_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
mrmr_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
relief_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
fisher_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
disr_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))

mim_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
mrmr_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
relief_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
fisher_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
disr_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))

mim_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
mrmr_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
relief_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
fisher_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
disr_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))

mim_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
mrmr_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
relief_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
fisher_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
disr_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))

param = ['loss', 'accuracy']
attack_scores = np.zeros((len(DATA_SETZ), len(param)))

# loop over each of the datasets 
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

    # these sets will be used to determine the overall consistency of the feature selection model
    mim_feats_clean, mrmr_feats_clean, disr_feats_clean, relief_feats_clean, fisher_feats_clean = [], [], [], [], []
    mim_feats_adv, mrmr_feats_adv, disr_feats_adv, relief_feats_adv, fisher_feats_adv = [], [], [], [], []
    
    cum_perturbation = 0
    cum_loss = 0
    cum_accuracy = 0


    # for each dataset (1) split the tr/te data, (2) generate the adv data, (3) run fs & get stability 
    for train_index, test_index in kf.split(X):
        # split the original data into training / testing datasets. we are not going to 
        # use the testing data since we are not going to learn a classifier. 
        Xtr, ytr, Xte, yte = X[train_index,:], y[train_index], X[test_index,:], y[test_index]

        # convert the labels to a one-hot encoding for the attack model then fit the classifier 
        ytr_ohe = tf.keras.utils.to_categorical(ytr, 2)
        yte_ohe = tf.keras.utils.to_categorical(yte, 2)
        # model = create_network(Xtr, NUM_CLASSES)
        
        if CLFR == 'svc': 
            clfr = SVC(C=1.0, kernel='rbf')
            clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
            clfr.fit(Xtr, ytr_ohe)
        elif CLFR == 'lr':
            model = LogisticRegression()
            model.fit(Xtr, ytr)
            clfr = SklearnClassifier(model)
        elif CLFR == 'rfc': 
            model = RandomForestClassifier(n_estimators=50, max_depth=5)
            clfr = SklearnClassifier(model)
            model.fit(Xtr, ytr)
        elif CLFR == 'nn':
            model = create_nn(Xtr, NUM_CLASSES)
            clfr = KerasClassifier(model=model, clip_values=(-5, 5))
            clfr.fit(Xtr, ytr_ohe, verbose = 0)
        elif CLFR == 'mlp':
            model = create_network(Xtr, NUM_CLASSES)
            clfr = KerasClassifier(model=model, clip_values=(-5, 5))
            clfr.fit(Xtr, ytr_ohe, verbose = 0)
        else: 
            sys.exit('Unknown classifier for the attack.')
        #clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
        #clfr.fit(Xtr, ytr_ohe)

        # generate the attack samples: find out how many samples were are going to generate 
        n_cv = len(ytr)
        n_adv = int(n_cv*RATIO)
        if ATTACK_TYPE == 'fgsm': 
            # generate attacks using the fsgm methods. labels are assigned to the label that was 
            # used to generate the sample.
            attack_fgsm = FastGradientMethod(estimator=clfr, eps=0.3) 
            i_random = np.random.randint(0, len(yte), n_adv)
            Xadv = attack_fgsm.generate(x=Xte[i_random])
            yadv = np.random.randint(0, 2, n_adv)
            if ((CLFR == 'mlp') or (CLFR == 'nn')):
                loss_test, accuracy_test = model.evaluate(Xadv, yte_ohe[i_random])
                cum_perturbation += np.mean(np.abs((Xadv - Xte[i_random])))
                cum_loss += loss_test
                cum_accuracy += accuracy_test
                print("Perturbations: ", np.mean(np.abs((Xadv - Xte[i_random]))))
            #yadv = np.where(yadv==1)[1]    
        elif ATTACK_TYPE == 'deepfool':
            # generate attacks using the deepfool method. labels are assigned to the label that was 
            # used to generate the sample.  
            i_random = np.random.randint(0, len(yte), n_adv)
            attack_df = DeepFool(clfr)
            Xadv = attack_df.generate(x=Xte[i_random])
            yadv = np.random.randint(0, 2, n_adv)
            if ((CLFR == 'mlp') or (CLFR == 'nn')):
                loss_test, accuracy_test = model.evaluate(Xadv, yte_ohe[i_random])
                cum_perturbation += np.mean(np.abs((Xadv - Xte[i_random])))
                cum_loss += loss_test
                cum_accuracy += accuracy_test
                print("Perturbations: ",np.mean(np.abs((Xadv - Xte[i_random])))) 
        elif ATTACK_TYPE == 'jsma':
            # generate attacks using jsma method. 
            attack = SaliencyMapMethod(clfr)
            i_random = np.random.randint(0, len(yte), n_adv)
            Xadv = attack.generate(x=Xte[i_random], y=yte[i_random])
            yadv = np.random.randint(0, 2, n_adv)
            if ((CLFR == 'mlp') or (CLFR == 'nn')):
                loss_test, accuracy_test = model.evaluate(Xadv, yte_ohe[i_random])
                cum_perturbation += np.mean(np.abs((Xadv - Xte[i_random])))
                cum_loss += loss_test
                cum_accuracy += accuracy_test
                print("Perturbations: ", np.mean(np.abs((Xadv - Xte[i_random]))))
        elif ATTACK_TYPE == 'cleanlabel_pattern':
        
            # generate attacks using clean label backdoor
            backdoor = PoisoningAttackBackdoor(add_pattern_bd)  
            attack = PoisoningAttackCleanLabelBackdoor(backdoor, clfr, [0,1])
            print("Shape and type of yx: ", ytr_ohe.shape, type(ytr_ohe))
            print("shape and type of Xx: ",Xtr.shape, type(Xtr)) 
            #Xadv, yadv = attack.poison(x = Xte, y = yte_ohe) 
            Xa, yy = attack.poison(x = Xtr, y = ytr_ohe)
            #yadv = np.random.randint(0, 2, len(yy))
            #yadv = np.where(yadv==1)[1]
            ya = np.where(yy==1)[1]
            if ((CLFR == 'mlp') or (CLFR == 'nn')):
                print("RUNNING MLP/NN")
                #loss_test, accuracy_test = model.evaluate(Xadv, ytr_ohe) 
                loss_test, accuracy_test = model.evaluate(Xa, yy)
                #cum_perturbation += np.mean(np.abs((Xadv - Xte)))
                cum_perturbation += np.mean(np.abs((Xa - Xtr)))
                cum_loss += loss_test
                cum_accuracy += accuracy_test
                #print("Perturbations: ", np.mean(np.abs((Xadv - Xte))))
                print("Perturbations: ", np.mean(np.abs((Xa - Xtr))))
        elif ATTACK_TYPE == 'cleanlabel_single':
            backdoor = PoisoningAttackBackdoor(add_single_bd)
            attack = PoisoningAttackCleanLabelBackdoor(backdoor, clfr, [0,1])
            #Xadv, yadv = attack.poison(x = Xte, y = yte_ohe)
            Xa, yy = attack.poison(x = Xtr, y = ytr_ohe)
            #yadv = np.random.randint(0,2,len(yy))
            #yadv = np.where(yadv==1)[1]
            ya = np.where(yy==1)[1]
            if ((CLFR == 'mlp') or (CLFR == 'nn')):
                print("RUNNING MLP/NN")
                loss_test, accuracy_test = model.evaluate(Xa, yy)
                cum_perturbation += np.mean(np.abs((Xa - Xtr)))
                cum_loss += loss_test
                cum_accuracy += accuracy_test
                print("Perturbations: ", np.mean(np.abs((Xa - Xtr))))
        elif ATTACK_TYPE == 'embedding':
            target_idx = 1
            target = np.zeros(2)
            target[target_idx] = 1
            backdoor = PoisoningAttackBackdoor(add_pattern_bd)
            attack = PoisoningAttackAdversarialEmbedding(clfr, backdoor, 2, target, pp_poison = 0.2)
            bdclassifier = attack.poison_estimator(Xtr, ytr_ohe, nb_epochs = epochs)
            Xa, yy, bd = attack.get_training_data()
            ya = np.where(yy==1)[1]
            if CLFR == 'nn':
                print("RUNNING NN")
                loss_test, accuracy_test = model.evaluate(Xa, yy)
                cum_perturbation += np.mean(np.abs((Xa - Xtr))) 
                cum_loss += loss_test
                cum_accuracy += accuracy_test
                print("Perturbations: ", np.mean(np.abs((Xa - Xtr))))
        elif ATTACK_TYPE == 'svc': 
            # generate attacks using the biggio's svm attack from ICML 2012. 
            i_random = np.random.randint(0, len(yte), n_adv)

            poison = SklearnClassifier(model=SVC(kernel='rbf'), clip_values=(-5,5))
            poison.fit(Xtr, ytr_ohe)
            attack = PoisoningAttackSVM(poison, SVC_ATTACK_PARAM['step'], 
                                        SVC_ATTACK_PARAM['eps'], 
                                        Xtr, 
                                        ytr_ohe, 
                                        Xte, 
                                        yte_ohe, 
                                        SVC_ATTACK_PARAM['T'])
            Xx = Xte[i_random] 
            yx = yte_ohe[i_random] 

            attack_y = np.ones((len(yx),2)) - yx
            Xadv, _ = attack.poison(np.array(Xx), y=np.array(yx))
            yadv = yte[i_random]

        else: 
            sys.exit('Unknown attack type.')

        # We get an automated adversarial dataset (i.e., Xa, ya) out for causative attacks, however, for exploratory attacks, we generate malicious
        # samples call it Xadv, yadv and then concatenate them to the original benign dataset Xtr and ytr
        if ((ATTACK_TYPE == 'cleanlabel_single') or (ATTACK_TYPE == 'cleanlabel_pattern') or (ATTACK_TYPE == 'embedding')):
            Xa, ya = Xa, ya
        else:
            Xa, ya = np.concatenate((Xtr, Xadv)), np.concatenate((ytr, yadv))

        # save off the data just for the first time so we can generate plots of the data in the 
        # jupyter notebook. only save the fgsm data [to do: change this in the future]
        if k == 0: 
            data_output= ''.join(['outputs/Classifier_NN/deepfool/adversarial_data_', data_set, '_', ATTACK_TYPE, '.npz'])
            if ((ATTACK_TYPE == 'cleanlabel_single') or (ATTACK_TYPE == 'cleanlabel_pattern') or (ATTACK_TYPE == 'embedding')):
                np.savez(data_output, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, Xtr_pois=Xa, ytr_pois=ya)
            else:
                np.savez(data_output, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, Xadv=Xadv, yadv=yadv, Xtr_pois=Xa, ytr_pois=ya)
            sel_clean_mim_1, _, _ = MIM.mim(Xtr, ytr, n_selected_features=NFEATURES)
            sel_clean_mrmr_1, _, _ = MRMR.mrmr(Xtr, ytr, n_selected_features=NFEATURES)
            sel_clean_disr_1, _, _ = DISR.disr(Xtr, ytr, n_selected_features=NFEATURES)
            sel_clean_relief_1 = feature_ranking(reliefF(Xtr, ytr))[:NFEATURES]
            sel_clean_fisher_1 = feature_ranking(fisher_score(Xtr, ytr))[:NFEATURES]
            
        
               

        # MIM 
        sel_clean, _, m1 = MIM.mim(Xtr, ytr, n_selected_features=NFEATURES)
        sel_adv, _, m2 = MIM.mim(Xa, ya , n_selected_features=NFEATURES)
        mim_jaccard_dist_clean[i] += jaccard(sel_clean, sel_clean_mim_1)
        mim_jaccard_dist_adv[i] += jaccard(sel_adv, sel_clean)
        mim_kuncheva_dist_clean[i] += kuncheva(sel_clean, sel_clean_mim_1, nf)
        mim_kuncheva_dist_adv[i] += kuncheva(sel_adv, sel_clean, nf)
        mim_feats_clean.append(sel_clean)
        mim_feats_adv.append(sel_adv)
        
        # print("Jaccard_distance clean MIM\n",mim_jaccard_dist_clean )
        # print("\n Jaccard_distance adv MIM\n", mim_jaccard_dist_adv )

        # MRMR
        sel_clean, _, _ = MRMR.mrmr(Xtr, ytr, n_selected_features=NFEATURES)
        sel_adv, _, _ = MRMR.mrmr(Xa, ya, n_selected_features=NFEATURES)
        mrmr_jaccard_dist_clean[i] += jaccard(sel_clean, sel_clean_mrmr_1)
        mrmr_jaccard_dist_adv[i] += jaccard(sel_adv, sel_clean)
        mrmr_kuncheva_dist_clean[i] += kuncheva(sel_clean, sel_clean_mrmr_1, nf)
        mrmr_kuncheva_dist_adv[i] += kuncheva(sel_adv, sel_clean, nf)
        mrmr_feats_clean.append(sel_clean)
        mrmr_feats_adv.append(sel_adv)

        # DISR
        sel_clean, _, _ = DISR.disr(Xtr, ytr, n_selected_features=NFEATURES)
        sel_adv, _, _ = DISR.disr(Xa, ya, n_selected_features=NFEATURES)
        disr_jaccard_dist_clean[i] += jaccard(sel_clean, sel_clean_disr_1)
        disr_jaccard_dist_adv[i] += jaccard(sel_adv, sel_clean)
        disr_kuncheva_dist_clean[i] += kuncheva(sel_clean, sel_clean_disr_1, nf)
        disr_kuncheva_dist_adv[i] += kuncheva(sel_adv, sel_clean, nf)
        disr_feats_clean.append(sel_clean)
        disr_feats_adv.append(sel_adv)

        # reliefF 
        sel_clean = feature_ranking(reliefF(Xtr, ytr))[:NFEATURES]
        sel_adv = feature_ranking(reliefF(Xa, ya))[:NFEATURES]
        relief_jaccard_dist_clean[i] += jaccard(sel_clean, sel_clean_relief_1)
        relief_jaccard_dist_adv[i] += jaccard(sel_adv, sel_clean)
        relief_kuncheva_dist_clean[i] += kuncheva(sel_clean, sel_clean_relief_1, nf)
        relief_kuncheva_dist_adv[i] += kuncheva(sel_adv, sel_clean, nf)
        relief_feats_clean.append(sel_clean)
        relief_feats_adv.append(sel_adv)
        
        # fisher score 
        sel_clean = feature_ranking(fisher_score(Xtr, ytr))[:NFEATURES]
        sel_adv = feature_ranking(fisher_score(Xa, ya))[:NFEATURES]
        fisher_jaccard_dist_clean[i] += jaccard(sel_clean, sel_clean_fisher_1)
        fisher_jaccard_dist_adv[i] += jaccard(sel_adv, sel_clean)
        fisher_kuncheva_dist_clean[i] += kuncheva(sel_clean, sel_clean_fisher_1, nf)
        fisher_kuncheva_dist_adv[i] += kuncheva(sel_adv, sel_clean, nf)
        fisher_feats_clean.append(sel_clean)
        fisher_feats_adv.append(sel_adv)

        k += 1
        mean_perturbation = cum_perturbation/TRIALS
        attack_scores[i, 0] = cum_loss/TRIALS
        attack_scores[i, 1] = cum_accuracy/TRIALS

    if ((CLFR == 'mlp') or (CLFR == 'nn')):
        print(DATA_SETZ[i])
        print("\nMean Perturbation for ", ATTACK_TYPE, 'attacks: ', mean_perturbation)
        print("Mean Loss for ", ATTACK_TYPE, 'attacks: ', cum_loss/TRIALS)
        print("Mean Accuracy for ", ATTACK_TYPE, 'attacks: ', cum_accuracy/TRIALS)    

    # scale the jaccard distance for CLEAN DATA by the number of cross fold runs 
    mim_jaccard_dist_clean[i] = mim_jaccard_dist_clean[i]/TRIALS
    mrmr_jaccard_dist_clean[i] = mrmr_jaccard_dist_clean[i]/TRIALS
    disr_jaccard_dist_clean[i] = disr_jaccard_dist_clean[i]/TRIALS
    relief_jaccard_dist_clean[i] = relief_jaccard_dist_clean[i]/TRIALS
    fisher_jaccard_dist_clean[i] = fisher_jaccard_dist_clean[i]/TRIALS
    
    # scale the jaccard distance of ADV DATA from clean data by the number of cross fold runs 
    mim_jaccard_dist_adv[i] = mim_jaccard_dist_adv[i]/TRIALS
    mrmr_jaccard_dist_adv[i] = mrmr_jaccard_dist_adv[i]/TRIALS
    disr_jaccard_dist_adv[i] = disr_jaccard_dist_adv[i]/TRIALS
    relief_jaccard_dist_adv[i] = relief_jaccard_dist_adv[i]/TRIALS
    fisher_jaccard_dist_adv[i] = fisher_jaccard_dist_adv[i]/TRIALS
    
    # scale the kuncheva distance of CLEAN DATA by the number of cross fold runs 
    mim_kuncheva_dist_clean[i] = mim_kuncheva_dist_clean[i]/TRIALS
    mrmr_kuncheva_dist_clean[i] = mrmr_kuncheva_dist_clean[i]/TRIALS
    disr_kuncheva_dist_clean[i] = disr_kuncheva_dist_clean[i]/TRIALS
    relief_kuncheva_dist_clean[i] = relief_kuncheva_dist_clean[i]/TRIALS
    fisher_kuncheva_dist_clean[i] = fisher_kuncheva_dist_clean[i]/TRIALS
    
    # scale the kuncheva distance of ADV DATA from clean data by the number of cross fold runs 
    mim_kuncheva_dist_adv[i] = mim_kuncheva_dist_adv[i]/TRIALS
    mrmr_kuncheva_dist_adv[i] = mrmr_kuncheva_dist_adv[i]/TRIALS
    disr_kuncheva_dist_adv[i] = disr_kuncheva_dist_adv[i]/TRIALS
    relief_kuncheva_dist_adv[i] = relief_kuncheva_dist_adv[i]/TRIALS
    fisher_kuncheva_dist_adv[i] = fisher_kuncheva_dist_adv[i]/TRIALS
    
    # measure the overall consistency of the feature selection algorithm without adversarial data
    mim_jaccard_consistency_clean[i], mim_kuncheva_consistency_clean[i] = total_consistency(mim_feats_clean, nf)
    mrmr_jaccard_consistency_clean[i], mrmr_kuncheva_consistency_clean[i] = total_consistency(mrmr_feats_clean, nf)
    disr_jaccard_consistency_clean[i], disr_kuncheva_consistency_clean[i] = total_consistency(disr_feats_clean, nf)
    relief_jaccard_consistency_clean[i], relief_kuncheva_consistency_clean[i] = total_consistency(relief_feats_clean, nf)
    fisher_jaccard_consistency_clean[i], fisher_kuncheva_consistency_clean[i] = total_consistency(fisher_feats_clean, nf)
    
    mim_jaccard_consistency_adv[i], mim_kuncheva_consistency_adv[i] = total_consistency(mim_feats_adv, nf)
    mrmr_jaccard_consistency_adv[i], mrmr_kuncheva_consistency_adv[i] = total_consistency(mrmr_feats_adv, nf)
    disr_jaccard_consistency_adv[i], disr_kuncheva_consistency_adv[i] = total_consistency(disr_feats_adv, nf)
    relief_jaccard_consistency_adv[i], relief_kuncheva_consistency_adv[i] = total_consistency(relief_feats_adv, nf)
    fisher_jaccard_consistency_adv[i], fisher_kuncheva_consistency_adv[i] = total_consistency(fisher_feats_adv, nf)

        
# write the output 
if not os.path.isdir('outputs/Classifier_NN/deepfool/'):
    os.mkdir('outputs/Classifier_NN/deepfool/')

np.savez(OUTPUT_FILE, 
         mim_jaccard_dist_clean = mim_jaccard_dist_clean,
         mim_jaccard_dist_adv = mim_jaccard_dist_adv, 
         mrmr_jaccard_dist_clean=mrmr_jaccard_dist_clean,
         mrmr_jaccard_dist_adv=mrmr_jaccard_dist_adv, 
         disr_jaccard_dist_clean=disr_jaccard_dist_clean,
         disr_jaccard_dist_adv=disr_jaccard_dist_adv,
         relief_jaccard_dist_clean = relief_jaccard_dist_clean,
         relief_jaccard_dist_adv=relief_jaccard_dist_adv, 
         fisher_jaccard_dist_clean=fisher_jaccard_dist_clean,  
         fisher_jaccard_dist_adv=fisher_jaccard_dist_adv,  
         mim_kuncheva_dist_clean=mim_kuncheva_dist_clean,
         mim_kuncheva_dist_adv=mim_kuncheva_dist_adv, 
         mrmr_kuncheva_dist_clean=mrmr_kuncheva_dist_clean,
         mrmr_kuncheva_dist_adv=mrmr_kuncheva_dist_adv, 
         disr_kuncheva_dist_clean = disr_kuncheva_dist_clean,
         disr_kuncheva_dist_adv = disr_kuncheva_dist_adv,
         relief_kuncheva_dist_clean = relief_kuncheva_dist_clean,
         relief_kuncheva_dist_adv = relief_kuncheva_dist_adv, 
         fisher_kuncheva_dist_clean = fisher_kuncheva_dist_clean,
         fisher_kuncheva_dist_adv = fisher_kuncheva_dist_adv,         
         mim_jaccard_consistency_clean=mim_jaccard_consistency_clean, 
         mrmr_jaccard_consistency_clean=mrmr_jaccard_consistency_clean, 
         disr_jaccard_consistency_clean=disr_jaccard_consistency_clean,
         relief_jaccard_consistency_clean=relief_jaccard_consistency_clean, 
         fisher_jaccard_consistency_clean=fisher_jaccard_consistency_clean,  
         mim_kuncheva_consistency_clean=mim_kuncheva_consistency_clean, 
         mrmr_kuncheva_consistency_clean=mrmr_kuncheva_consistency_clean, 
         disr_kuncheva_consistency_clean=disr_kuncheva_consistency_clean,
         relief_kuncheva_consistency_clean=relief_kuncheva_consistency_clean, 
         fisher_kuncheva_consistency_clean=fisher_kuncheva_consistency_clean, 
         mim_jaccard_consistency_adv=mim_jaccard_consistency_adv, 
         mrmr_jaccard_consistency_adv=mrmr_jaccard_consistency_adv, 
         disr_jaccard_consistency_adv=disr_jaccard_consistency_adv,
         relief_jaccard_consistency_adv=relief_jaccard_consistency_adv, 
         fisher_jaccard_consistency_adv=fisher_jaccard_consistency_adv,  
         mim_kuncheva_consistency_adv=mim_kuncheva_consistency_adv, 
         mrmr_kuncheva_consistency_adv=mrmr_kuncheva_consistency_adv, 
         disr_kuncheva_consistency_adv=disr_kuncheva_consistency_adv,
         relief_kuncheva_consistency_adv=relief_kuncheva_consistency_adv, 
         fisher_kuncheva_consistency_adv=fisher_kuncheva_consistency_adv, 
         mean_perturbation = mean_perturbation,
         MLP_attack_scores = attack_scores,
         TRIALS=TRIALS, 
         CLFR=CLFR, 
         ATTACK_TYPE=ATTACK_TYPE, 
         RATIO=RATIO, 
         DATANAMES=DATA_SETZ)
