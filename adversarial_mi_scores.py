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
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.information_theoretical_based.MIM import mim
from skfeature.function.information_theoretical_based.DISR import disr

from art.attacks.evasion import FastGradientMethod, DeepFool
from art.estimators.classification import SklearnClassifier
from art.attacks.poisoning import PoisoningAttackSVM

from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif

plt.style.use('bmh')


# number of cross validation runs to perform 
RUNS = 5
# number of features to rank 
NBEST = 10
# attack type: svc, fgsm, deepfool
ATTACK_TYPE = 'fgsm'
# this constant determines if the labels for deepfool or fgsm are assigned at random 
# but will do nothing for svc attacks 
RANDOM_LABEL = False
# svc attack parameters
SVC_ATTACK_PARAM ={'T':25, 'step':0.1, 'eps':1.}
# ratio setting up the number of adversarial samples in the poisoned dataset 
RATIO = .15
# this is used to construct the output file name 
if RANDOM_LABEL: 
    RLABEL = '_randlabel'
else: 
    RLABEL = ''
# get the names of the datasets
data_sets = [file[:-4] for file in os.listdir('data/')]

# loop over all of the datasets 
for ell in range(len(data_sets)): 
    # if data_sets[ell] == 'statlog-german-credit' or data_sets[ell] == 'ozone':
    #     pass 
    print(''.join(['Running ', data_sets[ell]]))

    # load the data and separate out the data and labels 
    df = pd.read_csv(''.join(['data/', data_sets[ell], '.csv']), header=None)
    X, y = df.values[:,:-1], df.values[:,-1]
    
    # shuffle the data 
    i_perm = np.random.permutation(len(y))
    X, y = X[i_perm], y[i_perm]
    ns = len(y)
    
    # split the data into tr/te splits 
    ns = int(np.floor(0.8*ns))
    Xtr, ytr, Xte, yte = X[:ns], y[:ns], X[ns:], y[ns:]

    mim_scores_norm, mim_scores_adv = np.zeros((X.shape[1],)), np.zeros((X.shape[1],))
    mrmr_scores_norm, mrmr_scores_adv = np.zeros((X.shape[1],)), np.zeros((X.shape[1],))
    disr_scores_norm, disr_scores_adv = np.zeros((X.shape[1],)), np.zeros((X.shape[1],))
    
    # generate the attack samples: find out how many samples were are going to generate 
    n_cv = len(ytr)
    n_adv = int(n_cv*RATIO)
    
    for k in range(RUNS): 
        # get a random sample of the normal data and an adversarial dataset 
        i_random_tr = np.random.randint(0, len(ytr), len(ytr))

        # convert the labels to a one-hot encoding for the attack model then fit the classifier 
        ytr_ohe = tf.keras.utils.to_categorical(ytr, 2)
        yte_ohe = tf.keras.utils.to_categorical(yte, 2)
    
        if ATTACK_TYPE == 'svc': 
            poison = SklearnClassifier(model=SVC(kernel='rbf'), clip_values=(-5,5))
            poison.fit(Xtr, ytr_ohe)
            attack = PoisoningAttackSVM(poison, SVC_ATTACK_PARAM['step'], 
                                        SVC_ATTACK_PARAM['eps'], 
                                        Xtr[i_random_tr], 
                                        ytr_ohe[i_random_tr], 
                                        Xte, 
                                        yte_ohe, 
                                        SVC_ATTACK_PARAM['T'])
            i_random_2 = np.random.randint(0, len(yte), n_adv)
            Xx = Xte[i_random_2] 
            yx = yte_ohe[i_random_2] 

            attack_y = np.ones((len(yx),2)) - yx
            Xadv, _ = attack.poison(np.array(Xx), y=np.array(yx))
            yadv = yte[i_random_2]
        elif ATTACK_TYPE == 'fgsm': 
            clfr = SVC(C=1.0, kernel='rbf')
            clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
            clfr.fit(Xtr[i_random_tr], ytr_ohe[i_random_tr])
            # generate attacks using the fsgm methods. labels are assigned to the label that was 
            # used to generate the sample. 
            attack = FastGradientMethod(estimator=clfr, eps=.2)
            i_random = np.random.randint(0, len(yte), n_adv)
            Xadv = attack.generate(x=Xte[i_random])

            if RANDOM_LABEL: 
                yadv = np.random.randint(0, 2, len(Xadv))
            else: 
                yadv = yte[i_random]

        elif ATTACK_TYPE == 'deepfool':
            clfr = SVC(C=1.0, kernel='rbf')
            clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
            clfr.fit(Xtr, ytr_ohe)
            # generate attacks using the deepfool method. labels are assigned to the label that was 
            # used to generate the sample.  
            attack = DeepFool(clfr)
            i_random = np.random.randint(0, len(yte), n_adv)
            Xadv = attack.generate(x=Xte[i_random])
            
            if RANDOM_LABEL: 
                yadv = np.random.randint(0, 2, len(Xadv))
            else: 
                yadv = yte[i_random]

        else: 
            sys.exit('Unknown attack type.')
        
        
        Xa, ya = np.concatenate((Xtr[i_random_tr], Xadv)), np.concatenate((ytr[i_random_tr], yadv))

        # MIM - Normal 
        mi_score = mutual_info_classif(Xtr[i_random_tr], ytr[i_random_tr])
        mim_scores_norm += mi_score
        # MIM - Adversarial 
        mi_score = mutual_info_classif(Xa, ya)
        mim_scores_adv += mi_score
        
        # mRMR - Normal 
        _, mi_score, _ = mrmr(Xtr[i_random_tr], ytr[i_random_tr], n_selected_features=X.shape[1])
        mrmr_scores_norm += mi_score
        # mRMR - Adversarial 
        _, mi_score, _ = mrmr(Xa, ya, n_selected_features=X.shape[1])
        mrmr_scores_adv += mi_score

        # DISR - Normal 
        _, mi_score, _ = disr(Xtr[i_random_tr], ytr[i_random_tr], n_selected_features=X.shape[1])
        disr_scores_norm += mi_score
        # DISR - Adversarial 
        _, mi_score, _ = disr(Xa, ya, n_selected_features=X.shape[1])
        disr_scores_adv += mi_score


    
    # clean up MIM scores 
    mim_scores_norm /= RUNS
    mim_scores_adv /= RUNS
    i_sorted = np.argsort(mim_scores_norm)[::-1]

    i_sorted = i_sorted[:NBEST]
    mim_scores_adv = mim_scores_adv[i_sorted]
    mim_scores_norm = mim_scores_norm[i_sorted]

    # clean up mRMR scores 
    mrmr_scores_norm /= RUNS
    mrmr_scores_adv /= RUNS
    i_sorted = np.argsort(mrmr_scores_norm)[::-1]

    i_sorted = i_sorted[:NBEST]
    mrmr_scores_adv = mrmr_scores_adv[i_sorted]
    mrmr_scores_norm = mrmr_scores_norm[i_sorted]

    # clean up mRMR scores 
    disr_scores_norm /= RUNS
    disr_scores_adv /= RUNS
    i_sorted = np.argsort(disr_scores_norm)[::-1]

    i_sorted = i_sorted[:NBEST]
    disr_scores_adv = disr_scores_adv[i_sorted]
    disr_scores_norm = disr_scores_norm[i_sorted]


    x = [i for i in range(len(i_sorted))]

    # ------------------------------------------------------------------------------------------------
    # create plot: mim scores 
    n_groups = len(x)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1.

    rects1 = plt.bar(index, mim_scores_norm, bar_width, alpha=opacity, label='Benign')
    rects2 = plt.bar(index+bar_width, mim_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    plt.xlabel('Feature Rank')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.tight_layout()
    plt.savefig(''.join(['outputs/barchar_mi_', data_sets[ell], RLABEL, '_', ATTACK_TYPE, '.pdf']))
    plt.close()


    # ------------------------------------------------------------------------------------------------
    # create plot: mrmr scores 
    n_groups = len(x)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1.

    rects1 = plt.bar(index, mrmr_scores_norm, bar_width, alpha=opacity, label='Benign')
    rects2 = plt.bar(index+bar_width, mrmr_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    plt.xlabel('Feature Rank')
    plt.ylabel('mRMR Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(''.join(['outputs/barchar_mrmr_', data_sets[ell], RLABEL, '_', ATTACK_TYPE, '.pdf']))
    plt.close()

    # ------------------------------------------------------------------------------------------------
    # create plot: disr scores 
    n_groups = len(x)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1.

    rects1 = plt.bar(index, disr_scores_norm, bar_width, alpha=opacity, label='Benign')
    rects2 = plt.bar(index+bar_width, disr_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    plt.xlabel('Feature Rank')
    plt.ylabel('mRMR Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(''.join(['outputs/barchar_disr_', data_sets[ell], RLABEL, '_', ATTACK_TYPE, '.pdf']))
    plt.close()
