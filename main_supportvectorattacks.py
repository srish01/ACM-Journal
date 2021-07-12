import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf

from art.attacks.evasion import FastGradientMethod, DeepFool
from art.estimators.classification import SklearnClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from art.utils import to_categorical

from utils import jaccard, kuncheva, total_consistency  
from skfeature.function.information_theoretical_based import MIM, MRMR, DISR
from skfeature.function.similarity_based.fisher_score import fisher_score, feature_ranking
from skfeature.function.similarity_based.reliefF import reliefF 

plt.style.use('bmh')

 # number of cross fold runs to perform
TRIALS = 5
# adversarial sample ratio
RATIO = 0.2
# number of features to select
NFEATURES = 10
# number of classes
NUM_CLASSES = 2
# Attack Type
ATTACK_TYPE = 'supportvector_attack'
# path the the data folder [must be downloaded from Gitlab]
BASE_PATH = 'data/'
# Path for SVC attacked dataset
ADV_PATH = 'outputs/Classifier_SVC/SupportVectorAttacks/'
# location where output file will be saved
OUTPUT_FILE = ''.join(['outputs/Classifier_SVC/SupportVectorAttacks/experiment2_', ATTACK_TYPE, '_svc.npz'])

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

# Jaccard distance of features selected from benign data to adversarial data

mim_jaccard_dist = np.zeros((len(DATA_SETZ,)))
mrmr_jaccard_dist = np.zeros((len(DATA_SETZ,)))
relief_jaccard_dist = np.zeros((len(DATA_SETZ,)))
fisher_jaccard_dist = np.zeros((len(DATA_SETZ,)))
disr_jaccard_dist = np.zeros((len(DATA_SETZ,)))

# Kuncheva distance of features selected from benign data to adversarial data
mim_kuncheva_dist = np.zeros((len(DATA_SETZ,)))
mrmr_kuncheva_dist = np.zeros((len(DATA_SETZ,)))
relief_kuncheva_dist = np.zeros((len(DATA_SETZ,)))
fisher_kuncheva_dist = np.zeros((len(DATA_SETZ,)))
disr_kuncheva_dist = np.zeros((len(DATA_SETZ,)))

# Jaccard consistency of benign features
mim_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
mrmr_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
relief_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
fisher_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,)))
disr_jaccard_consistency_clean = np.zeros((len(DATA_SETZ,))) 
                
# Jaccard consistency of adversarial features
mim_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
mrmr_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
relief_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
fisher_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,)))
disr_jaccard_consistency_adv = np.zeros((len(DATA_SETZ,))) 
    
# Kuncheva consistency of benign features
mim_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
mrmr_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
relief_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
fisher_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,)))
disr_kuncheva_consistency_clean = np.zeros((len(DATA_SETZ,))) 

# Kuncheva consistency of adversarial features
mim_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
mrmr_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
relief_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
fisher_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,)))
disr_kuncheva_consistency_adv = np.zeros((len(DATA_SETZ,))) 

for i in range(len(DATA_SETZ)):
    data_set = DATA_SETZ[i]
    
    D = np.load(''.join([ADV_PATH,'adversarial_data_', data_set, '_svc.npz'])) 
    Xtrk, ytrk, Xtek, ytek, Xadvk, yadvk = D['Xtr'], D['ytr'], D['Xte'], D['yte'], D['Xadv'], D['yadv']
    
    print(data_set)
    print('  > Benign: ' + str(len(ytrk)))
    print('  > Adversarial: ' + str(len(yadvk)))
    
    p0, p1 = 1./TRIALS, (1. - 1./TRIALS)
    N = len(Xtrk)
    Ntr, Nte = int(p1*N), int(p0*N)
    A = len(Xadvk)
    
    Xn, yn = np.concatenate((Xtrk, Xtek)), np.concatenate((ytrk, ytek))
    n, nf = Xn.shape
    
    mim_feats_clean, mrmr_feats_clean, disr_feats_clean, relief_feats_clean, fisher_feats_clean = [], [], [], [], [] 
    mim_feats_adv, mrmr_feats_adv, disr_feats_adv, relief_feats_adv, fisher_feats_adv = [], [], [], [], []
    
    for _ in range(TRIALS):
        i_perm = np.random.permutation(N)
        Xtr, ytr, Xte, yte = Xn[i_perm][:Ntr], yn[i_perm][:Ntr], Xn[i_perm][Nte:], yn[i_perm][Nte:]
        
        j_perm = np.random.permutation(A)
        Xadv, yadv = Xadvk[j_perm], yadvk[j_perm]
        
        Xa, ya = np.concatenate((Xtr, Xadv)), np.concatenate((ytr, yadv))
        
        
        #MIM
        sel_clean,_,_ = MIM.mim(Xtr, ytr, n_selected_features=NFEATURES)
        sel_adv, _,_ = MIM.mim(Xa, ya, n_selected_features=NFEATURES)
        mim_jaccard_dist[i] +=jaccard(sel_adv, sel_clean)
        mim_kuncheva_dist[i] += kuncheva(sel_adv, sel_clean, nf)
        mim_feats_clean.append(sel_clean)
        mim_feats_adv.append(sel_adv)
        
        #MRMR
        sel_clean,_,_ = MRMR.mrmr(Xtr, ytr, n_selected_features=NFEATURES)
        sel_adv,_,_ = MRMR.mrmr(Xa, ya, n_selected_features=NFEATURES)
        mrmr_jaccard_dist[i] += jaccard(sel_adv, sel_clean)
        mrmr_kuncheva_dist[i] += kuncheva(sel_adv, sel_clean, nf)
        mrmr_feats_clean.append(sel_clean)
        mrmr_feats_adv.append(sel_adv)
    
        #DISR
        sel_clean,_,_ = DISR.disr(Xtr, ytr, n_selected_features=NFEATURES)
        sel_adv,_,_ = DISR.disr(Xa, ya, n_selected_features=NFEATURES)
        disr_jaccard_dist[i] += jaccard(sel_adv, sel_clean)
        disr_kuncheva_dist[i] += kuncheva(sel_adv, sel_clean, nf)
        disr_feats_clean.append(sel_clean)
        disr_feats_adv.append(sel_adv)
        
        #reliefF
        sel_clean = feature_ranking(reliefF(Xtr, ytr))[:NFEATURES] 
        sel_adv = feature_ranking(reliefF(Xa, ya))[:NFEATURES] 
        relief_jaccard_dist[i] += jaccard(sel_adv, sel_clean)
        relief_kuncheva_dist[i] += kuncheva(sel_adv, sel_clean, nf)
        relief_feats_clean.append(sel_clean)
        relief_feats_adv.append(sel_adv)
        
        #fisher score
        sel_clean = feature_ranking(fisher_score(Xtr, ytr))[:NFEATURES] 
        sel_adv = feature_ranking(fisher_score(Xa, ya))[:NFEATURES] 
        fisher_jaccard_dist[i] += jaccard(sel_adv, sel_clean)
        fisher_kuncheva_dist[i] += kuncheva(sel_adv, sel_clean, nf)
        fisher_feats_clean.append(sel_clean)
        fisher_feats_adv.append(sel_adv)
    
    
    # scale the jaccard distance by the number of cross fold runs
    mim_jaccard_dist[i] = mim_jaccard_dist[i]/TRIALS
    mrmr_jaccard_dist[i] = mrmr_jaccard_dist[i]/TRIALS
    disr_jaccard_dist[i] = disr_jaccard_dist[i]/TRIALS
    relief_jaccard_dist[i] = relief_jaccard_dist[i]/TRIALS
    fisher_jaccard_dist[i] = fisher_jaccard_dist[i]/TRIALS  
        
    # scale the kuncheva distance by the number of cross fold runs
    mim_kuncheva_dist[i] = mim_kuncheva_dist[i]/TRIALS
    mrmr_kuncheva_dist[i] = mrmr_kuncheva_dist[i]/TRIALS
    disr_kuncheva_dist[i] = disr_kuncheva_dist[i]/TRIALS
    relief_kuncheva_dist[i] = relief_kuncheva_dist[i]/TRIALS
    fisher_kuncheva_dist[i] = fisher_kuncheva_dist[i]/TRIALS
    
    # measure the overall consistency of the feature selection algorithm on benign data 
    mim_jaccard_consistency_clean[i], mim_kuncheva_consistency_clean[i] = total_consistency(mim_feats_clean, nf)
    mrmr_jaccard_consistency_clean[i], mrmr_kuncheva_consistency_clean[i] = total_consistency(mrmr_feats_clean, nf)
    disr_jaccard_consistency_clean[i], disr_kuncheva_consistency_clean[i] = total_consistency(disr_feats_clean, nf)
    relief_jaccard_consistency_clean[i], relief_kuncheva_consistency_clean[i] = total_consistency(relief_feats_clean, nf)
    fisher_jaccard_consistency_clean[i], fisher_kuncheva_consistency_clean[i] = total_consistency(fisher_feats_clean, nf)
    
    # measure the overall consistency of the feature selection algorithm on adversarial data 
    mim_jaccard_consistency_adv[i], mim_kuncheva_consistency_adv[i] = total_consistency(mim_feats_adv, nf)
    mrmr_jaccard_consistency_adv[i], mrmr_kuncheva_consistency_adv[i] = total_consistency(mrmr_feats_adv, nf)
    disr_jaccard_consistency_adv[i], disr_kuncheva_consistency_adv[i] = total_consistency(disr_feats_adv, nf) 
    relief_jaccard_consistency_adv[i], relief_kuncheva_consistency_adv[i] = total_consistency(relief_feats_adv, nf)
    fisher_jaccard_consistency_adv[i], fisher_kuncheva_consistency_adv[i] = total_consistency(fisher_feats_adv, nf) 
    

# write the output
if not os.path.isdir('outputs/Classifier_SVC/SupportVectorAttacks/'):
    os.mkdir('outputs/Classifier_SVC/SupportVectorAttacks/')  
np.savez(OUTPUT_FILE,
        mim_jaccard_dist = mim_jaccard_dist,
        mim_kuncheva_dist = mim_kuncheva_dist,
        mrmr_jaccard_dist = mrmr_jaccard_dist,
        mrmr_kuncheva_dist = mrmr_kuncheva_dist,
        disr_jaccard_dist = disr_jaccard_dist,
        disr_kuncheva_dist = disr_kuncheva_dist,
        relief_jaccard_dist = relief_jaccard_dist,
        relief_kuncheva_dist = relief_kuncheva_dist,
        fisher_jaccard_dist = fisher_jaccard_dist,
        fisher_kuncheva_dist = fisher_kuncheva_dist,
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
        TRIALS=TRIALS,
        ATTACK_TYPE=ATTACK_TYPE,
        DATANAMES=DATA_SETZ
        )