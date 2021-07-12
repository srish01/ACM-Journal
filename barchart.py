import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.information_theoretical_based.MIM import mim
from skfeature.function.information_theoretical_based.DISR import disr

from skfeature.function.similarity_based.fisher_score import fisher_score, feature_ranking
from skfeature.function.similarity_based.reliefF import reliefF

from art.attacks.evasion import FastGradientMethod, DeepFool
from art.estimators.classification import SklearnClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from art.utils import to_categorical
from sklearn.manifold import TSNE
import xlwt
from xlwt import Workbook
import xlsxwriter

import random
from tqdm import tqdm

plt.style.use('bmh')

# get the names of the datasets
# data_sets = [file[:-4] for file in os.listdir('data/')]

data_sets = ['breast-cancer-wisc-diag',
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


# To load adversarial data (having both benign and malicous samples), Xtr_pois, ytr_pois is used for CleanLabel
# and Embedding attacks, else it is Xadv, yadv
 
fs_algo = ['MIM', 'MRMR', 'DISR', 'Fisher', 'Relief']
classifier = ['Classifier_LR', 'Classifier_MLP', 'Classifier_NN', 'Classifier_SVC'] 
attack = 'jsma' # deepfool, #fgsm, jsma, CleanLabel/Single, CleanLabel/Pattern, Embedding
post = 'jsma'   # #deepfool, fgsm, jsma, cleanlabel_single, cleanlabel_pattern
RUNS = 5
SEED = 1
NBEST = 10
np.random.seed(SEED)

#OUTPUT_FILE = ''.join(['outputs/Classifier_LR/deepfool/experiment_', attack, '_', CLFR, '.npz'])


for j in range(len(data_sets)):
#for j in range(5):
    # We will use these following lists to collect scores for all classifiers on a single dataset
    mim_scores_norm_all_clfr, mim_scores_adv_all_clfr = [], []
    mrmr_scores_norm_all_clfr, mrmr_scores_adv_all_clfr = [], []
    disr_scores_norm_all_clfr, disr_scores_adv_all_clfr = [], []
    relief_scores_norm_all_clfr, relief_scores_adv_all_clfr = [], []
    fisher_scores_norm_all_clfr, fisher_scores_adv_all_clfr = [], []
    
    norm_feature_matrix_all_clf, adv_feature_matrix_all_clf = [], []

    OUTPUT_FILE = ''.join(['outputs/barcharts/', attack, '/', data_sets[j], '_', post, '_top10.npz'])
    
    #index = np.array([])  # keeps track of number of groups for all the attack classifiers run on a single dataset
    n_groups = 0

    for i in range(len(classifier)):

        #benign_clf, adv_clf = [], []

        d = np.load('outputs/' + classifier[i] + '/' + attack + '/' + 'adversarial_data_' + data_sets[j] + '_' + post + '.npz')
        Xtr, ytr = d['Xtr'], d['ytr']
        if ((attack == 'deepfool') or (attack == 'fgsm') or (attack == 'jsma')):
            Xtr_pois, ytr_pois = d['Xadv'], d['yadv']
        else:
            Xtr_pois, ytr_pois = d['Xtr_pois'], d['ytr_pois']
       
        mim_scores_norm, mim_scores_adv = np.zeros((Xtr.shape[1],)), np.zeros((Xtr.shape[1],))
        mrmr_scores_norm, mrmr_scores_adv = np.zeros((Xtr.shape[1],)), np.zeros((Xtr.shape[1],))
        disr_scores_norm, disr_scores_adv = np.zeros((Xtr.shape[1],)), np.zeros((Xtr.shape[1],))
        relief_scores_norm, relief_scores_adv = np.zeros((Xtr.shape[1],)), np.zeros((Xtr.shape[1],))
        fisher_scores_norm, fisher_scores_adv = np.zeros((Xtr.shape[1],)), np.zeros((Xtr.shape[1],))


       
        for k in tqdm(range(RUNS)):           
            
            if ((attack == 'deepfool') or (attack == 'fgsm') or (attack == 'jsma')):
                Xa, ya = np.concatenate((Xtr, Xtr_pois)), np.concatenate((ytr, ytr_pois))
            else:
                Xa, ya = Xtr_pois, ytr_pois
            
            i_perm = np.random.permutation(len(Xtr))
            Xtr, ytr = Xtr[i_perm], ytr[i_perm]
           
            j_perm = np.random.permutation(len(Xa))
            Xa, ya = Xa[j_perm], ya[j_perm]
            

            # MIM -
            # Normal
            mi_score = mutual_info_classif(Xtr, ytr)
            mim_scores_norm += mi_score
            # Adversarial
            mi_score = mutual_info_classif(Xa, ya)
            mim_scores_adv += mi_score
       
            # mRMR -
            # Normal
            _, mi_score, _ = mrmr(Xtr, ytr, n_selected_features=Xtr.shape[1])
            mrmr_scores_norm += mi_score
            # Adversarial
            _, mi_score, _ = mrmr(Xa, ya, n_selected_features=Xtr.shape[1])
            mrmr_scores_adv += mi_score
 
            # DISR -
            # Normal
            _, mi_score, _ = disr(Xtr, ytr, n_selected_features=Xtr.shape[1])
            disr_scores_norm += mi_score
            # Adversarial
            _, mi_score, _ = disr(Xa, ya, n_selected_features=Xtr.shape[1])
            disr_scores_adv += mi_score
           
            # Relief -
            # Normal
            r_scores = reliefF(Xtr, ytr)
            relief_scores_norm += r_scores
            # Adversarial
            r_scores = reliefF(Xa, ya)
            relief_scores_adv += r_scores
           
            # Fisher -
            # Normal
            f_scores = fisher_score(Xtr, ytr)
            #print("Clean Fisher Score: ", f_scores, "for RUN: ", k+1)
            fisher_scores_norm += f_scores
            # Adversarial
            f_scores = fisher_score(Xa, ya)
            #print("Adv Fisher Score: ", f_scores, "for RUN: ", k+1)
            fisher_scores_adv += f_scores
           

 
   
        # clean up MIM scores
        mim_scores_norm /= RUNS
        mim_scores_adv /= RUNS
        i_sorted = np.argsort(mim_scores_norm)[::-1]
        adv_sorted = np.argsort(mim_scores_adv)[::-1]
       
        i_sorted_mim = i_sorted[:NBEST]
        adv_sorted_mim = adv_sorted[:NBEST]

        # print("MIM: top 10 features from normal data for", data_sets[j], ": ", i_sorted_mim)
        # print("MIM: top 10 features from adversarial data for ", data_sets[j], ": ",  adv_sorted_mim)
        
        mim_scores_adv = mim_scores_adv[i_sorted_mim]
        mim_scores_adv_all_clfr +=( mim_scores_adv.tolist() + [0])
        
        mim_scores_norm = mim_scores_norm[i_sorted_mim]
        mim_scores_norm_all_clfr += (mim_scores_norm.tolist() + [0]) # works like an append
 
        # clean up mRMR scores
        mrmr_scores_norm /= RUNS
        mrmr_scores_adv /= RUNS
        i_sorted = np.argsort(mrmr_scores_norm)[::-1]
        adv_sorted = np.argsort(mrmr_scores_adv)[::-1]
       
        i_sorted_mrmr = i_sorted[:NBEST]
        adv_sorted_mrmr = adv_sorted[:NBEST]

        # print("MRMR: top 10 features from normal data for ", data_sets[j], ": ", i_sorted_mrmr)
        # print("MRMR: top 10 features from adversarial data for ", data_sets[j], ": ", adv_sorted_mrmr)
        
        mrmr_scores_adv = mrmr_scores_adv[i_sorted_mrmr]
        mrmr_scores_adv_all_clfr += (mrmr_scores_adv.tolist() + [0])

        mrmr_scores_norm = mrmr_scores_norm[i_sorted_mrmr]
        mrmr_scores_norm_all_clfr += (mrmr_scores_norm.tolist() + [0])
 
        # clean up DISR scores
        disr_scores_norm /= RUNS
        disr_scores_adv /= RUNS
        i_sorted = np.argsort(disr_scores_norm)[::-1]
        adv_sorted = np.argsort(disr_scores_adv)[::-1]
        
        i_sorted_disr = i_sorted[:NBEST]
        adv_sorted_disr = adv_sorted[:NBEST]

        # print("DISR: top 10 features from normal data for ", data_sets[j], ": ", i_sorted_disr)
        # print("DISR: top 10 features from adversarial data for ", data_sets[j], ": ", adv_sorted_disr)

        disr_scores_adv = disr_scores_adv[i_sorted_disr]
        disr_scores_adv_all_clfr+= (disr_scores_adv.tolist() + [0])
        
        disr_scores_norm = disr_scores_norm[i_sorted_disr]
        disr_scores_norm_all_clfr += (disr_scores_norm.tolist() + [0])
       
        # clean up Relief scores
        relief_scores_norm /= RUNS
        relief_scores_adv /= RUNS
        i_sorted = np.argsort(relief_scores_norm)[::-1]
        adv_sorted = np.argsort(relief_scores_adv)[::-1]
       
        i_sorted_relief = i_sorted[:NBEST]
        adv_sorted_relief = adv_sorted[:NBEST]

        # print("Relief: top 10 features from normal data for ", data_sets[j], ": ", i_sorted_relief)
        # print("Relief: top 10 features from adversarial data for ", data_sets[j], ": ", adv_sorted_relief)

        relief_scores_adv = relief_scores_adv[i_sorted_relief]
        relief_scores_adv_all_clfr += (disr_scores_adv.tolist() + [0])
        relief_scores_norm = relief_scores_norm[i_sorted_relief]
        relief_scores_norm_all_clfr += (relief_scores_norm.tolist() + [0])
       
        # clean up Fisher scores
        fisher_scores_norm /= RUNS
        fisher_scores_adv /= RUNS
        i_sorted = np.argsort(fisher_scores_norm)[::-1]
        adv_sorted = np.argsort(fisher_scores_adv)[::-1]
       
        i_sorted_fisher = i_sorted[:NBEST]
        adv_sorted_fisher = adv_sorted[:NBEST]

        # print("Fisher: top 10 features from normal data for ", data_sets[j], ": ", i_sorted_fisher)
        # print("Fisher: top 10 features from adversarial data for ", data_sets[j], ": ", adv_sorted_fisher)
        fisher_scores_adv = fisher_scores_adv[i_sorted_fisher]
        fisher_scores_adv_all_clfr += (fisher_scores_adv.tolist() + [0])

        fisher_scores_norm = fisher_scores_norm[i_sorted_fisher]
        fisher_scores_norm_all_clfr += (fisher_scores_norm.tolist() + [0])
   
    
        #x = [i for i in range(len(i_sorted))]
        n_groups = n_groups + NBEST
        #index += list(range(1,NBEST+2,1))
        #index = np.append(index, np.arange(1, NBEST+2))
        #print(index)
        

        # Matrix for 1-d plots
        norm_feature_matrix_all_clf.append(np.array([i_sorted_mim, i_sorted_mrmr, i_sorted_disr, i_sorted_relief, i_sorted_fisher]))
        adv_feature_matrix_all_clf.append(np.array([adv_sorted_mim, adv_sorted_mrmr, adv_sorted_disr, adv_sorted_relief, adv_sorted_fisher]))

    norm_feature_matrix_all_clf = np.array(norm_feature_matrix_all_clf)
    adv_feature_matrix_all_clf = np.array(adv_feature_matrix_all_clf)
    print("Normal top 10 features for", data_sets[j], '\n', norm_feature_matrix_all_clf)
    print("Adversarial top 10 features for", data_sets[j], '\n', adv_feature_matrix_all_clf)

    np.savez(OUTPUT_FILE, norm_feature_matrix_all_clf = norm_feature_matrix_all_clf, adv_feature_matrix_all_clf = adv_feature_matrix_all_clf)

    #ngroups += len(classifier)-1


    # Generating barchart plots

    index = np.arange(n_groups + len(classifier)) 
    l = [0, 5, 9, 11, 16, 20, 22, 27, 31, 33, 38, 42]
    rep = ['0', 'LR', '10', '0', '1-NN', '10', '0', '5-NN', '10', '0', 'SVM', '10']
    bar_width = 0.35
    opacity = 1.
    norm_fs_scores_list = [mim_scores_norm_all_clfr, mrmr_scores_norm_all_clfr, disr_scores_norm_all_clfr, fisher_scores_norm_all_clfr, relief_scores_norm_all_clfr]
    adv_fs_scores_list = [mim_scores_adv_all_clfr, mrmr_scores_adv_all_clfr, disr_scores_adv_all_clfr, fisher_scores_adv_all_clfr, relief_scores_adv_all_clfr]

    print("norm_fs_scores_list \n", norm_fs_scores_list)
    print("adv_fs_scores_list \n", adv_fs_scores_list)
 
    for f in range(len(fs_algo)):
        fig, ax = plt.subplots()
        rects1 = plt.bar(index, norm_fs_scores_list[f], bar_width, alpha=opacity, label='Benign')
        rects2 = plt.bar(index+bar_width, adv_fs_scores_list[f], bar_width, alpha=opacity, label='Adversarial')
        plt.ylabel(fs_algo[f] + ' Scores')
        plt.legend(prop = {'size': 6})
        plt.tight_layout()
        ax.set_xticks(l)
        ax.set_xticklabels(rep)
        plt.rcParams["figure.figsize"] = (5,2)
        plt.savefig(''.join(['outputs/barcharts/', attack, '/', fs_algo[f], '_', data_sets[j], '_all_clfr.pdf']))

    # ------------------------------------------------------------------------------------------------
    # create plot: mim scores
    #n_groups = len(i_sorted) #n_groups = len(x)
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 1.
 
    # # rects1 = plt.bar(index, mim_scores_norm, bar_width, alpha=opacity, label='Benign')
    # # rects2 = plt.bar(index+bar_width, mim_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    
    # rects1 = plt.bar(index, mim_scores_norm_all_clfr, bar_width, alpha=opacity, label='Benign')
    # rects2 = plt.bar(index+bar_width, mim_scores_adv_all_clfr, bar_width, alpha=opacity, label='Adversarial')
    # #plt.xlabel('Top 10 features for all classifiers')
    # plt.ylabel('MIM Score')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(''.join(['outputs/barcharts/', attack, '/MIM_', data_sets[j], '_all_clfr_.pdf']))
    # #plt.savefig(''.join(['outputs/', classifier[i], '/barcharts/', 'MIM/barcharts_', data_sets[j], '_', post, '.pdf']))
    # #plt.close()
   
    
    # # ------------------------------------------------------------------------------------------------
    # # create plot: mrmr scores
    # #n_groups = len(x)
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 1.
 
    # # rects1 = plt.bar(index, mrmr_scores_norm, bar_width, alpha=opacity, label='Benign')
    # # rects2 = plt.bar(index+bar_width, mrmr_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    # rects1 = plt.bar(index, mrmr_scores_norm_all_clfr, bar_width, alpha=opacity, label='Benign')
    # rects2 = plt.bar(index+bar_width, mrmr_scores_adv_all_clfr, bar_width, alpha=opacity, label='Adversarial')
    # #plt.xlabel('Top 10 features for all classifiers')
    # plt.ylabel('mRMR Score')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(''.join(['outputs/barcharts/', attack, '/MRMR_', data_sets[j], '_all_clfr_.pdf']))
    # #plt.savefig(''.join(['outputs/', classifier[i], '/barcharts/', 'MRMR/barcharts_', data_sets[j], '_', post, '.pdf']))
    # #plt.close()
       
    # # create plot: disr scores
    # #n_groups = len(x)
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 1.
 
    # # rects1 = plt.bar(index, disr_scores_norm, bar_width, alpha=opacity, label='Benign')
    # # rects2 = plt.bar(index+bar_width, disr_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    # rects1 = plt.bar(index, disr_scores_norm_all_clfr, bar_width, alpha=opacity, label='Benign')
    # rects2 = plt.bar(index+bar_width, disr_scores_adv_all_clfr, bar_width, alpha=opacity, label='Adversarial')
    # #plt.xlabel('Top 10 features for all classifiers')
    # plt.ylabel('DISR Score')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(''.join(['outputs/barcharts/', attack, '/DISR_', data_sets[j], '_all_clfr_.pdf']))
    # #plt.savefig(''.join(['outputs/', classifier[i], '/', attack, '/barcharts/underlying_classifier/disr_', data_sets[j], '_', post, '.pdf']))
    # #plt.savefig(''.join(['outputs/', classifier[i], '/', attack, '/barcharts/', 'DISR/barcharts_', data_sets[j], '_', post, '.pdf']))
    # #plt.close()
       
    # # create plot: relief scores
    # #n_groups = len(x)
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 1.
 
    # # rects1 = plt.bar(index, relief_scores_norm, bar_width, alpha=opacity, label='Benign')
    # # rects2 = plt.bar(index+bar_width, relief_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    # rects1 = plt.bar(index, relief_scores_norm_all_clfr, bar_width, alpha=opacity, label='Benign')
    # rects2 = plt.bar(index+bar_width, relief_scores_adv_all_clfr, bar_width, alpha=opacity, label='Adversarial')
    # #plt.xlabel('Top 10 features for all classifiers')
    # plt.ylabel('Relief Score')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(''.join(['outputs/barcharts/', attack, '/Relief_', data_sets[j], '_all_clfr_.pdf']))
    # #plt.savefig(''.join(['outputs/', classifier[i], '/barcharts/', 'Relief/barcharts_', data_sets[j], '_', post, '.pdf']))
    # #plt.close()
       
    # # create plot: fisher scores
    # #n_groups = len(x)
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 1.
 
    # # rects1 = plt.bar(index, fisher_scores_norm, bar_width, alpha=opacity, label='Benign')
    # # rects2 = plt.bar(index+bar_width, fisher_scores_adv, bar_width, alpha=opacity, label='Adversarial')
    # rects1 = plt.bar(index, fisher_scores_norm_all_clfr, bar_width, alpha=opacity, label='Benign')
    # rects2 = plt.bar(index+bar_width, fisher_scores_adv_all_clfr, bar_width, alpha=opacity, label='Adversarial')
    # #plt.xlabel('Top 10 features for all classifiers')
    # plt.ylabel('Fisher Scores')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(''.join(['outputs/barcharts/', attack, '/Fisher_', data_sets[j], '_all_clfr_.pdf']))
    # #plt.close()

