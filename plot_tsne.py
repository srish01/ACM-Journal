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
import numpy as np
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
plt.style.use('bmh')

# attack type. to do: add in svc dataset once the code finishes on the server
ATTACK_TYPES = ['fgsm', 'deepfool']
data_sets = [file[:-4] for file in os.listdir('data/')]

for attack in ATTACK_TYPES: 
    for i in range(len(data_sets)): 
        # set the random seed for reproducibility 
        np.random.seed(1)
        
        # load the data and split the data by benign and adversarial 
        D = np.load(''.join(['outputs/adversarial_data_', data_sets[i], '_', attack, '.npz']))
        Xtr, ytr, Xadv, yadv = D['Xtr'], D['ytr'], D['Xadv'], D['yadv']
    
        # just print something so we know where we're at 
        print(data_sets[i])
        print('  > Benign: ' + str(len(ytr)))
        print('  > Adversarial: ' + str(len(yadv)))

        # perform TNSE
        X = np.concatenate((Xtr, Xadv))
        y = np.zeros((len(X)))
        y[len(ytr):] = 1
        Xe = TSNE(n_components=2).fit_transform(X)

        # plot the data, save it, close it and move on. 
        plt.figure()
        plt.scatter(Xe[y==0, 0], Xe[y==0, 1], label='Benign')
        plt.scatter(Xe[y==1, 0], Xe[y==1, 1], label='Adversarial')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(''.join(['outputs/adversarial_data_', data_sets[i], '_', attack, '.pdf']))
        plt.close()
