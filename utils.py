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


def jaccard(a, b):
    """
    Jaccard index
    """
    a, b = set(a), set(b)
    return 1.*len(a.intersection(b))/len(a.union(b))

def kuncheva(a, b, K): 
    """
    Kuncheva index 
    """
    a, b = set(a), set(b)
    r = 1.*len(a.intersection(b))
    k = 1.*len(a)
    return 1.*(r*K - k**2)/(k*(K - k)) 


def total_consistency(sel_feats, n_features): 
    """
    Measure the Jaccard and Kuncheva stability for sets of feature subsets that
    were collected from cross fold validation. 
    """
    n = len(sel_feats)

    ck, cj = 0., 0.
    k = 0
    for i in range(n): 
        for j in range(n): 
            if j > i: 
                cj += jaccard(sel_feats[i], sel_feats[j])
                ck += kuncheva(sel_feats[i], sel_feats[j], n_features)
                k += 1
    return cj/k, ck/k 