#!/usr/bin/env python
# -*- coding: utf-8 -*-
#author: jackft
#date: 2016-07-16

from scipy import polyfit, poly1d
import numpy as np
from itertools import tee
from collections import Counter, defaultdict
import pandas as pd
import csv

import warnings
warnings.simplefilter('ignore', np.RankWarning)

def main():
    f = open('../data/test.csv', 'r')
    csv_reader = csv.DictReader(f)
    rows = [row for row in csv_reader]
    sequences = {int(row['Id']): [int(s) for s in row['Sequence'].split(',')] for row in rows}

    A0 = list(filter(strictly_increasing, sequences.values()))
    A1 = list(filter(non_decreasing, sequences.values()))
    A2 = list(filter(strictly_decreasing, sequences.values()))
    A3 = list(filter(non_increasing, sequences.values()))
    A4 = list(filter(non_monotonic, sequences.values()))

    lines = []

    #strictly_increasing
    fs0 = [poly(n) for n in range(1,6 + 1)]
    fs0 = fs0 ++ [ngram(A0, n=n,threshold=1e-05) for n in range(1, 4 + 1)]
    fs0 = fs0 ++ [mode]

    for A in A0:
        predictions(lines, fs0, A)

    #non_increasing
    fs1 = [poly(n) for n in range(1,6 + 1)]
    fs1 = fs1 ++ [ngram(A1, n=n,threshold=1e-04) for n in range(1, 4 + 1)]
    fs1 = fs1 ++ [mode]

    for A in A1:
        predictions(lines, fs1, A)

    #strictly_decreasing
    fs2 = [poly(n) for n in range(1,6 + 1)]
    fs2 = fs2 ++ [ngram(A2, n=n,threshold=1e-3) for n in range(1, 4 + 1)]
    fs2 = fs2 ++ [mode]

    for A in A2:
        predictions(lines, fs2, A)

    #non_decreasing
    fs3 = [poly(n) for n in range(1,6 + 1)]
    fs3 = fs3 ++ [ngram(A3, n=n,threshold=1e-03) for n in range(1, 4 + 1)]
    fs3 = fs3 ++ [mode]

    for A in A3:
        predictions(lines, fs3, A)

    #non-monotonic
    fs4 = [poly(n) for n in range(1,6 + 1)]
    fs4 = fs4 ++ [ngram(A4, n=n,threshold=1e-03) for n in range(1, 4 + 1)]
    fs4 = fs4 ++ [mode]

    for A in A4:
        predictions(lines, fs4, A)

    create(lines)


###############################################################################
# categories
###############################################################################

def strictly_increasing(A):
    return all(x < y for x, y in zip(A, A[1:]))

def non_decreasing(A):
    return all(x <= y for x, y in zip(A, A[1:])) and not all(x < y for x, y in zip(A, A[1:]))

def strictly_decreasing(A):
    return all(x > y for x, y in zip(A, A[1:]))

def non_increasing(A):
    return all(x >= y for x, y in zip(A, A[1:])) and not all(x > y for x, y in zip(A, A[1:]))

def non_monotonic(A):
    return not any([strictly_increasing(A), strictly_decreasing(A), non_increasing(A), non_decreasing(A)])

###############################################################################
# predictors
###############################################################################

def poly(n):
    def predict(A):
        if len(A) > min(3, n):
            xtrain = np.linspace(1,len(A),len(A))
            ytrain = A
            xtest = len(A) + 1
            f = poly1d(polyfit(xtrain, ytrain, n))
            return f(xtest).round() if all(f(xtrain).round() == ytrain) else None
    return predict

def ngram(As, threshold=0.001, n=1, prior=False):
    d = max_next(nwise_dict(As, n=n, prior=prior))
    def pred(A):
        return d[tuple(A[-n:])][0] if tuple(A[-n:]) in d and d[tuple(A[-n:])][1] >= threshold else None
    return pred

mode = lambda A: max(map(lambda val: (A.count(val), val), set(A)))[1]

succ = lambda A: A[-1] + 1

###############################################################################
# predictor helpers
###############################################################################

def nwise(iterable, n):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    iterables = tee(iterable, n)
    for i, it in enumerate(iterables):
        for j in range(i):
            next(it,None)
    return zip(*iterables)

def counts(As, n=1, end=None, normalized = False):
    As = list(map(lambda A: nwise(A,n), As))
    elems = [e for A in As for e in A[-end:]] if end else [e for A in As for e in A]
    hist = Counter(elems)
    if normalized:
        N=sum(hist.values())
        return {e: cnt/N for e, cnt in hist.items()}
    return hist



def nwise_dict(As,n=1, prior=False):
    cnts = counts(As, n=n+1, normalized=True)
    X = list(zip(*cnts.keys()))
    keys = list(zip(*X[:-1]))
    values = X[-1]

    weights = cnts.values()
    wprior = counts(As, normalized=True)

    d = defaultdict(lambda : defaultdict(int))
    for key, value, w in zip(keys, values, weights):
        d[key][value] = w * (wprior[value,] if prior else 1)
    return d

def max_next(nwise_d):
    return {key: max(value.items(), key=lambda x: x[1]) for key, value in nwise_d.items()}

###############################################################################
# utilities
###############################################################################

def predictions(lines, fs, A):
    idx, A = A
    for f in fs:
        y_hat = f(A)
        if y_hat:
            lines.append({"Id": idx, "Last": y_hat})

def create(lines):
    df = pd.DataFrame(lines)
    df.to_csv("submission1.csv",header=True, index=False)

###############################################################################
# script
###############################################################################

if __name__ == '__main__':
    main()
