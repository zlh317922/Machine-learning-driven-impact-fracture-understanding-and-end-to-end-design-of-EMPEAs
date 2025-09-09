# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:42:11 2025

@author: zlh
"""


import numpy as np
import pandas as pd 
from sklearn.utils import shuffle 
from sklearn.model_selection import LeaveOneOut
from genetic_selection import GeneticSelectionCV 
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from joblib import load
import time
start_time = time.time()

filepath = r' '
resultpath =  r' '
data = pd.read_excel(filepath, sheet_name=' ')  
Feature_data = data.iloc[:, :].values
MPS_data =  data.iloc[:, :].values
Feature_names = data.iloc[:,:].tolist()
Features_data = np.array(Feature_data)
MPS_data = np.array(MPS_data)
X = Features_data  
Y = MPS_data
Features = Feature_names
model = load(fr'model.m') 

Selected_Features_SVR_100GA = []
for j in range(100):
    shuffled_data, shuffled_label = shuffle(X, Y)
    X = shuffled_data
    Y = shuffled_label

    estimator = model
    selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=2,
        scoring="r2",
        max_features=12,
        n_population=100,
        crossover_proba=0.75,
        mutation_proba=0.05,
        n_generations=100,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=20,
        caching=True,
        n_jobs=-1,
    )
    selector = selector.fit(X, Y)
    print("SVR 第", j + 1, "轮")
    selected_features = [index for index in range(len(selector.support_)) if selector.support_[index] == 1]
    Selected_Features_SVR_100GA.extend(selected_features)
    
Selected_Features_SVR_100GA.sort()
from collections import Counter
result = Counter(Selected_Features_SVR_100GA)
df = pd.DataFrame.from_dict(result, orient='index')
df.columns = ['Count']
with pd.ExcelWriter(r' ') as writer:
     df.to_excel(writer,sheet_name= 'sheet')