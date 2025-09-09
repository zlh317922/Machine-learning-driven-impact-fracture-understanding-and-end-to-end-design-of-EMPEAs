# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 10:59:53 2025

@author: zlh
"""


from joblib import dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import RandomizedSearchCV #随机搜索
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import r2_score, mean_squared_error, make_scorer,explained_variance_score
from Hyperparameter.Search_SVR import GridSearchCV_SVR, RandomizedSearchCV_SVR, BayesSearchCV_SVR
from Hyperparameter.Search_AdaSVR import BayesSearchCV_AdaSVR
from Hyperparameter.Search_RF import GridSearchCV_RF, RandomizedSearchCV_RF, BayesSearchCV_RF
from Hyperparameter.Search_XG import RandomizedSearchCV_XG, BayesSearchCV_XG
from Hyperparameter.Search_GPR import GridSearchCV_GPR, RandomizedSearchCV_GPR, BayesSearchCV_GPR
from Hyperparameter.Search_KNN import GridSearchCV_KNN, RandomizedSearchCV_KNN, BayesSearchCV_KNN
import time
import os
start_time = time.time()

filepath = ''
resultpath =  ''
data = pd.read_excel(filepath, sheet_name='')  
Feature_data = data.iloc[:,:].values
MPS_data =  data.iloc[:, :].values
Feature_names = data.iloc[:,:].tolist()
Features_data = np.array(Feature_data)
MPS_data = np.array(MPS_data)
X = Features_data  
Y = MPS_data

os.environ["LOKY_ENCODING"] = "utf-8"  # 解决中文路径问题
os.environ['JOBLIB_TEMP_FOLDER'] = r'C:\temp_joblib'  # 设置临时目录


####################################################特征降维###################################
selected_features = [] 
selected_features_name = []
for i in range(len(selected_features)):
    selected_features_name.append(Feature_names[selected_features[i]])
print('选择特征为：', selected_features_name)

####################################################超参数优化+保存模型###################################
with open(fr"{resultpath}\HyperParameter_optimization\5F_optimization", "w") as file:
    # 网格搜索
    best_model_grid, best_params_grid, best_score_grid = GridSearchCV_SVR(X, Y)             
    best_model_grid.fit(X, Y)
    Y_pre_grid = best_model_grid.predict(X)
    file.write("网格搜索 SVR\n")
    file.write(f"最优参数: {best_params_grid}\n")
    file.write(f"最优分数: {best_score_grid}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_grid)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_grid))}\n\n")
    print("网格搜索决定系数 R2:", r2_score(Y, Y_pre_grid))
    print("网格搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_grid)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_grid, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('GridSearchCV_SVR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    
    # # 随机搜索
    best_model_random, best_params_random, best_score_random = RandomizedSearchCV_SVR(X, Y)    
    best_model_random.fit(X, Y)
    Y_pre_random = best_model_random.predict(X)
    file.write("随机搜索 SVR\n")
    file.write(f"最优参数: {best_params_random}\n")
    file.write(f"最优分数: {best_score_random}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_random)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_random))}\n\n")
    print("随机搜索决定系数 R2:", r2_score(Y, Y_pre_random))
    print("随机搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_random)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_random, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('RandomizedSearchCV_SVR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")

    # 贝叶斯优化
    best_model_bayes, best_params_bayes, best_score_bayes = BayesSearchCV_SVR(X, Y)         
    best_model_bayes.fit(X, Y)
    Y_pre_bayes = best_model_bayes.predict(X)
    file.write("贝叶斯优化 SVR\n")
    file.write(f"最优参数: {best_params_bayes}\n")
    file.write(f"最优分数: {best_score_bayes}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_bayes)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_bayes))}\n\n")
    print("贝叶斯优化决定系数 R2:", r2_score(Y, Y_pre_bayes))
    print("贝叶斯优化均方根误差 RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_bayes)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_bayes, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('BayesSearchCV_SVR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    plt.show()
    
    dump(best_model_grid, fr'{resultpath}\HyperParameter_optimization\5F_grid_SVR.m')          
    dump(best_model_random, fr'{resultpath}\HyperParameter_optimization\5F_random_SVR.m')
    dump(best_model_bayes, fr'{resultpath}\HyperParameter_optimization\5F_bayes_SVR.m')  

    # # 网格搜索
    best_model_grid, best_params_grid, best_score_grid = GridSearchCV_RF(X, Y)             
    best_model_grid.fit(X, Y)
    Y_pre_grid = best_model_grid.predict(X)
    file.write("网格搜索 RF\n")
    file.write(f"最优参数: {best_params_grid}\n")
    file.write(f"最优分数: {best_score_grid}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_grid)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_grid))}\n\n")
    print("网格搜索决定系数 R2:", r2_score(Y, Y_pre_grid))
    print("网格搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_grid)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_grid, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('GridSearchCV_RF')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    
    # # 随机搜索
    best_model_random, best_params_random, best_score_random = RandomizedSearchCV_RF(X, Y)    
    best_model_random.fit(X, Y)
    Y_pre_random = best_model_random.predict(X)
    file.write("随机搜索 RF\n")
    file.write(f"最优参数: {best_params_random}\n")
    file.write(f"最优分数: {best_score_random}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_random)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_random))}\n\n")
    print("随机搜索决定系数 R2:", r2_score(Y, Y_pre_random))
    print("随机搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_random)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_random, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('RandomizedSearchCV_RF')
    plt.xlabel("True value")
    plt.ylabel("Predict value")

    # 贝叶斯优化
    best_model_bayes, best_params_bayes, best_score_bayes = BayesSearchCV_RF(X, Y)         
    best_model_bayes.fit(X, Y)
    Y_pre_bayes = best_model_bayes.predict(X)
    file.write("贝叶斯优化 RF\n")
    file.write(f"最优参数: {best_params_bayes}\n")
    file.write(f"最优分数: {best_score_bayes}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_bayes)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_bayes))}\n\n")
    print("贝叶斯优化决定系数 R2:", r2_score(Y, Y_pre_bayes))
    print("贝叶斯优化均方根误差 RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_bayes)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_bayes, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('BayesSearchCV_RF')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    plt.show()
    
    dump(best_model_grid, fr'{resultpath}\HyperParameter_optimization\5F_grid_RF.m')          
    dump(best_model_random, fr'{resultpath}\HyperParameter_optimization\5F_random_RF.m')
    dump(best_model_bayes, fr'{resultpath}\HyperParameter_optimization\5F_bayes_RF.m')  
    
    
    # 随机搜索
    best_model_random, best_params_random, best_score_random = RandomizedSearchCV_XG(X, Y)    
    best_model_random.fit(X, Y)
    Y_pre_random = best_model_random.predict(X)
    file.write("随机搜索 XG\n")
    file.write(f"最优参数: {best_params_random}\n")
    file.write(f"最优分数: {best_score_random}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_random)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_random))}\n\n")
    print("随机搜索决定系数 R2:", r2_score(Y, Y_pre_random))
    print("随机搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_random)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_random, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('RandomizedSearchCV_XG')
    plt.xlabel("True value")
    plt.ylabel("Predict value")

    # 贝叶斯优化
    best_model_bayes, best_params_bayes, best_score_bayes = BayesSearchCV_XG(X, Y)         
    best_model_bayes.fit(X, Y)
    Y_pre_bayes = best_model_bayes.predict(X)
    file.write("贝叶斯优化 XG\n")
    file.write(f"最优参数: {best_params_bayes}\n")
    file.write(f"最优分数: {best_score_bayes}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_bayes)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_bayes))}\n\n")
    print("贝叶斯优化决定系数 R2:", r2_score(Y, Y_pre_bayes))
    print("贝叶斯优化均方根误差 RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_bayes)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_bayes, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('BayesSearchCV_XG')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    plt.show()
         
    dump(best_model_random, fr'{resultpath}\HyperParameter_optimization\5F_random_XG.m')
    dump(best_model_bayes, fr'{resultpath}\HyperParameter_optimization\5F_bayes_XG.m')  
    
    # 贝叶斯优化
    best_model_bayes, best_params_bayes, best_score_bayes = BayesSearchCV_AdaSVR(X, Y)         
    best_model_bayes.fit(X, Y)
    Y_pre_bayes = best_model_bayes.predict(X)
    file.write("贝叶斯优化 AdaSVR\n")
    file.write(f"最优参数: {best_params_bayes}\n")
    file.write(f"最优分数: {best_score_bayes}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_bayes)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_bayes))}\n\n")
    print("贝叶斯优化决定系数 R2:", r2_score(Y, Y_pre_bayes))
    print("贝叶斯优化均方根误差 RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_bayes)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_bayes, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('BayesSearchCV_AdaSVR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    plt.show()
         
    dump(best_model_bayes, fr'{resultpath}\HyperParameter_optimization\5F_bayes_AdaSVR.m')  
    
    # 网格搜索
    best_model_grid, best_params_grid, best_score_grid = GridSearchCV_GPR(X, Y)             
    best_model_grid.fit(X, Y)
    Y_pre_grid = best_model_grid.predict(X)
    file.write("网格搜索 GPR\n")
    file.write(f"最优参数: {best_params_grid}\n")
    file.write(f"最优分数: {best_score_grid}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_grid)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_grid))}\n\n")
    print("网格搜索决定系数 R2:", r2_score(Y, Y_pre_grid))
    print("网格搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_grid)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_grid, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('GridSearchCV_GPR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    
    # # 随机搜索
    best_model_random, best_params_random, best_score_random = RandomizedSearchCV_GPR(X, Y)    
    best_model_random.fit(X, Y)
    Y_pre_random = best_model_random.predict(X)
    file.write("随机搜索 GPR\n")
    file.write(f"最优参数: {best_params_random}\n")
    file.write(f"最优分数: {best_score_random}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_random)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_random))}\n\n")
    print("随机搜索决定系数 R2:", r2_score(Y, Y_pre_random))
    print("随机搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_random)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_random, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('RandomizedSearchCV_GPR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")

    # 贝叶斯优化
    best_model_bayes, best_params_bayes, best_score_bayes = BayesSearchCV_GPR(X, Y)         
    best_model_bayes.fit(X, Y)
    Y_pre_bayes = best_model_bayes.predict(X)
    file.write("贝叶斯优化 GPR\n")
    file.write(f"最优参数: {best_params_bayes}\n")
    file.write(f"最优分数: {best_score_bayes}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_bayes)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_bayes))}\n\n")
    print("贝叶斯优化决定系数 R2:", r2_score(Y, Y_pre_bayes))
    print("贝叶斯优化均方根误差 RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_bayes)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_bayes, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('BayesSearchCV_GPR')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    plt.show()
    
    dump(best_model_grid, fr'{resultpath}\HyperParameter_optimization\5F_grid_GPR.m')          
    dump(best_model_random, fr'{resultpath}\HyperParameter_optimization\5F_random_GPR.m')
    dump(best_model_bayes, fr'{resultpath}\HyperParameter_optimization\5F_bayes_GPR.m')  
    
    if not os.path.exists(r'C:\temp_joblib'):
        os.makedirs(r'C:\temp_joblib')
    # 网格搜索
    best_model_grid, best_params_grid, best_score_grid = GridSearchCV_KNN(X, Y)             
    best_model_grid.fit(X, Y)
    Y_pre_grid = best_model_grid.predict(X)
    file.write("网格搜索 KNN\n")
    file.write(f"最优参数: {best_params_grid}\n")
    file.write(f"最优分数: {best_score_grid}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_grid)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_grid))}\n\n")
    print("网格搜索决定系数 R2:", r2_score(Y, Y_pre_grid))
    print("网格搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_grid)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_grid, color='blue')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.gca().set_aspect('equal')
    plt.title('GridSearchCV_KNN')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    
    # 随机搜索
    best_model_random, best_params_random, best_score_random = RandomizedSearchCV_KNN(X, Y)    
    best_model_random.fit(X, Y)
    Y_pre_random = best_model_random.predict(X)
    file.write("随机搜索 KNN\n")
    file.write(f"最优参数: {best_params_random}\n")
    file.write(f"最优分数: {best_score_random}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_random)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_random))}\n\n")
    print("随机搜索决定系数 R2:", r2_score(Y, Y_pre_random))
    print("随机搜索均方根误差RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_random)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_random, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('RandomizedSearchCV_KNN')
    plt.xlabel("True value")
    plt.ylabel("Predict value")

    # 贝叶斯优化
    best_model_bayes, best_params_bayes, best_score_bayes = BayesSearchCV_KNN(X, Y)         
    best_model_bayes.fit(X, Y)
    Y_pre_bayes = best_model_bayes.predict(X)
    file.write("贝叶斯优化 KNN\n")
    file.write(f"最优参数: {best_params_bayes}\n")
    file.write(f"最优分数: {best_score_bayes}\n")
    file.write(f"决定系数 R2: {r2_score(Y, Y_pre_bayes)}\n")
    file.write(f"均方根误差 RMSE: {np.sqrt(mean_squared_error(Y, Y_pre_bayes))}\n\n")
    print("贝叶斯优化决定系数 R2:", r2_score(Y, Y_pre_bayes))
    print("贝叶斯优化均方根误差 RMSE:", np.sqrt(mean_squared_error(Y, Y_pre_bayes)))
    print()
    plt.figure()
    plt.scatter(Y, Y_pre_bayes, color='blue')
    plt.xlim(0.6,2.0)
    plt.ylim(0.6,2.0)
    plt.gca().set_aspect('equal')
    plt.title('BayesSearchCV_KNN')
    plt.xlabel("True value")
    plt.ylabel("Predict value")
    plt.show()
     
    dump(best_model_grid, fr'{resultpath}\HyperParameter_optimization\5F_grid_KNN.m')          
    dump(best_model_random, fr'{resultpath}\HyperParameter_optimization\5F_random_KNN.m')
    dump(best_model_bayes, fr'{resultpath}\HyperParameter_optimization\5F_bayes_KNN.m')  