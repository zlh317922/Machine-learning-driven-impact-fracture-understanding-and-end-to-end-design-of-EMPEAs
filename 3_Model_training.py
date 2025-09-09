# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:20:02 2022

@author: zlh
"""

from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold 
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm 
import time
import os
start_time = time.time()

filepath = r''
resultpath =  r''
data = pd.read_excel(filepath, sheet_name='Sheet')  

Feature_data = data.iloc[:, :].values
MPS_data =  data.iloc[:, :].values
Feature_names = data.iloc[:,:].tolist()
Features_data = np.array(Feature_data)
MPS_data = np.array(MPS_data)
X = Features_data  
Y = MPS_data

####################################################特征降维###################################
selected_features = [ ]  
X = X[:,selected_features]
selected_features_name = []
for i in range(len(selected_features)):
    selected_features_name.append(Feature_names[selected_features[i]])
print('选择特征为：', selected_features_name)

####################################################最优模型###################################



####################################################测试###################################
Model_23F = ['23F_grid_SVR','23F_random_SVR','23F_bayes_SVR','23F_grid_RF','23F_bayes_RF',
              '23F_random_XG','23F_bayes_XG','23F_bayes_AdaSVR','23F_bayes_GPR','23F_grid_KNN']  
Model_5F = ['5F_grid_SVR','5F_random_SVR','5F_bayes_SVR','5F_grid_RF','5F_bayes_RF',
              '5F_random_XG','5F_bayes_XG','5F_bayes_AdaSVR','5F_grid_KNN']  
Model_23F = ['23F_grid_SVR','23F_grid_RF','23F_random_XG','23F_bayes_AdaSVR','23F_bayes_GPR','23F_grid_KNN']  
Model_23F = ['23F_grid_KNN'] 
os.makedirs(fr'{resultpath}\HyperParameter_optimization\picture', exist_ok=True)
os.makedirs(fr'{resultpath}\HyperParameter_optimization\detailed_results', exist_ok=True)

# 为每个模型创建单独的图片文件夹
for model_name in Model_23F:
    model_pic_dir = os.path.join(resultpath, 'HyperParameter_optimization', 'picture', model_name)
    os.makedirs(model_pic_dir, exist_ok=True)

# 存储最终平均结果
results = []
# 创建一个Excel writer对象用于保存多sheet的详细结果
detailed_results_path = fr'{resultpath}\HyperParameter_optimization\detailed_results\all_folds_results.xlsx'
with pd.ExcelWriter(detailed_results_path, engine='openpyxl') as writer:
    for model_name in tqdm(Model_23F, desc="Evaluating models"):
        # 加载模型
        clf = load(fr'{resultpath}\HyperParameter_optimization\{model_name}.m')
        clf = KNeighborsRegressor(algorithm='ball_tree', n_neighbors=1, weights='uniform')

        # 获取当前模型的图片保存目录
        model_pic_dir = os.path.join(resultpath, 'HyperParameter_optimization','picture', model_name)
        
        # 初始化存储列表
        all_train_r2 = []
        all_test_r2 = []
        all_train_rmse = []
        all_test_rmse = []
        model_detailed_results = []  # 存储当前模型的详细结果
        
        # 将连续的Y值分箱成5个类别（与5折对应）
        def create_stratification_bins(y, n_splits=5):  
        # 使用分位数分箱确保每个类别有大致相同数量的样本
            return pd.qcut(y, n_splits, labels=False, duplicates='drop')
        
        # 进行100次重复的五折交叉验证
        for repeat in range(100):
            # 为当前重复创建分层标签
            stratification_bins = create_stratification_bins(Y, n_splits=5)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)

            repeat_train_r2 = []
            repeat_test_r2 = []
            repeat_train_rmse = []
            repeat_test_rmse = []
            
            # 使用分层标签进行交叉验证
            for fold, (train_index, test_index) in enumerate(skf.split(X, stratification_bins)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                
                # 训练和预测
                clf.fit(X_train, y_train)
                train_pred = clf.predict(X_train)
                test_pred = clf.predict(X_test)
                
                # 计算指标
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # 保存当前折结果
                repeat_train_r2.append(train_r2)
                repeat_test_r2.append(test_r2)
                repeat_train_rmse.append(train_rmse)
                repeat_test_rmse.append(test_rmse)
                
                # 保存详细结果
                model_detailed_results.append({
                    'Repeat': repeat + 1,
                    'Fold': fold + 1,
                    'Type': 'Train',
                    'R2': train_r2,
                    'RMSE': train_rmse
                })
                model_detailed_results.append({
                    'Repeat': repeat + 1,
                    'Fold': fold + 1,
                    'Type': 'Test',
                    'R2': test_r2,
                    'RMSE': test_rmse
                })
                
                # 保存前10次重复的散点图
                if repeat < 10:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(y_train, train_pred, color='lightblue', alpha=0.7, 
                               edgecolor='dodgerblue', label=f'Train R² = {train_r2:.3f}')
                    plt.scatter(y_test, test_pred, color='lightcoral', alpha=0.7,
                               edgecolor='red', label=f'Test   R² = {test_r2:.3f}')
                    plt.plot([0.7, 2.1], [0.7, 2.1], '--', color='dimgray', linewidth=1.5)
                    
                    plt.xlim(0.7, 2.1)
                    plt.ylim(0.7, 2.1)
                    ticks = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
                    plt.xticks(ticks)
                    plt.yticks(ticks)
                    plt.gca().set_aspect('equal')
                    plt.xlabel('Actual value/mm', fontsize=24)
                    plt.ylabel('Predicted value/mm', fontsize=24)
                    
                    ax = plt.gca()
                    ax.tick_params(axis='both', which='both', direction='in',
                                  labelsize=22, bottom=True, left=True,
                                  top=False, right=False)
                    
                    plt.legend(loc='upper left', frameon=False, fontsize=15)
                    plt.rcParams['font.family'] = 'Times New Roman'
                    plt.savefig(
                        os.path.join(model_pic_dir, f'repeat_{repeat+1}_fold_{fold+1}.jpg'), 
                        dpi=600, 
                        bbox_inches='tight'
                    )
                    plt.close()
            
            # 保存当前重复的平均指标
            all_train_r2.append(np.mean(repeat_train_r2))
            all_test_r2.append(np.mean(repeat_test_r2))
            all_train_rmse.append(np.mean(repeat_train_rmse))
            all_test_rmse.append(np.mean(repeat_test_rmse))
        
        # 计算100次重复的平均指标
        avg_train_r2 = np.mean(all_train_r2)
        avg_test_r2 = np.mean(all_test_r2)
        avg_train_rmse = np.mean(all_train_rmse)
        avg_test_rmse = np.mean(all_test_rmse)
        
        # 打印结果
        print(f"\n{model_name}")
        print(f"100次重复五折交叉验证平均训练集 R²: {avg_train_r2:.4f}")
        print(f"100次重复五折交叉验证平均测试集 R²: {avg_test_r2:.4f}")
        print(f"100次重复五折交叉验证平均训练集 RMSE: {avg_train_rmse:.4f}")
        print(f"100次重复五折交叉验证平均测试集 RMSE: {avg_test_rmse:.4f}")
        
        # 保存平均结果
        results.append({
            '模型': model_name,
            '训练集 R2': avg_train_r2,
            '测试集 R2': avg_test_r2,
            '训练集 RMSE': avg_train_rmse,
            '测试集 RMSE': avg_test_rmse
        })
        
        # 将当前模型的详细结果保存到单独的工作表
        df_model_detailed = pd.DataFrame(model_detailed_results)
        df_model_detailed.to_excel(writer, sheet_name=model_name[:31], index=False)  # 工作表名最多31字符

# 保存平均结果到Excel
df_results = pd.DataFrame(results)
df_results.to_excel(fr'{resultpath}\HyperParameter_optimization\model_5F_evaluation.xlsx', index=False)
print("\n所有模型评估完成！结果已保存。")
print(f"详细结果已保存至: {detailed_results_path}")
