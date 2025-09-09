# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:51:03 20222
@author: zlh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import os


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

###################################################归一化###################################
# X最值归一化
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) 
X_minmax = min_max_scaler.fit_transform(X)
X = X_minmax

# Y最值归一化
min_val = np.min(Y)
max_val = np.max(Y)
normalized_vector = (Y - min_val) / (max_val - min_val)
Y = normalized_vector  
##################################################归一化###################################


# ##################################################PCC相关系数###################################
PCC_data = np.array(np.column_stack((Feature_data, MPS_data)), dtype=float)
pcc_matrix = np.corrcoef(PCC_data.T)
mask = np.triu(np.ones_like(pcc_matrix, dtype=bool), k=1)
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 25})
fig, ax = plt.subplots(figsize=(18, 18))

heatmap = sns.heatmap(
    pcc_matrix,
    annot=True,
    fmt=".2f",
    vmax=1,
    vmin=-1,
    xticklabels=True,
    yticklabels=True,
    square=True,
    cmap="RdBu_r",
    mask=mask,  
    annot_kws={"size": 15} , # 调整注释字体大小
    cbar_kws={
        "shrink": 0.8,  # 调整颜色条长度
        "aspect": 30,   # 调整颜色条粗细
        "pad": 0.02     # 调整颜色条与热图的间距
    }
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)  
labels = ['v', 'm', 'UTHC', 'THC', 'ρ', 'ΔHmix', 'ΔSmix', 'Ω', 'SHC', 'ΔSHC', 'TC', 'ΔTC', 'TE', 'ΔTE', 
          'R', 'ΔR', 'χ', 'Δχ', 'VEC', 'ΔVEC', 'E', 'ΔE', 'G', 'ΔG', 'K', 'ΔK', 'PR', 'ΔPR', 'MPS']
ax.set_xticks(np.arange(len(labels)) + 0.5)  # 将刻度设置在单元格中心
ax.set_yticks(np.arange(len(labels)) + 0.5)
ax.set_xticklabels(
    labels, 
    fontsize=15, 
    rotation=45, 
    ha='right', 
    rotation_mode='anchor'  # 旋转时以锚点对齐
)
ax.set_yticklabels(labels, fontsize=15)

plt.tight_layout()
plt.savefig(os.path.join(resultpath, 'PCC-new.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# 筛选相关系数大于 0.95 的特征对
high_corr_pairs = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):  # 只检查上三角部分
        if 0.95<abs(pcc_matrix[i, j]) :
            high_corr_pairs.append((labels[i], labels[j], pcc_matrix[i, j]))
            high_corr_pairs.append((i, j))  # 保存索引对
# 提取所有涉及的特征索引           
high_corr_indices = set()
for pair in high_corr_pairs:
    high_corr_indices.add(pair[0])
    high_corr_indices.add(pair[1])
# 将索引去重并排序
high_corr_indices = sorted(high_corr_indices)
###################################################PCC相关系数###################################


X = X[:, high_corr_indices]  # 只选择高相关特征
# 初始化特征重要性数组

num_features = X.shape[1]
num_iterations = 100
feature_importances = np.zeros((num_iterations, num_features))
for i in range(num_iterations):
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=30,
        random_state=i  
    )
    gb_model.fit(X,Y)
    feature_importances[i, :] = gb_model.feature_importances_
average_feature_importances = np.mean(feature_importances, axis=0)
std_feature_importances = np.std(feature_importances, axis=0)


# 6. 将重要性与特征名称对应
importance_df = pd.DataFrame({
    'Feature': [labels[i] for i in high_corr_indices],  # 使用原始标签
    'Importance': average_feature_importances,
    'Std': std_feature_importances
}).sort_values('Importance', ascending=False)

print("特征重要性排序:")
print(importance_df)

# 7. 可视化重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, xerr=importance_df['Std']*2,palette='RdBu_r')
plt.title('Gradient Boosting Feature Importance ')
plt.tight_layout()
plt.savefig(os.path.join(resultpath, 'GB_Feature_Importance_Averaged.png'), dpi=300)
plt.show()
plt.figure(figsize=(10, 6))

# 1. 绘制条形图（隐藏默认误差棒）
ax = sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df, 
    palette='RdBu_r',
    alpha=1  # 半透明显示，突出哑铃误差棒
)

# 2. 自定义哑铃形误差棒
for i, (feature, mean, std) in enumerate(zip(importance_df['Feature'], 
                                            importance_df['Importance'], 
                                            importance_df['Std'])):
    # 计算误差范围（假设误差是 ±2*std）
    lower = mean - 2 * std
    upper = mean + 2 * std
    
    # 绘制哑铃的横线
    ax.plot([lower, upper], [i, i], 
            color='black', 
            linewidth=1.5,
            solid_capstyle='round')  # 圆角端点
    
    # 绘制哑铃的左右端点
    ax.scatter([lower, upper], [i, i], 
                color='black', 
                s=40,  # 点的大小
                marker='|',  # 竖线形状
                linewidth=1.5)

# 3. 美化图形
plt.title('Gradient Boosting Feature Importance', pad=20)
plt.xlabel('Importance Score')

plt.tight_layout()
plt.savefig(os.path.join(resultpath, 'GB_Feature_Importance.jpg'), dpi=300, bbox_inches='tight')
plt.show()


