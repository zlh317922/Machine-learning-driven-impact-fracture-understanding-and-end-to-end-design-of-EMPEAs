# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:08:29 2024

@author: zlh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import shap

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


selected_features = [ ] 
features = ['v', 'ΔSmix', 'ΔR', 'ΔTE', 'ΔK']
selected_features_name = []
for i in range(len(selected_features)):
    selected_features_name.append(Feature_names[selected_features[i]])
print('选择特征为：', selected_features_name)

X = X[:, selected_features]

Modelpath =  r' '
model = load(fr'model.m') 
model.fit(X, Y)


# 初始化SHAP
shap.initjs()
explainer = shap.KernelExplainer(model.predict, X)
shap_values = explainer.shap_values(X)
plt.rc('font', family='Times New Roman')
# 保存第一个SHAP图
plt.xlim(-0.3, 0.3)
shap.summary_plot(shap_values, X, feature_names=features,show=False)
# 手动调整当前图形的属性
ax = plt.gca()
ax.set_xlabel('SHAP value (impact on model output)', fontsize=20)
# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(
    axis='both',
    labelsize=16,
    direction='in',  # 刻度朝内
    length=6,        # 刻度线长度
    width=1          # 刻度线宽度
)
plt.savefig(fr'{resultpath}\AdaSVR-SHAP1.tif', dpi=300, format='tif')
plt.close()


# # 保存第二个SHAP图
plt.xlim(0, 0.2)
shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar",show=False)
# 手动调整当前图形的属性
ax = plt.gca()
ax.set_xlabel('mean(|SHAP value|) (average impact on model output)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(
    axis='both',
    labelsize=16,
    direction='in',  # 刻度朝内
    length=6,        # 刻度线长度
    width=1          # 刻度线宽度
)
plt.savefig(fr'{resultpath}\AdaSVR-SHAP2.tif', dpi=300, format='tif')
plt.close()



