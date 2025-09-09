import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random
import os
from gplearn.functions import make_function

# 创建保存输出的文件夹
output_folder = 'symbolic_regression_output'
os.makedirs(output_folder, exist_ok=True)

# # 加载和准备数据
filepath = r''
resultpath =  r''
data = pd.read_excel(filepath, sheet_name='Symbolic_regression')  
data_mat = data.iloc[:, :].values
label_mat =  data.iloc[:, :].values.T
alloy_names = data.iloc[:, :].values
feature_names = data.iloc[:, :].tolist()
data_mat = np.array(data_mat)
label_mat = np.array(label_mat)

# 创建变量名的映射
variable_mapping = {f'X{i}': name for i, name in enumerate(feature_names)}

def set_random_seed():
    return random.randint(0, 10000)

def plot_and_save(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, formula, train_metrics, test_metrics, iteration):
    plt.figure(figsize=(12, 8))
    plt.scatter(y_train, y_train_pred, alpha=0.5, label='Training Set', color='blue')
    plt.scatter(y_test, y_test_pred, alpha=0.5, label='Test Set', color='red')
    
    min_val = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Iteration {iteration + 1}: Actual vs Predicted\nFormula: {formula}')
    plt.legend()
    
    # 添加指标文本
    text = f"Train R² = {train_metrics['R2']:.4f}, RMSE = {train_metrics['RMSE']:.4f}\n"
    text += f"Test R² = {test_metrics['R2']:.4f}, RMSE = {test_metrics['RMSE']:.4f}"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    filename = f'iteration_{iteration+1}_combined.png'
    plt.savefig(os.path.join(output_folder, filename))
    
    # 显示图片
    plt.show()

# 定义安全的指数函数
def safe_exp(x):
    with np.errstate(over='ignore'):
        return np.where(x < 100, np.exp(x), np.exp(100))
# 定义安全的自然对数函数
def safe_ln(x):
    return np.log(np.maximum(x, 1e-10))

# 创建自定义函数
exp_func = make_function(function=safe_exp,
                         name='exp',
                         arity=1)

ln_func = make_function(function=safe_ln,
                        name='ln',
                        arity=1)

# 创建一个列表来存储每次迭代的结果
summary_results = []

for iteration in range(1):
    # 设置随机种子
    random_seed1 = set_random_seed()
    random_seed2 = set_random_seed()
    random_seed3 = set_random_seed()
    
    # 随机生成 population_size 和 generations
    generations = random.randint(10, 60)
    population_size = random.randint(max(100, generations * 10), 1000)

    print(f"\n迭代 {iteration + 1}:")
    print(f"Random Seeds: {random_seed1}, {random_seed2}, {random_seed3}")
    print(f"Population Size: {population_size}")
    print(f"Generations: {generations}")

    # 数据洗牌和划分
    shuffled_data, shuffled_label, shuffled_alloy = shuffle(data_mat, label_mat, alloy_names, random_state=random_seed1)
    y_binned = pd.cut(shuffled_label, bins=5, labels=False)
    X_train, X_test, y_train, y_test, alloy_train, alloy_test = train_test_split(
        shuffled_data, shuffled_label, shuffled_alloy, test_size=0.2, random_state=random_seed2, stratify=y_binned
    )

    est_gp = SymbolicRegressor(population_size=population_size,
                               generations=generations, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=0,
                               function_set=['add', 'sub', 'mul', 'div', 'sqrt',
                                             'inv', exp_func, ln_func], 
                               parsimony_coefficient=0.001, random_state=random_seed3) 
                               #对目标变量进行标准化（standardization）和区间缩放（min-max-scaling）
                               #可以有效避免常数值区间不符的问题。
    est_gp.fit(X_train, y_train)

    # 获取最佳公式并替换变量名
    best_program = est_gp._program
    formula = str(best_program)
    for var, descriptor in variable_mapping.items():
        formula = formula.replace(var, descriptor)

    print(f"最佳公式: {formula}")

    # 获取预测结果并计算指标
    train_pred = est_gp.predict(X_train)
    test_pred = est_gp.predict(X_test)

    train_metrics = {
        'R2': r2_score(y_train, train_pred),
        'RMSE': mean_squared_error(y_train, train_pred, squared=False),
        'MAE': mean_absolute_error(y_train, train_pred)
    }

    test_metrics = {
        'R2': r2_score(y_test, test_pred),
        'RMSE': mean_squared_error(y_test, test_pred, squared=False),
        'MAE': mean_absolute_error(y_test, test_pred)
    }

    # 打印结果
    print(f"训练集 - R²: {train_metrics['R2']:.4f}, RMSE: {train_metrics['RMSE']:.4f}, MAE: {train_metrics['MAE']:.4f}")
    print(f"测试集 - R²: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, MAE: {test_metrics['MAE']:.4f}")

    # 绘制、保存并显示散点图
    plot_and_save(X_train, y_train, train_pred, X_test, y_test, test_pred, formula, train_metrics, test_metrics, iteration)

    # 准备并保存数据到Excel
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df.insert(0, '合金名', alloy_train)
    train_df['实际值'] = y_train
    train_df['预测值'] = train_pred
    train_df['数据集'] = 'Train'

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df.insert(0, '合金名', alloy_test)
    test_df['实际值'] = y_test
    test_df['预测值'] = test_pred
    test_df['数据集'] = 'Test'

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # 添加指标信息
    metrics_df = pd.DataFrame({
        '指标': ['Train R2', 'Train RMSE', 'Train MAE', 'Test R2', 'Test RMSE', 'Test MAE', 'Formula'],
        '值': [train_metrics['R2'], train_metrics['RMSE'], train_metrics['MAE'],
               test_metrics['R2'], test_metrics['RMSE'], test_metrics['MAE'], formula]
    })

    # 保存到Excel
    with pd.ExcelWriter(os.path.join(output_folder, f'iteration_{iteration+1}_results.xlsx')) as writer:
        combined_df.to_excel(writer, sheet_name='Data', index=False)
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

    # 添加结果到summary_results列表
    summary_results.append({
        '迭代': iteration + 1,
        '公式': formula,
        '训练集_R2': train_metrics['R2'],
        '训练集_RMSE': train_metrics['RMSE'],
        '训练集_MAE': train_metrics['MAE'],
        '测试集_R2': test_metrics['R2'],
        '测试集_RMSE': test_metrics['RMSE'],
        '测试集_MAE': test_metrics['MAE']
    })

# 将summary_results保存为Excel文件
summary_df = pd.DataFrame(summary_results)
summary_xlsx_path = os.path.join(output_folder, 'summary_results.xlsx')
with pd.ExcelWriter(summary_xlsx_path, engine='openpyxl') as writer:
    summary_df.to_excel(writer, index=False, sheet_name='汇总结果')
    # 调整列宽
    worksheet = writer.sheets['汇总结果']
    for idx, col in enumerate(summary_df):
        max_length = max(summary_df[col].astype(str).map(len).max(), len(col))
        worksheet.column_dimensions[worksheet.cell(row=1, column=idx+1).column_letter].width = max_length + 2

print(f"\n所有结果已保存到 '{output_folder}' 文件夹中")
print(f"汇总结果已保存到 '{summary_xlsx_path}' 文件中")

# 为了确保控制台输出也正确显示中文，可以添加以下代码：
print("\n汇总结果预览：")
print(summary_df.to_string(index=False))


