import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler

seed = 42

# 1. 加载数据
df = pd.read_excel(r"E:\dm\data\data_featured_去冗余.xlsx")
target_column = 'rate_constant(/min)'
columns_to_drop = ['source']

X = df.drop(columns=[target_column])
y = df[target_column]

# 检查NaN值
print(f"原始数据形状: X{X.shape}, y{y.shape}")
print(f"X中的NaN数量: {X.isna().sum().sum()}")
print(f"y中的NaN数量: {y.isna().sum()}")

# 2. 使用KNN填充缺失值
print("使用KNN填充缺失值...")
# 首先检查每列是否有缺失值
nan_columns = X.columns[X.isna().any()].tolist()
print(f"包含缺失值的列数: {len(nan_columns)}")

# 创建KNNImputer进行缺失值填充
knn_imputer = KNNImputer(n_neighbors=10, weights='uniform')
X_imputed = knn_imputer.fit_transform(X)

# 转换为DataFrame并保持列名
X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

print(f"填充后X中的NaN数量: {X_imputed.isna().sum().sum()}")

# 检查y中是否有缺失值
if y.isna().sum() > 0:
    # 如果有，删除对应的行
    y_clean = y.dropna()
    X_imputed = X_imputed.loc[y_clean.index]
    y = y_clean
    print(f"删除了{y.shape[0] - y_clean.shape[0]}个y中的缺失值")

# 3. 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=seed)

# 4. 识别特征类型并进行标准化
# 定义不应该标准化的特征前缀列表
no_scale_prefixes = ['maccs_', 'anode_', 'cathode_', 'electrolyte_']

# 识别不应该标准化的特征
no_scale_cols = []
for col in X.columns:
    if col in ['anode_area(cm2)', 'electrolyte_concentration(mM)']:
        continue
    for prefix in no_scale_prefixes:
        if col.startswith(prefix):
            no_scale_cols.append(col)
            break  # 找到匹配前缀后就跳出

# 其他特征应该标准化
scale_cols = [col for col in X.columns if col not in no_scale_cols]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[scale_cols])
X_test_scaled = scaler.transform(X_test[scale_cols])

# 不标准化的特征保持原样
X_train_no_scale = X_train[no_scale_cols].values
X_test_no_scale = X_test[no_scale_cols].values

# 重新组合特征
X_train = np.hstack([X_train_scaled, X_train_no_scale])
X_test = np.hstack([X_test_scaled, X_test_no_scale])

print(f"训练集形状: X{X_train.shape}, y{y_train.shape}")
print(f"测试集形状: X{X_test.shape}, y{y_test.shape}")

# 5. 贝叶斯调参（只在训练集上）
search_space = {
    'C': Real(0.1, 1000, prior='log-uniform'),  # 正则化参数
    'epsilon': Real(0.01, 1.0, prior='log-uniform'),  # epsilon不敏感带
    'gamma': Real(0.0001, 1.0, prior='log-uniform'),  # RBF核参数
    'kernel': ['rbf'],
}

model = SVR()

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=seed,
    n_jobs=14,
    verbose=0,
    n_points=2,
)

print("\n在训练集上进行贝叶斯调参...")
bayes_search.fit(X_train, y_train)
best_params = bayes_search.best_params_

print(f"最佳参数: {best_params}")
print(f"训练集CV R2: {bayes_search.best_score_:.3f}")

# 6. 创建最优模型
final_model = SVR(**best_params)
final_model.fit(X_train, y_train)

# 预测训练集和测试集
y_train_pred = final_model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

y_test_pred = final_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n模型性能指标:")
print(f"训练集R2:  {train_r2:.3f}")
print(f"测试集R2:  {test_r2:.3f}")
print(f"训练集RMSE: {train_rmse:.3f}")
print(f"测试集RMSE: {test_rmse:.3f}")
print(f"训练集MAE:  {train_mae:.3f}")
print(f"测试集MAE:  {test_mae:.3f}")

# 计算泛化差距
generalization_gap = train_r2 - test_r2
print(f"泛化差距(R2): {generalization_gap:.3f}")

if generalization_gap > 0.2:
    print("警告：模型可能存在较大过拟合！")
elif generalization_gap > 0.1:
    print("注意：模型存在一定过拟合。")
else:
    print("模型泛化性能良好。")


# 7. 修改的保存函数：在第二个sheet存储训练集测试集的RMSE MAE R2 以及超参数
def save_results_with_metrics_and_params(y_train_true, y_train_pred, y_test_true, y_test_pred,
                                         train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae,
                                         best_params, filepath, imputer_info=None):
    """
    保存预测结果和模型评估指标到Excel文件

    Sheet1: 预测结果
    Sheet2: 模型评估指标和超参数
    """
    # 创建训练集结果DataFrame
    train_df = pd.DataFrame({
        'Actual_Train': y_train_true,
        'Predicted_Train': y_train_pred
    })

    # 创建测试集结果DataFrame
    test_df = pd.DataFrame({
        'Actual_Test': y_test_true,
        'Predicted_Test': y_test_pred
    })

    # 合并预测结果到一个DataFrame
    max_len = max(len(train_df), len(test_df))
    predictions_df = pd.DataFrame(index=range(max_len))

    # 填充训练集数据
    predictions_df['Actual_Train'] = train_df['Actual_Train'].reset_index(drop=True)
    predictions_df['Predicted_Train'] = train_df['Predicted_Train'].reset_index(drop=True)

    # 填充测试集数据
    predictions_df['Actual_Test'] = test_df['Actual_Test'].reset_index(drop=True)
    predictions_df['Predicted_Test'] = test_df['Predicted_Test'].reset_index(drop=True)

    # 创建模型评估指标DataFrame
    metrics_data = {
        'Metric': ['训练集样本数', '测试集样本数', '训练集R2', '测试集R2',
                   '训练集RMSE', '测试集RMSE', '训练集MAE', '测试集MAE',
                   '训练集CV R2', '泛化差距(R2)'],
        'Value': [len(train_df), len(test_df), f"{train_r2:.3f}", f"{test_r2:.3f}",
                  f"{train_rmse:.3f}", f"{test_rmse:.3f}", f"{train_mae:.3f}", f"{test_mae:.3f}",
                  f"{bayes_search.best_score_:.3f}", f"{generalization_gap:.3f}"]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # 创建超参数DataFrame
    params_data = []
    for param_name, param_value in best_params.items():
        params_data.append([param_name, param_value])

    # 添加其他重要参数
    params_data.append(['random_state', seed])
    params_data.append(['n_iter', bayes_search.n_iter])
    params_data.append(['cv', bayes_search.cv])
    params_data.append(['scoring', 'r2'])
    params_data.append(['model', 'SVR'])

    # 添加数据预处理信息
    if imputer_info:
        params_data.append(['imputer_method', 'KNNImputer'])
        params_data.append(['n_neighbors', imputer_info.get('n_neighbors', 10)])
        params_data.append(['imputer_weights', imputer_info.get('weights', 'uniform')])

    params_df = pd.DataFrame(params_data, columns=['Parameter', 'Value'])

    # 使用ExcelWriter保存多个sheet
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet1: 预测结果
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)

        # Sheet2: 模型评估指标
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

        # Sheet3: 超参数
        params_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

        # Sheet4: 模型性能对比
        performance_df = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'R2': [round(train_r2, 3), round(test_r2, 3)],
            'RMSE': [round(train_rmse, 3), round(test_rmse, 3)],
            'MAE': [round(train_mae, 3), round(test_mae, 3)],
            'Generalization_Gap': [np.nan, round(generalization_gap, 3)]
        })
        performance_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)

    print(f"\n结果已保存到: {filepath}")

    print(f"\n训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")

    return predictions_df, metrics_df, params_df, performance_df


# 保存所有结果
output_path = r"E:\dm\result\svr.xlsx"
imputer_info = {
    'n_neighbors': 10,
    'weights': 'uniform'
}

predictions_df, metrics_df, params_df, performance_df = save_results_with_metrics_and_params(
    y_train, y_train_pred, y_test, y_test_pred,
    train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae,
    best_params, output_path, imputer_info
)
