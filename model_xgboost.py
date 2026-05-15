import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler

seed = 42

# 1. 加载数据
df = pd.read_excel(r"E:\dm\data\data_featured_去冗余.xlsx")
target_column = 'rate_constant(/min)'
columns_to_drop = ['source']

# ==============================
# 参数控制
# ==============================

X = df.drop(columns=[target_column] + columns_to_drop)
y = df[target_column]
print(f"数据形状: X{X.shape}, y{y.shape}")

# 2. 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


# 3. 识别特征类型并进行标准化
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

print(f"\n标准化处理: {len(scale_cols)}个特征标准化{scale_cols}, {len(no_scale_cols)}个特征保持原样")

# 4. 贝叶斯调参（只在训练集上）
search_space = {
    'n_estimators': Integer(200, 1800),
    'max_depth': Integer(3, 9),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 5),  # 最小分裂损失
    'reg_alpha': Real(0, 5),  # L1正则
    'reg_lambda': Real(0, 5),  # L2正则
    'min_child_weight': Integer(1, 6),  # 叶子节点最小样本数
    'max_delta_step': Integer(0, 10)  # 控制每棵树权重的最大增量
}

model = xgb.XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=50,
    cv=5,  # 训练集上进行5或10折交叉验证
    scoring='r2',
    random_state=seed,
    n_jobs=-1,
    verbose=0,
    n_points=2,  # 并行设置
)

print("在训练集上进行贝叶斯调参...")
bayes_search.fit(X_train, y_train)
best_params = bayes_search.best_params_

print(f"最佳参数: {best_params}")

# 5. 创建最优模型
final_model = xgb.XGBRegressor(**best_params, random_state=seed, n_jobs=-1, verbosity=0)
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

print(f"训练集R2:  {train_r2:.3f}")
print(f"测试集R2:  {test_r2:.3f}")
print(f"训练集RMSE: {train_rmse:.3f}")
print(f"测试集RMSE: {test_rmse:.3f}")
print(f"训练集MAE:  {train_mae:.3f}")
print(f"测试集MAE:  {test_mae:.3f}")


# 6. 修改的保存函数：在第二个sheet存储训练集测试集的RMSE MAE R2 以及超参数
def save_results_with_metrics_and_params(y_train_true, y_train_pred, y_test_true, y_test_pred,
                                         train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae,
                                         best_params, filepath):
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
                   '训练集CV R2'],
        'Value': [len(train_df), len(test_df), f"{train_r2:.4f}", f"{test_r2:.4f}",
                  f"{train_rmse:.4f}", f"{test_rmse:.4f}", f"{train_mae:.4f}", f"{test_mae:.4f}",
                  f"{bayes_search.best_score_:.4f}"]
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
            'MAE': [round(train_mae, 3), round(test_mae, 3)]
        })
        performance_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)

    print(f"结果已保存到: {filepath}")


    return predictions_df, metrics_df, params_df, performance_df


# 保存所有结果
output_path = r"E:\dm\result\xgb.xlsx"
predictions_df, metrics_df, params_df, performance_df = save_results_with_metrics_and_params(
    y_train, y_train_pred, y_test, y_test_pred,
    train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae,
    best_params, output_path
)

# 7. 可选：保存模型
# final_model.save_model(r'E:\dm\result\final_xgb_model.json')
