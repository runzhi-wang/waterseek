import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Integer, Real

# 1. 基础设置
seed = 66
input_path = r"E:\dm\R1_data\data_featured_去冗余_knn插补.xlsx"
output_path = r"E:\dm\result\rf.xlsx"

start = time.time()

# 2. 读取数据
df = pd.read_excel(input_path)
target = 'rate_constant(/min)'
df = df.dropna(subset=[target])

X = df.drop(columns=[target, 'source', 'group', 'processing time (s)'], errors='ignore')
y = df[target]

# 数据检查
if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
    raise ValueError("X包含非数值特征")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("X仍存在NaN或无穷值")

print(f"数据形状: X{X.shape}, y{y.shape}")

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# 4. 贝叶斯超参数优化
search_space = {
    'n_estimators': Integer(100, 2000),
    'max_depth': Integer(5, 25),
    'min_samples_split': Integer(2, 15),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.4, 1.0)
}

model = RandomForestRegressor(
    random_state=seed,
    n_jobs=-1,
    bootstrap=True
)

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=seed,
    n_jobs=8,
    n_points=4,
    verbose=0,
    refit=True,
    error_score='raise'
)

bayes_search.fit(X_train, y_train)
final_model = bayes_search.best_estimator_

print("\n最佳参数:", bayes_search.best_params_)
print(f"CV R2: {bayes_search.best_score_:.3f}")

# 5. 模型预测与评价
train_pred = final_model.predict(X_train)
test_pred = final_model.predict(X_test)


def evaluate(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

train_metrics = evaluate(y_train, train_pred)
test_metrics = evaluate(y_test, test_pred)

for name, m in [('Train', train_metrics), ('Test', test_metrics)]:
    print(
        f"{name} | R2: {m['R2']:.3f} | "
        f"RMSE: {m['RMSE']:.3f} | MAE: {m['MAE']:.3f}"
    )

print(f"泛化差距(R2): {train_metrics['R2'] - test_metrics['R2']:.3f}")

# 6. 保存结果
train_df = pd.DataFrame({
    'Actual_Train': np.asarray(y_train),
    'Predicted_Train': train_pred
})
test_df = pd.DataFrame({
    'Actual_Test': np.asarray(y_test),
    'Predicted_Test': test_pred
})

predictions = pd.concat(
    [train_df.reset_index(drop=True),
     test_df.reset_index(drop=True)],
    axis=1
)

metrics = pd.DataFrame({
    'Metric': [
        '训练集样本数', '测试集样本数', '训练集CV R2',
        '训练集R2', '测试集R2',
        '训练集RMSE', '测试集RMSE',
        '训练集MAE', '测试集MAE',
        '泛化差距(R2)'
    ],
    'Value': [
        len(y_train), len(y_test), bayes_search.best_score_,
        train_metrics['R2'], test_metrics['R2'],
        train_metrics['RMSE'], test_metrics['RMSE'],
        train_metrics['MAE'], test_metrics['MAE'],
        train_metrics['R2'] - test_metrics['R2']
    ]
})

params = pd.DataFrame(
    list(bayes_search.best_params_.items()) + [
        ('random_state', seed),
        ('n_iter', 50),
        ('cv', 5),
        ('n_features', X.shape[1]),
        ('preprocessing', 'Pre-imputed KNN dataset')
    ],
    columns=['Parameter', 'Value']
)

performance = pd.DataFrame([
    {'Dataset': 'Train', **train_metrics},
    {'Dataset': 'Test', **test_metrics}
])

details = pd.concat([
    pd.DataFrame({
        'Original_Index': X_train.index,
        'Dataset': 'Train',
        'Actual': np.asarray(y_train),
        'Predicted': train_pred
    }),
    pd.DataFrame({
        'Original_Index': X_test.index,
        'Dataset': 'Test',
        'Actual': np.asarray(y_test),
        'Predicted': test_pred
    })
], ignore_index=True)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    predictions.to_excel(writer, sheet_name='Predictions', index=False)
    metrics.to_excel(writer, sheet_name='Metrics', index=False)
    params.to_excel(writer, sheet_name='Hyperparameters', index=False)
    performance.to_excel(writer, sheet_name='Performance_Comparison', index=False)
    details.to_excel(writer, sheet_name='Detailed_Predictions', index=False)

print(f"\n结果已保存: {output_path}")
print(f"总耗时: {(time.time() - start) / 60:.2f} 分钟")
