
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from skopt import BayesSearchCV
from skopt.space import Real


# ============================================================
# 1. 基础设置
# ============================================================

seed = 66

input_path = r"E:\dm\R1_data\data_featured_去冗余_knn插补.xlsx"
output_path = r"E:\dm\result\svr.xlsx"

target_column = 'rate_constant(/min)'
columns_to_drop = ['source', 'group']

TEST_SIZE = 0.2
N_ITER = 50
CV = 5
N_JOBS = 14


# ============================================================
# 2. 加载数据
# ============================================================

df = pd.read_excel(input_path)

# 删除目标变量为空的样本
n_missing_y = df[target_column].isna().sum()

if n_missing_y > 0:
    df = df.dropna(subset=[target_column]).copy()
    print(f"删除目标变量缺失的样本数: {n_missing_y}")

# 删除目标列和非建模列
X = df.drop(
    columns=[target_column] + columns_to_drop,
    errors='ignore'
)

y = df[target_column]

# 检查是否存在非数值特征
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric_cols:
    print("发现非数值特征:", non_numeric_cols)
    raise ValueError("特征矩阵包含非数值列，请先检查并编码。")

# 检查无穷值
X = X.replace([np.inf, -np.inf], np.nan)

print("\n========== 数据基本信息 ==========")
print(f"原始特征矩阵: {X.shape}")
print(f"目标变量: {y.shape}")
print(f"X中NaN总数: {X.isna().sum().sum()}")
print(f"y中NaN总数: {y.isna().sum()}")

nan_columns = X.columns[X.isna().any()].tolist()

print(f"存在缺失值的特征数: {len(nan_columns)}")

if nan_columns:
    print("\n缺失值统计:")
    print(X[nan_columns].isna().sum().sort_values(ascending=False))


# ============================================================
# 3. 划分训练集和测试集
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=seed
)

print("\n========== 数据划分 ==========")
print(f"训练集: X{X_train.shape}, y{y_train.shape}")
print(f"测试集: X{X_test.shape}, y{y_test.shape}")


# ============================================================
# 4. 识别特征类型
# ============================================================

# 不进行标准化的特征前缀
no_scale_prefixes = [
    'maccs_',
    'anode_',
    'cathode_',
    'electrolyte_'
]

# 即使符合上述前缀，也需要标准化的连续变量
force_scale_cols = [
    'anode_area(cm2)',
    'electrolyte_concentration(mM)'
]

no_scale_cols = []

for col in X.columns:

    if col in force_scale_cols:
        continue

    if any(col.startswith(prefix) for prefix in no_scale_prefixes):
        no_scale_cols.append(col)

scale_cols = [
    col for col in X.columns
    if col not in no_scale_cols
]

print("\n========== 特征类型 ==========")
print(f"标准化特征数: {len(scale_cols)}")
print(f"不标准化特征数: {len(no_scale_cols)}")

# 检查不标准化的特征是否全部为0/1编码
non_binary_cols = []

for col in no_scale_cols:
    unique_values = X_train[col].dropna().unique()

    if not np.isin(unique_values, [0, 1]).all():
        non_binary_cols.append(col)

if non_binary_cols:
    print("\n警告：以下不标准化特征不是严格的0/1编码:")
    print(non_binary_cols)


# ============================================================
# 5. 构建数据预处理Pipeline
# ============================================================

# 连续特征:
# KNN填充 -> StandardScaler
numeric_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(
        n_neighbors=10,
        weights='uniform',
        keep_empty_features=True
    )),
    ('scaler', StandardScaler())
])

# 0/1特征:
# 缺失值使用众数填充，不进行标准化
binary_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(
        strategy='most_frequent',
        keep_empty_features=True
    ))
])

# 按特征类型进行预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_pipeline, scale_cols),
        ('binary', binary_pipeline, no_scale_cols)
    ],
    remainder='drop'
)

# 完整Pipeline:
# 预处理 -> SVR
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR(kernel='rbf'))
])


# ============================================================
# 6. 贝叶斯超参数优化
# ============================================================

search_space = {
    'svr__C': Real(
        0.1, 1000,
        prior='log-uniform'
    ),

    'svr__epsilon': Real(
        0.01, 1.0,
        prior='log-uniform'
    ),

    'svr__gamma': Real(
        0.0001, 1.0,
        prior='log-uniform'
    )
}

bayes_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space,
    n_iter=N_ITER,
    cv=CV,
    scoring='r2',
    random_state=seed,
    n_jobs=N_JOBS,
    n_points=2,
    verbose=0,
    refit=True,
    error_score='raise'
)

print("\n========== 贝叶斯超参数优化 ==========")

bayes_search.fit(X_train, y_train)

best_params = bayes_search.best_params_

print("\n最佳超参数:")
for name, value in best_params.items():
    print(f"{name}: {value}")

print(f"\n训练集CV R2: {bayes_search.best_score_:.3f}")


# ============================================================
# 7. 获取最优模型并预测
# ============================================================

# BayesSearchCV已经在全部训练集上重新拟合最优模型
final_model = bayes_search.best_estimator_

# 训练集预测
y_train_pred = final_model.predict(X_train)

train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

# 测试集预测
y_test_pred = final_model.predict(X_test)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

# 泛化差距
generalization_gap = train_r2 - test_r2


# ============================================================
# 8. 输出模型性能
# ============================================================

print("\n========== SVR模型性能 ==========")

print(f"训练集R2:   {train_r2:.3f}")
print(f"测试集R2:   {test_r2:.3f}")

print(f"训练集RMSE: {train_rmse:.3f}")
print(f"测试集RMSE: {test_rmse:.3f}")

print(f"训练集MAE:  {train_mae:.3f}")
print(f"测试集MAE:  {test_mae:.3f}")

print(f"训练集CV R2: {bayes_search.best_score_:.3f}")
print(f"泛化差距(R2): {generalization_gap:.3f}")

if generalization_gap > 0.2:
    print("警告：模型可能存在较大过拟合！")
elif generalization_gap > 0.1:
    print("注意：模型存在一定过拟合。")
else:
    print("训练集与测试集R2差距较小。")


# ============================================================
# 9. 保存预测结果和模型参数
# ============================================================

def save_results():

    # Sheet1: 训练集和测试集预测结果
    train_df = pd.DataFrame({
        'Actual_Train': np.asarray(y_train),
        'Predicted_Train': y_train_pred
    })

    test_df = pd.DataFrame({
        'Actual_Test': np.asarray(y_test),
        'Predicted_Test': y_test_pred
    })

    max_len = max(len(train_df), len(test_df))

    predictions_df = pd.DataFrame(index=range(max_len))

    predictions_df['Actual_Train'] = train_df['Actual_Train']
    predictions_df['Predicted_Train'] = train_df['Predicted_Train']

    predictions_df['Actual_Test'] = test_df['Actual_Test']
    predictions_df['Predicted_Test'] = test_df['Predicted_Test']


    # Sheet2: 模型评价指标
    metrics_df = pd.DataFrame({
        'Metric': [
            '训练集样本数',
            '测试集样本数',
            '训练集R2',
            '测试集R2',
            '训练集RMSE',
            '测试集RMSE',
            '训练集MAE',
            '测试集MAE',
            '训练集CV R2',
            '泛化差距(R2)'
        ],
        'Value': [
            len(y_train),
            len(y_test),
            train_r2,
            test_r2,
            train_rmse,
            test_rmse,
            train_mae,
            test_mae,
            bayes_search.best_score_,
            generalization_gap
        ]
    })


    # Sheet3: 超参数与预处理信息
    params_data = []

    for param_name, param_value in best_params.items():
        params_data.append([
            param_name.replace('svr__', ''),
            param_value
        ])

    params_data.extend([
        ['kernel', 'rbf'],
        ['random_state', seed],
        ['n_iter', N_ITER],
        ['cv', CV],
        ['scoring', 'r2'],
        ['model', 'SVR'],
        ['imputer_method', 'KNNImputer'],
        ['n_neighbors', 10],
        ['imputer_weights', 'uniform'],
        ['binary_imputer', 'most_frequent'],
        ['scaler', 'StandardScaler'],
        ['test_size', TEST_SIZE],
        ['n_features', X.shape[1]],
        ['n_scaled_features', len(scale_cols)],
        ['n_unscaled_features', len(no_scale_cols)]
    ])

    params_df = pd.DataFrame(
        params_data,
        columns=['Parameter', 'Value']
    )


    # Sheet4: 模型性能对比
    performance_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'R2': [train_r2, test_r2],
        'RMSE': [train_rmse, test_rmse],
        'MAE': [train_mae, test_mae],
        'Generalization_Gap': [
            np.nan,
            generalization_gap
        ]
    })


    # Sheet5: 原始样本索引与预测结果
    detailed_predictions_df = pd.concat([
        pd.DataFrame({
            'Original_Index': X_train.index,
            'Dataset': 'Train',
            'Actual': np.asarray(y_train),
            'Predicted': y_train_pred
        }),
        pd.DataFrame({
            'Original_Index': X_test.index,
            'Dataset': 'Test',
            'Actual': np.asarray(y_test),
            'Predicted': y_test_pred
        })
    ], ignore_index=True)


    # 创建输出文件夹
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # 保存Excel
    with pd.ExcelWriter(
        output_path,
        engine='openpyxl'
    ) as writer:

        predictions_df.to_excel(
            writer,
            sheet_name='Predictions',
            index=False
        )

        metrics_df.to_excel(
            writer,
            sheet_name='Metrics',
            index=False
        )

        params_df.to_excel(
            writer,
            sheet_name='Hyperparameters',
            index=False
        )

        performance_df.to_excel(
            writer,
            sheet_name='Performance_Comparison',
            index=False
        )

        detailed_predictions_df.to_excel(
            writer,
            sheet_name='Detailed_Predictions',
            index=False
        )

    print(f"\n结果已保存至: {output_path}")


# ============================================================
# 10. 执行结果保存
# ============================================================

save_results()

print("\n========== SVR建模完成 ==========")
