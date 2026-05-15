import copy
import warnings
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# =========================================================
# 1. 基础设置
# =========================================================
SEED = 42
DATA_PATH = r"E:\dm\data\data_featured_去冗余.xlsx"
RESULT_PATH = r"E:\dm\result\ann.xlsx"
MODEL_PATH = r"E:\dm\result\ann_model.pth"

TARGET_COLUMN = "rate_constant(/min)"
COLUMNS_TO_DROP = ["source"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================================================
# 2. 加载数据
# =========================================================
df = pd.read_excel(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"原始数据形状: X{X.shape}, y{y.shape}")
print(f"X中的NaN数量: {X.isna().sum().sum()}")
print(f"y中的NaN数量: {y.isna().sum()}")

# =========================================================
# 3. 缺失值处理（按你的要求：保持原逻辑，先KNN再划分）
# =========================================================
knn_imputer = KNNImputer(n_neighbors=10, weights="uniform")
X_imputed = knn_imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# 删除 y 中缺失
valid_idx = y.dropna().index
X_imputed = X_imputed.loc[valid_idx].reset_index(drop=True)
y = y.loc[valid_idx].reset_index(drop=True)

print(f"\n删除 y 缺失后: X{X_imputed.shape}, y{y.shape}")


# =========================================================
# 4. 数据划分
#   先 train/test
#   再从 train 中划出 val，避免用 test 调参
# =========================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=SEED
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# =========================================================
# 5. 特征标准化
# =========================================================
def identify_scale_columns(columns):
    """识别需要标准化的特征列"""
    no_scale_prefixes = ["maccs_", "anode_", "cathode_", "electrolyte_"]
    scale_cols, no_scale_cols = [], []

    for col in columns:
        if any(col.startswith(prefix) for prefix in no_scale_prefixes):
            no_scale_cols.append(col)
        else:
            scale_cols.append(col)

    return scale_cols, no_scale_cols


def preprocess_features_fit_transform(X_train, X_val, X_test, scale_cols, no_scale_cols):
    """
    仅在训练集上拟合 scaler，再变换 train/val/test
    保留你原来“不缩放 one-hot / 指纹类特征”的思路
    """
    scaler = None

    if scale_cols:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[scale_cols])
        X_val_scaled = scaler.transform(X_val[scale_cols])
        X_test_scaled = scaler.transform(X_test[scale_cols])
    else:
        X_train_scaled = np.empty((X_train.shape[0], 0), dtype=np.float32)
        X_val_scaled = np.empty((X_val.shape[0], 0), dtype=np.float32)
        X_test_scaled = np.empty((X_test.shape[0], 0), dtype=np.float32)

    if no_scale_cols:
        X_train_noscale = X_train[no_scale_cols].to_numpy(dtype=np.float32)
        X_val_noscale = X_val[no_scale_cols].to_numpy(dtype=np.float32)
        X_test_noscale = X_test[no_scale_cols].to_numpy(dtype=np.float32)

        X_train_final = np.hstack([X_train_scaled, X_train_noscale])
        X_val_final = np.hstack([X_val_scaled, X_val_noscale])
        X_test_final = np.hstack([X_test_scaled, X_test_noscale])
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        X_test_final = X_test_scaled

    return (
        X_train_final.astype(np.float32),
        X_val_final.astype(np.float32),
        X_test_final.astype(np.float32),
        scaler
    )


scale_cols, no_scale_cols = identify_scale_columns(X.columns)
X_train_final, X_val_final, X_test_final, scaler = preprocess_features_fit_transform(
    X_train, X_val, X_test, scale_cols, no_scale_cols
)

# 最终训练时需要对 train_full 也做一次同样的变换
def transform_train_full(X_train_full, scale_cols, no_scale_cols, scaler):
    if scale_cols:
        X_train_full_scaled = scaler.transform(X_train_full[scale_cols])
    else:
        X_train_full_scaled = np.empty((X_train_full.shape[0], 0), dtype=np.float32)

    if no_scale_cols:
        X_train_full_noscale = X_train_full[no_scale_cols].to_numpy(dtype=np.float32)
        X_train_full_final = np.hstack([X_train_full_scaled, X_train_full_noscale])
    else:
        X_train_full_final = X_train_full_scaled

    return X_train_full_final.astype(np.float32)


X_train_full_final = transform_train_full(X_train_full, scale_cols, no_scale_cols, scaler)


# =========================================================
# 6. 转 Tensor
# =========================================================
def to_tensor(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)
    return X_tensor, y_tensor


X_train_tensor, y_train_tensor = to_tensor(X_train_final, y_train)
X_val_tensor, y_val_tensor = to_tensor(X_val_final, y_val)
X_test_tensor, y_test_tensor = to_tensor(X_test_final, y_test)
X_train_full_tensor, y_train_full_tensor = to_tensor(X_train_full_final, y_train_full)


# =========================================================
# 7. ANN模型定义
#   去掉原来不稳定的 residual_connection
# =========================================================
class EnhancedANNRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate,
                 use_batchnorm=True, activation="relu"):
        super().__init__()

        activation_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "selu": nn.SELU()
        }

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_batchnorm and hidden_size > 1:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(activation_map[activation])

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =========================================================
# 8. 通用工具函数
# =========================================================
def create_dataloader(X_tensor, y_tensor, batch_size, shuffle=True):
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }


def predict_model(model, X_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(DEVICE)).cpu().numpy().flatten()
    return preds


def train_one_model(
    model,
    train_loader,
    val_loader,
    learning_rate,
    weight_decay,
    epochs=200,
    patience=20,
    l1_lambda=1e-5
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    best_val_r2 = -np.inf
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ---------- train ----------
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        # ---------- val ----------
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                total_val_loss += loss.item() * X_batch.size(0)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(y_batch.cpu().numpy().flatten())

        val_loss = total_val_loss / len(val_loader.dataset)
        val_r2 = r2_score(val_targets, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, best_val_r2


# =========================================================
# 9. Optuna超参数优化
#   只用 train / val，不碰 test
# =========================================================
def objective(trial):
    '''
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_sizes = [
        trial.suggest_int(f"hidden_size_{i}", 32, 256, log=True)
        for i in range(n_layers)
    ]
    '''
    hidden_sizes = trial.suggest_categorical("hidden_sizes", [
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (128, 128, 64),
        (256, 128, 64),
        (64, 64, 32),
        (128, 96, 64, 32)
    ])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "selu"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
    l1_lambda = trial.suggest_float("l1_lambda", 1e-7, 1e-4, log=True)

    train_loader = create_dataloader(X_train_tensor, y_train_tensor, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_tensor, y_val_tensor, batch_size, shuffle=False)

    model = EnhancedANNRegressor(
        input_size=X_train_tensor.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        activation=activation
    ).to(DEVICE)

    model, _, _, best_val_r2 = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=120,
        patience=15,
        l1_lambda=l1_lambda
    )

    return best_val_r2


print("\n开始超参数优化...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print(f"\n最优超参数: {best_params}")
print(f"最佳验证集R²: {study.best_value:.4f}")


# =========================================================
# 10. 使用最优参数训练最终模型
#   在 train_full 上训练，内部再划一个小验证集做 early stopping
# =========================================================
'''
n_layers = best_params["n_layers"]
hidden_sizes = [best_params[f"hidden_size_{i}"] for i in range(n_layers)]
'''
hidden_sizes = best_params["hidden_sizes"]
final_params = {
    "hidden_sizes": hidden_sizes,
    "dropout_rate": best_params["dropout_rate"],
    "learning_rate": best_params["learning_rate"],
    "weight_decay": best_params["weight_decay"],
    "activation": best_params["activation"],
    "batch_size": best_params["batch_size"],
    "use_batchnorm": best_params["use_batchnorm"],
    "l1_lambda": best_params["l1_lambda"]
}

# 从 train_full 中再划一部分做 early stopping
X_subtrain, X_subval, y_subtrain, y_subval = train_test_split(
    X_train_full_final, y_train_full, test_size=0.15, random_state=SEED
)

X_subtrain_tensor, y_subtrain_tensor = to_tensor(X_subtrain, y_subtrain)
X_subval_tensor, y_subval_tensor = to_tensor(X_subval, y_subval)

train_loader_final = create_dataloader(
    X_subtrain_tensor, y_subtrain_tensor, final_params["batch_size"], shuffle=True
)
val_loader_final = create_dataloader(
    X_subval_tensor, y_subval_tensor, final_params["batch_size"], shuffle=False
)

final_model = EnhancedANNRegressor(
    input_size=X_train_full_tensor.shape[1],
    hidden_sizes=final_params["hidden_sizes"],
    dropout_rate=final_params["dropout_rate"],
    use_batchnorm=final_params["use_batchnorm"],
    activation=final_params["activation"]
).to(DEVICE)

print("\n使用最优参数训练最终模型...")
final_model, train_losses, val_losses, best_val_r2 = train_one_model(
    model=final_model,
    train_loader=train_loader_final,
    val_loader=val_loader_final,
    learning_rate=final_params["learning_rate"],
    weight_decay=final_params["weight_decay"],
    epochs=300,
    patience=25,
    l1_lambda=final_params["l1_lambda"]
)


# =========================================================
# 11. 最终评估
# =========================================================
y_train_full_pred = predict_model(final_model, X_train_full_tensor)
y_test_pred = predict_model(final_model, X_test_tensor)

y_train_full_true = y_train_full_tensor.numpy().flatten()
y_test_true = y_test_tensor.numpy().flatten()

train_metrics = evaluate_metrics(y_train_full_true, y_train_full_pred)
test_metrics = evaluate_metrics(y_test_true, y_test_pred)

print(f"\n{'=' * 60}")
print("模型性能评估 (优化后)")
print(f"{'=' * 60}")
print(f"训练集 - R²:  {train_metrics['r2']:.4f}")
print(f"         RMSE: {train_metrics['rmse']:.4f}")
print(f"         MAE:  {train_metrics['mae']:.4f}")
print(f"测试集 - R²:  {test_metrics['r2']:.4f}")
print(f"         RMSE: {test_metrics['rmse']:.4f}")
print(f"         MAE:  {test_metrics['mae']:.4f}")
print(f"泛化差距: {train_metrics['r2'] - test_metrics['r2']:.4f}")


# =========================================================
# 12. 保存结果
# =========================================================
def save_optimized_results(
    y_train_true, y_train_pred,
    y_test_true, y_test_pred,
    train_metrics, test_metrics,
    final_params, study,
    train_losses, val_losses,
    best_val_r2, filepath
):
    """保存优化结果到Excel"""

    # 训练集预测
    train_pred_df = pd.DataFrame({
        "Actual_Train": y_train_true,
        "Predicted_Train": y_train_pred,
        "Error_Train": y_train_true - y_train_pred
    })

    # 测试集预测
    test_pred_df = pd.DataFrame({
        "Actual_Test": y_test_true,
        "Predicted_Test": y_test_pred,
        "Error_Test": y_test_true - y_test_pred
    })

    # 模型信息
    model_info = pd.DataFrame({
        "参数": [
            "隐藏层结构", "Dropout率", "学习率", "权重衰减", "L1正则",
            "激活函数", "批次大小", "BatchNorm",
            "训练集R²", "测试集R²",
            "训练集RMSE", "测试集RMSE",
            "训练集MAE", "测试集MAE",
            "泛化差距", "最佳验证R²"
        ],
        "值": [
            str(final_params["hidden_sizes"]),
            final_params["dropout_rate"],
            final_params["learning_rate"],
            final_params["weight_decay"],
            final_params["l1_lambda"],
            final_params["activation"],
            final_params["batch_size"],
            final_params["use_batchnorm"],
            train_metrics["r2"],
            test_metrics["r2"],
            train_metrics["rmse"],
            test_metrics["rmse"],
            train_metrics["mae"],
            test_metrics["mae"],
            train_metrics["r2"] - test_metrics["r2"],
            best_val_r2
        ]
    })

    # 训练历史
    loss_history = pd.DataFrame({
        "Epoch": range(1, len(train_losses) + 1),
        "Train_Loss": train_losses,
        "Val_Loss": val_losses
    })

    # Optuna结果
    trials_df = study.trials_dataframe()

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        train_pred_df.to_excel(writer, sheet_name="Train_Predictions", index=False)
        test_pred_df.to_excel(writer, sheet_name="Test_Predictions", index=False)
        model_info.to_excel(writer, sheet_name="Model_Info", index=False)
        loss_history.to_excel(writer, sheet_name="Loss_History", index=False)
        trials_df.to_excel(writer, sheet_name="Optimization_Results", index=False)

    print(f"\n优化结果已保存到: {filepath}")


save_optimized_results(
    y_train_full_true, y_train_full_pred,
    y_test_true, y_test_pred,
    train_metrics, test_metrics,
    final_params, study,
    train_losses, val_losses,
    best_val_r2,
    RESULT_PATH
)


# =========================================================
# 13. 保存模型
# =========================================================
torch.save({
    "model_state_dict": final_model.state_dict(),
    "best_params": final_params,
    "input_size": X_train_full_tensor.shape[1],
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "scale_cols": scale_cols,
    "no_scale_cols": no_scale_cols
}, MODEL_PATH)

print("最优模型已保存")
print(f"最终测试集R²: {test_metrics['r2']:.4f}")