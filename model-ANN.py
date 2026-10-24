import argparse, copy, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent
DATA_PATH = r"E:\dm\R1_data\data_featured_去冗余_knn插补.xlsx"
RESULT_PATH = r"E:\dm\result\ann.xlsx"
MODEL_PATH = r"E:\dm\result\ann_model.pth"
TARGET = 'rate_constant(/min)'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Regressor(nn.Module):
    def __init__(self, inputs, width, depth, dropout):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), nn.SiLU(), nn.Dropout(dropout)])
            inputs = width
        layers.append(nn.Linear(inputs, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)

def fit(x, y, p, epochs, seed, validation=None):
    seed_all(seed)
    model = Regressor(x.shape[1], p['width'], p['depth'], p['dropout']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.65)
    xt, yt = tensor(x), tensor(y.reshape(-1, 1))
    if validation is not None:
        xv, yv = tensor(validation[0]), tensor(validation[1].reshape(-1, 1))
    best_loss, best_epoch, state = float('inf'), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            train_loss = nn.functional.mse_loss(model(xt), yt).item()
            val_loss = nn.functional.mse_loss(model(xv), yv).item() if validation is not None else None
        history.append({'epoch': epoch, 'train_mse_scaled': train_loss, 'val_mse_scaled': val_loss})
        if validation is not None:
            if val_loss < best_loss - 1e-7:
                best_loss, best_epoch = val_loss, epoch
                state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= 100:
                break
    if validation is not None:
        model.load_state_dict(state)
    else:
        best_epoch = epochs
    return model, best_epoch, history

def predict(model, x, ys):
    model.eval()
    with torch.no_grad():
        pred = model(tensor(x)).cpu().numpy()
    return ys.inverse_transform(pred).ravel()

def metrics(y, pred):
    return {'r2': float(r2_score(y, pred)), 'rmse': float(np.sqrt(mean_squared_error(y, pred))), 'mae': float(mean_absolute_error(y, pred))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(DATA_PATH))
    parser.add_argument('--result', type=Path, default=Path(RESULT_PATH))
    parser.add_argument('--model', type=Path, default=Path(MODEL_PATH))
    parser.add_argument('--output', type=Path, default=Path(RESULT_PATH).parent / 'ann_details')
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--seed', type=int, default=66)
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument('--no-target-scaling', dest='no_target_scaling', action='store_true', help='Train directly on the supplied target values (default)')
    scaling.add_argument('--target-scaling', dest='no_target_scaling', action='store_false', help='Enable target standardization')
    parser.set_defaults(no_target_scaling=True)
    args = parser.parse_args()
    if args.trials < 1 or args.epochs < 1:
        parser.error('trials and epochs must be positive')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.data)
    df = df.loc[df[TARGET].notna()].copy()
    x = df.drop(columns=[TARGET, 'source', 'group'], errors='ignore').apply(pd.to_numeric, errors='raise')
    x = x.replace([np.inf, -np.inf], np.nan)
    no_scale_prefixes = ('maccs_', 'anode_', 'cathode_', 'electrolyte_')
    scale_indices = [i for i, col in enumerate(x.columns) if not col.startswith(no_scale_prefixes)]
    no_scale_columns = [col for col in x.columns if col.startswith(no_scale_prefixes)]
    print(f'Standardized continuous features: {len(scale_indices)}; unscaled binary features: {len(no_scale_columns)}', flush=True)
    y = df[TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Target contains infinite values')
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=0.2, random_state=args.seed)
    fit_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=args.seed)
    # Fit all preprocessing on training rows only. Keep test fixed for final evaluation.
    def prepare(rows):
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        scaler = ColumnTransformer(
            [('continuous', StandardScaler(), scale_indices)], remainder='passthrough')
        ys = StandardScaler(with_mean=not args.no_target_scaling,
                            with_std=not args.no_target_scaling).fit(y[rows, None])
        transformed = scaler.fit_transform(imputer.fit_transform(x.iloc[rows]))
        return imputer, scaler, ys, transformed.astype(np.float32)
    imputer, scaler, ys, xf = prepare(fit_ids)
    xv = scaler.transform(imputer.transform(x.iloc[val_ids])).astype(np.float32)
    yf, yv = ys.transform(y[fit_ids, None]).ravel(), ys.transform(y[val_ids, None]).ravel()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    def objective(trial):
        p = {'width': trial.suggest_categorical('width', [128, 256, 512]),
             'depth': trial.suggest_int('depth', 2, 4),
             'dropout': trial.suggest_categorical('dropout', [0.0, 0.02, 0.05]),
             'lr': trial.suggest_float('lr', 3e-4, 3e-3, log=True),
             'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)}
        model, epoch, _ = fit(xf, yf, p, args.epochs, args.seed, (xv, yv))
        trial.set_user_attr('best_epoch', epoch)
        score = r2_score(y[val_ids], predict(model, xv, ys))
        print(f'Trial {trial.number}: validation R2={score:.5f}, epoch={epoch}', flush=True)
        return score
    study.enqueue_trial({'width': 256, 'depth': 3, 'dropout': 0.0, 'lr': 0.001, 'weight_decay': 1e-5})
    study.optimize(objective, n_trials=args.trials)
    p, epochs = study.best_params, study.best_trial.user_attrs['best_epoch']
    imputer, scaler, ys, xtrain = prepare(train_ids)
    model, _, history = fit(xtrain, ys.transform(y[train_ids, None]).ravel(), p, epochs, args.seed)
    # Only now transform and evaluate the independent test set; never select using test R2.
    xtest = scaler.transform(imputer.transform(x.iloc[test_ids])).astype(np.float32)
    train_pred, test_pred = predict(model, xtrain, ys), predict(model, xtest, ys)
    report = {'train': metrics(y[train_ids], train_pred), 'test': metrics(y[test_ids], test_pred),
              'validation_r2': study.best_value, 'params': p, 'refit_epochs': epochs, 'seed': args.seed,
              'rows': len(y), 'features': x.shape[1], 'unscaled_columns': no_scale_columns,
              'preprocessing': 'continuous standardized; specified binary prefixes unchanged',
              'target_standardized': not args.no_target_scaling}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    predictions = {}
    for name, rows, pred in [('train', train_ids, train_pred), ('test', test_ids, test_pred)]:
        predictions[name] = pd.DataFrame({'excel_row': df.index.to_numpy()[rows] + 2,
                                         'actual': y[rows], 'prediction': pred, 'residual': y[rows] - pred})
        predictions[name].to_csv(args.output / f'{name}_predictions.csv', index=False)
    pd.DataFrame(history).to_csv(args.output / 'refit_history.csv', index=False)
    study.trials_dataframe().to_csv(args.output / 'trials.csv', index=False)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.result, engine='openpyxl') as writer:
        predictions['train'].to_excel(writer, sheet_name='Train_Predictions', index=False)
        predictions['test'].to_excel(writer, sheet_name='Test_Predictions', index=False)
        pd.DataFrame([{'split': name, **report[name]} for name in ('train', 'test')]).to_excel(
            writer, sheet_name='Metrics', index=False)
        pd.DataFrame([{'parameter': key, 'value': value} for key, value in {
            **p, 'refit_epochs': epochs, 'seed': args.seed,
            'target_standardized': not args.no_target_scaling,
            'validation_r2': study.best_value}.items()]).to_excel(writer, sheet_name='Model_Info', index=False)
        pd.DataFrame(history).to_excel(writer, sheet_name='Loss_History', index=False)
        study.trials_dataframe().to_excel(writer, sheet_name='Optimization_Results', index=False)
    torch.save({'model_state_dict': model.cpu().state_dict(), 'params': p, 'feature_columns': list(x.columns),
                'imputer': imputer, 'scaler': scaler, 'target_scaler': ys, 'report': report}, args.model)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Results saved to: {args.result}\nModel saved to: {args.model}', flush=True)

if __name__ == '__main__':
    main()
