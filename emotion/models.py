"""XGBoost classifier with fixed hyper-parameters (same for every condition) and early stopping on the
validation actor. Class imbalance handled with 'balanced' sample weights (train and validation)."""
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

N_CLASSES = 7
XGB_PARAMS = dict(max_depth=4, learning_rate=0.05, n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
                  objective='multi:softprob', eval_metric='mlogloss', early_stopping_rounds=50,
                  tree_method='hist', random_state=0, n_jobs=8)


def fit_xgb(X, y, X_val, y_val, params=XGB_PARAMS):
    model = XGBClassifier(**params)
    model.fit(X, y, sample_weight=compute_sample_weight('balanced', y),
              eval_set=[(X_val, y_val)], sample_weight_eval_set=[compute_sample_weight('balanced', y_val)],
              verbose=False)
    return model


def gain_importance(model, names):
    """Total gain per feature, normalised to sum to 1 (features never used get 0)."""
    g = model.get_booster().get_score(importance_type='total_gain')
    v = np.array([g.get(f'f{i}', 0.0) for i in range(len(names))])
    return dict(zip(names, v / max(v.sum(), 1e-12)))
