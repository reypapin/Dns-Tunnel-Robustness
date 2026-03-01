# models/

Trained model files go here after running the notebook.

The notebook saves them to Google Drive and also expects to load them from there, so this folder is mainly for reference or if you adapt the code to run locally.

**Baseline models** (produced by Experiment 1):
```
LogisticRegression_sota.joblib
RandomForest_sota.joblib
XGBoost_sota.joblib
LightGBM_sota.joblib
CNN_sota.keras
LSTM_sota.keras
scaler_sota.joblib
```

**Hardened models** (produced by Experiment 4):
```
LogisticRegression_hardened.joblib
RandomForest_hardened.joblib
XGBoost_hardened.joblib
LightGBM_hardened.joblib
CNN_hardened.keras
LSTM_hardened.keras
```

Model files are not committed to the repo because of size.
