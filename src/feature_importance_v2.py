# =============================================================================
# ANÁLISIS PROFUNDO: FEATURE IMPORTANCE Y ROBUSTEZ DIFERENCIAL (TODOS LOS MODELOS)
# VERSIÓN V2 — 4 correcciones sobre el notebook original:
#   1. RF con std real (entre árboles individuales)
#   2. CSV consolidado: 1 fila por feature × columnas por modelo (importance + std)
#   3. Columna 'category' exportada: Positional / Stateful / Other
#   4. recall_score real (clase ataque) en lugar de accuracy global
#   5. Barras de error (xerr=std) en figura para RF, CNN, LSTM
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.metrics import recall_score          # ← FIX 4
from tensorflow import keras

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MODEL_DIR  = "/content/drive/MyDrive/Tunnel/CSV_CIC21/Models_SOTA_Hybrid"
ARAGAT_DIR = "/content/drive/MyDrive/Tunnel/CSV_CIC21/CSV_Generated"

FEATS_SL = [
    "FQDN_count", "subdomain_length", "upper", "lower", "numeric",
    "entropy", "special", "labels", "labels_max", "labels_average", "len"
]
FEATS_SF = [
    "rr",
    "A_frequency", "AAAA_frequency", "CNAME_frequency", "TXT_frequency",
    "MX_frequency", "NS_frequency", "NULL_frequency",
    "rr_count", "distinct_ip", "unique_ttl", "total_queries"
]
ALL_FEATS = FEATS_SL + FEATS_SF   # 23 features

# =============================================================================
# FIX 3 — Categorías de features (definidas globalmente para poder exportarlas)
# =============================================================================

# Positional: dependen del contenido/posición del payload DNS name
POSITIONAL_FEATS = [
    'FQDN_count', 'subdomain_length', 'upper', 'lower', 'numeric',
    'entropy', 'special', 'labels', 'labels_max', 'labels_average', 'len'
]
# Stateful: estadísticas de flujo, más invariantes a posición
STATEFUL_FEATS = [
    'rr', 'A_frequency', 'AAAA_frequency', 'CNAME_frequency', 'TXT_frequency',
    'MX_frequency', 'NS_frequency', 'NULL_frequency',
    'rr_count', 'distinct_ip', 'unique_ttl', 'total_queries'
]

def assign_category(feature_name):
    if feature_name in POSITIONAL_FEATS:
        return 'Positional'
    elif feature_name in STATEFUL_FEATS:
        return 'Stateful'
    return 'Other'

# =============================================================================
# FUNCIÓN PARA AUTO-DETECTAR SEEDS
# =============================================================================

def detect_available_seeds(directory, verbose=True):
    if verbose:
        print(f"\n🔍 Scanning directory: {directory}")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"❌ Directory not found: {directory}")
    files = os.listdir(directory)
    pattern = re.compile(r'stateless_features-bridge\.pcap_(\d+)\.csv')
    seeds_found = set()
    for filename in files:
        match = pattern.match(filename)
        if match:
            seed = int(match.group(1))
            if f"stateful_features-bridge.pcap_{seed}.csv" in files:
                seeds_found.add(seed)
                if verbose:
                    try:
                        df_temp = pd.read_csv(os.path.join(directory, filename))
                        n_attack = (df_temp['label'] == 1).sum() if 'label' in df_temp.columns else 0
                        print(f"   ✅ Found seed {seed}: {len(df_temp):,} flows ({n_attack} attack)")
                    except Exception:
                        print(f"   ✅ Found seed {seed}")
            elif verbose:
                print(f"   ⚠️  Seed {seed}: stateless found but stateful missing")
    seeds_sorted = sorted(list(seeds_found))
    if verbose:
        print(f"\n📊 Total seeds found: {len(seeds_sorted)}")
        if seeds_sorted:
            print(f"   Seeds: {', '.join(map(str, seeds_sorted))}")
    return seeds_sorted

# =============================================================================
# FUNCIÓN PARA CARGAR DATOS (DYNAMIC SEEDS)
# =============================================================================

def process_single_seed(stateless_path, stateful_path, seed_name, verbose=True):
    if verbose:
        print(f"\n   📂 Processing seed {seed_name}...")
    df_sl = pd.read_csv(stateless_path)
    df_sf = pd.read_csv(stateful_path)
    if verbose:
        print(f"      ├─ Stateless: {len(df_sl):,} rows")
        print(f"      └─ Stateful:  {len(df_sf):,} rows")
    for c in FEATS_SL:
        df_sl[c] = pd.to_numeric(df_sl.get(c, 0), errors='coerce').fillna(0)
    aggregation_dict = {feat: 'mean' for feat in FEATS_SL if feat in df_sl.columns}
    if 'label' in df_sl.columns:
        aggregation_dict['label'] = 'first'
    df_sl_agg = df_sl.groupby('tunnel_id').agg(aggregation_dict).reset_index()
    df_sl_agg['tunnel_id'] = df_sl_agg['tunnel_id'].astype(str) + f"_s{seed_name}"
    if verbose:
        print(f"      ├─ Aggregated to {len(df_sl_agg):,} flows")
    for c in FEATS_SF:
        df_sf[c] = pd.to_numeric(df_sf.get(c, 0), errors='coerce').fillna(0)
    df_sf['tunnel_id'] = df_sf['tunnel_id'].astype(str) + f"_s{seed_name}"
    df_merged = pd.merge(df_sl_agg, df_sf, on='tunnel_id', how='inner', suffixes=('', '_sf'))
    if verbose:
        print(f"      └─ Merged: {len(df_merged):,} flows")
    df_merged = df_merged.reindex(columns=ALL_FEATS + ['label', 'tunnel_id'], fill_value=0)
    df_merged['label'] = df_merged['label'].fillna(1).astype(int) if 'label' in df_merged.columns else 1
    df_merged[ALL_FEATS] = df_merged[ALL_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0)
    df_merged['seed'] = int(seed_name)
    return df_merged

def load_aragat_dynamic_seeds(specific_seeds=None, verbose=True):
    if verbose:
        print("\n" + "="*70)
        print("💀 LOADING ARAGAT/MUTANT-DNS DATASET (DYNAMIC SEEDS)")
        print("="*70)
    available_seeds = detect_available_seeds(ARAGAT_DIR, verbose=verbose)
    if not available_seeds:
        raise ValueError("❌ No seeds found in directory!")
    seeds_to_load = specific_seeds if specific_seeds is not None else available_seeds
    if specific_seeds is not None:
        missing = set(seeds_to_load) - set(available_seeds)
        if missing:
            raise ValueError(f"❌ Seeds not found: {sorted(missing)}")
        if verbose:
            print(f"\n📥 Loading {len(seeds_to_load)} seed(s): {', '.join(map(str, seeds_to_load))}")
    elif verbose:
        print(f"\n📥 Loading ALL {len(seeds_to_load)} seed(s): {', '.join(map(str, seeds_to_load))}")
    dfs = []
    for seed in seeds_to_load:
        df_seed = process_single_seed(
            os.path.join(ARAGAT_DIR, f"stateless_features-bridge.pcap_{seed}.csv"),
            os.path.join(ARAGAT_DIR, f"stateful_features-bridge.pcap_{seed}.csv"),
            seed_name=str(seed), verbose=verbose
        )
        dfs.append(df_seed)
    if verbose:
        print(f"\n   🔗 Combining {len(dfs)} seed(s)...")
    df_combined = pd.concat(dfs, ignore_index=True)
    if verbose:
        n_attack = (df_combined['label'] == 1).sum()
        n_benign = (df_combined['label'] == 0).sum()
        print(f"      ✅ Total combined: {len(df_combined):,} flows")
        print(f"         ├─ Attack:  {n_attack:,} ({n_attack/len(df_combined)*100:.1f}%)")
        print(f"         └─ Benign:  {n_benign:,} ({n_benign/len(df_combined)*100:.1f}%)")
    return (df_combined[ALL_FEATS].values,
            df_combined['label'].values,
            df_combined['tunnel_id'].values,
            df_combined['seed'].values)

# =============================================================================
# FIX 1 — extract_tree_importance CON STD REAL PARA RF
# =============================================================================

def extract_tree_importance(model, model_name, feature_names):
    """
    Extrae feature importance de modelos tree-based.
    Para RandomForest calcula std real entre los árboles individuales.
    XGBoost y LightGBM no exponen estimators_ → std = 0.
    """
    importance = model.feature_importances_

    # FIX 1: std entre árboles individuales (solo RandomForest tiene estimators_)
    if hasattr(model, 'estimators_') and model_name == 'RandomForest':
        importances_per_tree = np.array([t.feature_importances_
                                         for t in model.estimators_])
        std = importances_per_tree.std(axis=0)
    else:
        std = np.zeros(len(importance))  # XGB/LGB: sin acceso por árbol

    return pd.DataFrame({
        'feature'   : feature_names,
        'importance': importance,
        'std'       : std,          # ← nuevo
        'model'     : model_name,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

# =============================================================================
# EXTRAER COEFICIENTES DE LOGISTIC REGRESSION
# =============================================================================

def extract_logreg_coefficients(model, feature_names):
    coef = model.coef_[0]
    return pd.DataFrame({
        'feature'        : feature_names,
        'coefficient'    : coef,
        'abs_coefficient': np.abs(coef),
        'std'            : np.zeros(len(coef)),   # sin std para LogReg
        'model'          : 'LogisticRegression',
    }).sort_values('abs_coefficient', ascending=False).reset_index(drop=True)

# =============================================================================
# PERMUTATION IMPORTANCE PARA DL MODELS (CNN Y LSTM)
# =============================================================================

def compute_dl_permutation_importance(model, X_test, y_test, feature_names,
                                      model_name, n_repeats=5):
    print(f"   🔍 Computing {model_name} permutation importance...")
    from sklearn.metrics import f1_score

    class DLWrapper(BaseEstimator):
        def __init__(self, dl_model): self.model = dl_model
        def fit(self, X, y): return self
        def predict(self, X):
            Xr = X.reshape(X.shape[0], X.shape[1], 1) if X.ndim == 2 else X
            return (self.model.predict(Xr, verbose=0) > 0.5).astype(int).flatten()
        def score(self, X, y):
            return f1_score(y, self.predict(X), zero_division=0)

    wrapper = DLWrapper(model)
    result = permutation_importance(
        wrapper, X_test, y_test,
        n_repeats=n_repeats, random_state=42,
        scoring=lambda e, X, y: f1_score(y, e.predict(X), zero_division=0)
    )
    return pd.DataFrame({
        'feature'   : feature_names,
        'importance': result.importances_mean,
        'std'       : result.importances_std,   # ya existía
        'model'     : model_name,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

# =============================================================================
# FIX 2 — CSV CONSOLIDADO: 1 fila por feature × columnas por modelo
# =============================================================================

def build_consolidated_csv(importance_dict, recall_dict, save_path=None):
    """
    Genera DataFrame con 23 filas (una por feature) y columnas:
        feature | category |
        rf_importance | rf_std |
        xgb_importance | xgb_std |
        lgb_importance | lgb_std |
        cnn_importance | cnn_std |
        lstm_importance | lstm_std |
        logreg_abs_coef | logreg_std

    FIX 3: columna 'category' con Positional / Stateful / Other
    """
    df_out = pd.DataFrame({'feature': ALL_FEATS})
    df_out['category'] = df_out['feature'].apply(assign_category)   # FIX 3

    model_col_map = {
        'RandomForest'      : ('rf_importance',    'rf_std'),
        'XGBoost'           : ('xgb_importance',   'xgb_std'),
        'LightGBM'          : ('lgb_importance',   'lgb_std'),
        'CNN'               : ('cnn_importance',   'cnn_std'),
        'LSTM'              : ('lstm_importance',  'lstm_std'),
        'LogisticRegression': ('logreg_abs_coef',  'logreg_std'),
    }

    for model_name, (col_imp, col_std) in model_col_map.items():
        if model_name not in importance_dict:
            df_out[col_imp] = np.nan
            df_out[col_std] = np.nan
            continue
        df_m = importance_dict[model_name].copy()
        # LogReg usa 'abs_coefficient'; los demás usan 'importance'
        imp_col = 'abs_coefficient' if model_name == 'LogisticRegression' else 'importance'
        merge_cols = ['feature', imp_col, 'std']
        df_out = df_out.merge(df_m[merge_cols], on='feature', how='left')
        df_out = df_out.rename(columns={imp_col: col_imp, 'std': col_std})

    if save_path:
        df_out.to_csv(save_path, index=False)
        print(f"   ✅ CSV consolidado guardado: {save_path}")
    return df_out

# =============================================================================
# FIX 5 — FIGURA CON BARRAS DE ERROR (xerr=std)
# =============================================================================

def plot_feature_importance_comparison(importance_dict, recall_dict, save_path=None):
    """
    Compara feature importance entre 6 modelos.
    FIX 5: añade xerr=std para RF, CNN, LSTM.
    Colores: rojo = Positional, verde = Stateful, gris = Other.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Feature Importance (±1 SD): All Models — Positional vs. Stateful',
                 fontsize=15, fontweight='bold')

    COLOR_CAT = {'Positional': '#c0392b', 'Stateful': '#27ae60', 'Other': '#7f8c8d'}

    panels = [
        ('RandomForest',      'importance',     'rf_imp',   axes[0, 0]),
        ('XGBoost',           'importance',     'xgb_imp',  axes[0, 1]),
        ('LightGBM',          'importance',     'lgb_imp',  axes[0, 2]),
        ('LSTM',              'importance',     'lstm_imp', axes[1, 0]),
        ('CNN',               'importance',     'cnn_imp',  axes[1, 1]),
        ('LogisticRegression','abs_coefficient','lr_imp',   axes[1, 2]),
    ]

    for model_name, imp_col, _, ax in panels:
        if model_name not in importance_dict:
            ax.set_visible(False)
            continue
        df_m  = importance_dict[model_name].head(15).copy()
        cats  = df_m['feature'].apply(assign_category)
        colors = [COLOR_CAT[c] for c in cats]
        xerr  = df_m['std'].values if (df_m['std'] > 0).any() else None

        ax.barh(df_m['feature'], df_m[imp_col],
                xerr=xerr,
                color=colors, alpha=0.75,
                error_kw=dict(ecolor='#2c3e50', capsize=3, lw=0.8))
        ax.invert_yaxis()

        recall_pct = recall_dict.get(model_name, 0.0) * 100
        imp_label  = '|Coefficient|' if model_name == 'LogisticRegression' else 'Importance'
        ax.set_xlabel(imp_label, fontsize=9, fontweight='bold')
        ax.set_title(f'{model_name}\n({recall_pct:.2f}% Recall)',
                     fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)

    # Leyenda global
    legend_patches = [
        mpatches.Patch(facecolor=COLOR_CAT['Positional'],
                       label='Positional — DNS name patterns'),
        mpatches.Patch(facecolor=COLOR_CAT['Stateful'],
                       label='Stateful — flow statistics'),
        mpatches.Patch(facecolor=COLOR_CAT['Other'], label='Other'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, 0.00))
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Figura guardada: {save_path}")
    plt.show()

# =============================================================================
# TOP FEATURES POR TIPO DE MODELO (sin cambios)
# =============================================================================

def plot_top_features_by_model_type(importance_dict, recall_dict, save_path=None):
    from collections import Counter
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    positional_features = POSITIONAL_FEATS

    panels_cfg = [
        (['RandomForest', 'XGBoost', 'LightGBM'], 'Tree-Based Models', 'red',    axes[0]),
        (['LSTM'],                                 'LSTM Model',        'orange', axes[1]),
        (['LogisticRegression', 'CNN'],            'Robust Models',     'green',  axes[2]),
    ]
    for models, title, color, ax in panels_cfg:
        all_feats = []
        for m in models:
            if m in importance_dict:
                imp_col = 'abs_coefficient' if m == 'LogisticRegression' else 'feature'
                all_feats.extend(importance_dict[m].head(5)['feature'].tolist())
        counts = Counter(all_feats)
        df_c = pd.DataFrame(counts.items(), columns=['feature', 'count']
                            ).sort_values('count', ascending=False).head(10)
        bar_colors = ['#c0392b' if f in positional_features else '#27ae60'
                      for f in df_c['feature']]
        ax.barh(df_c['feature'], df_c['count'], color=bar_colors, alpha=0.7)
        avg_recall = np.mean([recall_dict.get(m, 0) for m in models]) * 100
        ax.set_title(f'{title}\n({avg_recall:.2f}% Recall)',
                     fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('Frequency in Top 5', fontweight='bold')
        ax.invert_yaxis()

    plt.suptitle('Feature Patterns: Vulnerable vs Robust Models',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    plt.show()

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main(specific_seeds=None, subset_size=1500, n_repeats=5):
    print("\n" + "="*70)
    print("🔬 FEATURE IMPORTANCE ANALYSIS V2 — All 6 Models")
    print("="*70)

    X, y, tunnel_ids, seed_labels = load_aragat_dynamic_seeds(
        specific_seeds=specific_seeds, verbose=True)
    print(f"\n✅ Data loaded: {X.shape[0]:,} flows, {X.shape[1]} features")

    scaler_path = os.path.join(MODEL_DIR, "scaler_sota.joblib")
    scaler    = joblib.load(scaler_path)
    X_scaled  = scaler.transform(X)

    importance_dict  = {}
    recall_dict      = {}

    # ── 1. TREE-BASED MODELS ─────────────────────────────────────────────────
    print("\n📊 Tree-based models...")
    for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
        model_path = os.path.join(MODEL_DIR, f"{model_name}_sota.joblib")
        if not os.path.exists(model_path):
            print(f"   ⚠️  {model_name} not found, skipping.")
            continue
        model = joblib.load(model_path)
        importance_dict[model_name] = extract_tree_importance(model, model_name, ALL_FEATS)
        preds = model.predict(X)                                   # tree models use raw X
        # FIX 4: recall real de la clase ataque
        recall_dict[model_name] = recall_score(y, preds, zero_division=0)
        top5 = importance_dict[model_name].head(5)
        print(f"\n   {model_name} top 5 (recall={recall_dict[model_name]:.3f}):")
        for _, row in top5.iterrows():
            print(f"      {row['feature']:20s}: {row['importance']:.4f} ± {row['std']:.4f}")

    # ── 2. LOGISTIC REGRESSION ───────────────────────────────────────────────
    print("\n📊 LogisticRegression...")
    lr_path = os.path.join(MODEL_DIR, "LogisticRegression_sota.joblib")
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)
        importance_dict['LogisticRegression'] = extract_logreg_coefficients(lr_model, ALL_FEATS)
        preds_lr = lr_model.predict(X_scaled)
        recall_dict['LogisticRegression'] = recall_score(y, preds_lr, zero_division=0)  # FIX 4
        top5 = importance_dict['LogisticRegression'].head(5)
        print(f"   recall={recall_dict['LogisticRegression']:.3f}")
        for _, row in top5.iterrows():
            print(f"      {row['feature']:20s}: {row['coefficient']:+.4f}")
    else:
        print("   ⚠️  LogisticRegression not found, skipping.")

    # ── 3. CNN ────────────────────────────────────────────────────────────────
    print("\n📊 CNN permutation importance...")
    cnn_path = os.path.join(MODEL_DIR, "CNN_sota.keras")
    X_cnn = None
    if os.path.exists(cnn_path):
        cnn_model = keras.models.load_model(cnn_path)
        np.random.seed(42)
        subset_idx = np.random.choice(len(X), size=min(subset_size, len(X)), replace=False)
        X_sub = X_scaled[subset_idx]; y_sub = y[subset_idx]
        importance_dict['CNN'] = compute_dl_permutation_importance(
            cnn_model, X_sub, y_sub, ALL_FEATS, 'CNN', n_repeats=n_repeats)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        preds_cnn = (cnn_model.predict(X_cnn, verbose=0) > 0.5).astype(int).flatten()
        recall_dict['CNN'] = recall_score(y, preds_cnn, zero_division=0)  # FIX 4
        top5 = importance_dict['CNN'].head(5)
        print(f"   recall={recall_dict['CNN']:.3f}")
        for _, row in top5.iterrows():
            print(f"      {row['feature']:20s}: {row['importance']:.4f} ± {row['std']:.4f}")
    else:
        print("   ⚠️  CNN not found, skipping.")

    # ── 4. LSTM ───────────────────────────────────────────────────────────────
    print("\n📊 LSTM permutation importance...")
    lstm_path = os.path.join(MODEL_DIR, "LSTM_sota.keras")
    if os.path.exists(lstm_path):
        lstm_model = keras.models.load_model(lstm_path)
        if X_cnn is None:   # por si CNN no estaba disponible
            X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        importance_dict['LSTM'] = compute_dl_permutation_importance(
            lstm_model, X_sub, y_sub, ALL_FEATS, 'LSTM', n_repeats=n_repeats)
        preds_lstm = (lstm_model.predict(X_cnn, verbose=0) > 0.5).astype(int).flatten()
        recall_dict['LSTM'] = recall_score(y, preds_lstm, zero_division=0)  # FIX 4
        top5 = importance_dict['LSTM'].head(5)
        print(f"   recall={recall_dict['LSTM']:.3f}")
        for _, row in top5.iterrows():
            print(f"      {row['feature']:20s}: {row['importance']:.4f} ± {row['std']:.4f}")
    else:
        print("   ⚠️  LSTM not found, skipping.")

    # ── 5. VISUALIZACIONES ────────────────────────────────────────────────────
    print("\n📈 Generando figuras...")
    plot_feature_importance_comparison(
        importance_dict, recall_dict,
        save_path=os.path.join(MODEL_DIR, "feature_importance_with_std.png"))
    plot_top_features_by_model_type(
        importance_dict, recall_dict,
        save_path=os.path.join(MODEL_DIR, "top_features_by_model_type_v2.png"))

    # ── 6. CSV CONSOLIDADO (FIX 2 + FIX 3) ───────────────────────────────────
    print("\n" + "="*70)
    print("📋 CSV CONSOLIDADO (feature × modelo)")
    print("="*70)
    csv_cons = os.path.join(MODEL_DIR, "feature_importance_consolidated.csv")
    df_cons  = build_consolidated_csv(importance_dict, recall_dict, save_path=csv_cons)
    print("\nVista previa (top 6 por rf_importance):")
    preview_cols = [c for c in ['feature','category','rf_importance','rf_std',
                                'cnn_importance','cnn_std','lstm_importance','lstm_std']
                    if c in df_cons.columns]
    print(df_cons.sort_values('rf_importance', ascending=False,
                              na_position='last').head(6)[preview_cols].to_string(index=False))

    # ── 7. RESUMEN POR CATEGORÍA ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 IMPORTANCIA MEDIA POR CATEGORÍA")
    print("="*70)
    for col, label in [('rf_importance', 'RF'),
                       ('cnn_importance', 'CNN'),
                       ('lstm_importance', 'LSTM')]:
        if col not in df_cons.columns:
            continue
        grp = df_cons.groupby('category')[col].mean().sort_values(ascending=False)
        print(f"\n{label}:")
        for cat, val in grp.items():
            print(f"   {cat:12s}: {val:.4f}")

    # ── 8. SUMMARY TABLE (igual que original) ─────────────────────────────────
    print("\n" + "="*70)
    print("📊 QUANTITATIVE SUMMARY")
    print("="*70)
    summary = []
    for model_name, df_m in importance_dict.items():
        top10 = df_m.head(10)['feature'].tolist()
        n_pos = sum(f in POSITIONAL_FEATS for f in top10)
        n_sta = sum(f in STATEFUL_FEATS   for f in top10)
        top1  = df_m.iloc[0]['feature']
        summary.append({
            'model'              : model_name,
            'top1_feature'       : top1,
            'category_top1'      : assign_category(top1),
            'positional_in_top10': n_pos,
            'stateful_in_top10'  : n_sta,
            'recall'             : f"{recall_dict.get(model_name, 0):.4f}",
        })
    df_summary = pd.DataFrame(summary).sort_values('recall')
    print("\n" + df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(MODEL_DIR, "feature_analysis_summary_v2.csv"), index=False)

    print("\n📁 Archivos guardados:")
    print(f"   ├─ feature_importance_with_std.png")
    print(f"   ├─ top_features_by_model_type_v2.png")
    print(f"   ├─ feature_importance_consolidated.csv   ← NUEVO")
    print(f"   └─ feature_analysis_summary_v2.csv")
    print("\n" + "="*70)
    print("✅ FEATURE IMPORTANCE V2 COMPLETO")
    print("="*70)

    return importance_dict, recall_dict, df_cons

# =============================================================================
# EJECUCIÓN — COLAB COMPATIBLE
# =============================================================================

if __name__ == "__main__":
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        SPECIFIC_SEEDS = None   # None = todos los disponibles
        SUBSET_SIZE    = 1500
        N_REPEATS      = 5
        print("🔬 Google Colab mode")
        importance_dict, recall_dict, df_consolidated = main(
            specific_seeds=SPECIFIC_SEEDS,
            subset_size=SUBSET_SIZE,
            n_repeats=N_REPEATS,
        )
    else:
        import argparse
        parser = argparse.ArgumentParser(description='Feature Importance V2')
        parser.add_argument('--seeds', type=int, nargs='+', default=None)
        parser.add_argument('--subset-size', type=int, default=1500)
        parser.add_argument('--n-repeats', type=int, default=5)
        args = parser.parse_args()
        importance_dict, recall_dict, df_consolidated = main(
            specific_seeds=args.seeds,
            subset_size=args.subset_size,
            n_repeats=args.n_repeats,
        )
