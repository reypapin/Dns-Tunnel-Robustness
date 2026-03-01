# =============================================================================
# STATISTICAL TESTS + LATEX TABLE
# Dos tareas independientes:
#
# TAREA 1 — Tabla LaTeX (lee feature_importance_consolidated.csv)
#   Top 10 features por RF | Feature | Category | RF±std | CNN±std | LSTM±std | LogReg
#   Separador entre Positional y Stateful
#   Importancias negativas se reportan tal cual (son hallazgo, no error)
#
# TAREA 2 — Tests estadísticos (paired t-test + Cohen's d)
#   Para cada modelo: recall en los 9 seeds individuales
#   Comparaciones: cada modelo robusto vs cada modelo fallido
#   Resultado: tabla LaTeX de p-values y Cohen's d
#
# USO EN COLAB:
#   Subir este script y ejecutar como nueva celda.
#   No requiere que feature_importance_v2.py esté en memoria.
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ttest_rel, ttest_ind
from sklearn.metrics import recall_score
from tensorflow import keras

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR   = "/content/drive/MyDrive/Tunnel/CSV_CIC21"
MODEL_DIR  = os.path.join(BASE_DIR, "Models_SOTA_Hybrid")
ARAGAT_DIR = os.path.join(BASE_DIR, "CSV_Generated")
CSV_CONS   = os.path.join(MODEL_DIR, "feature_importance_consolidated.csv")

FEATS_SL = [
    "FQDN_count", "subdomain_length", "upper", "lower", "numeric",
    "entropy", "special", "labels", "labels_max", "labels_average", "len"
]
FEATS_SF = [
    "rr", "A_frequency", "AAAA_frequency", "CNAME_frequency", "TXT_frequency",
    "MX_frequency", "NS_frequency", "NULL_frequency",
    "rr_count", "distinct_ip", "unique_ttl", "total_queries"
]
ALL_FEATS = FEATS_SL + FEATS_SF


# =============================================================================
# TAREA 1 — TABLA LaTeX DESDE CSV CONSOLIDADO
# =============================================================================

def fmt_imp(importance, std, decimals=4):
    """
    Formatea importance ± std para LaTeX.
    Negativo → reportar tal cual con signo (hallazgo, no error).
    """
    if pd.isna(importance):
        return r"\textemdash"
    sign = "-" if importance < 0 else ""
    imp_str = f"{abs(importance):.{decimals}f}"
    std_str = f"{abs(std):.{decimals}f}" if not pd.isna(std) and std > 0 else "---"
    if std_str == "---":
        return f"{sign}{imp_str}"
    return f"${sign}{imp_str}{{\pm}}{std_str}$"


def generate_latex_table(csv_path, top_n=10, save_path=None):
    """
    Genera tabla LaTeX con top-N features.
    Columnas: Feature | Category | RF (mean±std) | CNN | LSTM | LogReg |coef|
    Separador visual entre Positional y Stateful.
    """
    df = pd.read_csv(csv_path)

    # Top 10 por RF importance
    df_top = df.nlargest(top_n, 'rf_importance').copy()
    df_top = df_top.sort_values('rf_importance', ascending=False).reset_index(drop=True)

    # ── Construir filas ───────────────────────────────────────────────────────
    lines = []

    # Header
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Top-10 features ranked by Random Forest importance.")
    lines.append(r"  Values show mean~$\pm$~std. Negative permutation importance")
    lines.append(r"  (CNN/LSTM) indicates the feature acts as noise for that model.")
    lines.append(r"  \textsuperscript{$\dagger$}Std between individual trees.")
    lines.append(r"  \textsuperscript{$\ddagger$}Std from $n{=}5$ permutation repeats.}")
    lines.append(r"  \label{tab:feature_importance}")
    lines.append(r"  \resizebox{\columnwidth}{!}{%")
    lines.append(r"  \begin{tabular}{llcccc}")
    lines.append(r"  \toprule")
    lines.append(r"  \textbf{Feature} & \textbf{Cat.} & "
                 r"\textbf{RF\textsuperscript{$\dagger$}} & "
                 r"\textbf{CNN\textsuperscript{$\ddagger$}} & "
                 r"\textbf{LSTM\textsuperscript{$\ddagger$}} & "
                 r"\textbf{LogReg $|\beta|$} \\")
    lines.append(r"  \midrule")

    prev_cat = None
    for _, row in df_top.iterrows():
        cat = row['category']

        # Separador entre grupos de categoría
        if prev_cat is not None and cat != prev_cat:
            lines.append(r"  \midrule")
        prev_cat = cat

        feat_name = row['feature'].replace('_', r'\_')
        cat_short = 'Pos.' if cat == 'Positional' else ('Sta.' if cat == 'Stateful' else 'Oth.')

        rf_str   = fmt_imp(row.get('rf_importance'),  row.get('rf_std'))
        cnn_str  = fmt_imp(row.get('cnn_importance'), row.get('cnn_std'))
        lstm_str = fmt_imp(row.get('lstm_importance'),row.get('lstm_std'))
        lr_str   = fmt_imp(row.get('logreg_abs_coef'),row.get('logreg_std'), decimals=4)

        lines.append(f"  \\texttt{{{feat_name}}} & {cat_short} & "
                     f"{rf_str} & {cnn_str} & {lstm_str} & {lr_str} \\\\")

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}}")
    lines.append(r"  \end{table}")

    table_str = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"   ✅ Tabla LaTeX guardada: {save_path}")

    print("\n" + "="*70)
    print("TABLA LaTeX — copiar al paper:")
    print("="*70)
    print(table_str)
    return table_str


# =============================================================================
# TAREA 2 — RECALL POR SEED (evaluación individual)
# =============================================================================

def load_single_seed_data(seed, scaler):
    """Carga y escala los datos de UN seed específico."""
    path_sl = os.path.join(ARAGAT_DIR,
                           f"stateless_features-bridge.pcap_{seed}.csv")
    path_sf = os.path.join(ARAGAT_DIR,
                           f"stateful_features-bridge.pcap_{seed}.csv")
    df_sl = pd.read_csv(path_sl)
    df_sf = pd.read_csv(path_sf)

    for c in FEATS_SL:
        df_sl[c] = pd.to_numeric(df_sl.get(c, 0), errors='coerce').fillna(0)
    for c in FEATS_SF:
        df_sf[c] = pd.to_numeric(df_sf.get(c, 0), errors='coerce').fillna(0)

    agg_dict = {f: 'mean' for f in FEATS_SL if f in df_sl.columns}
    if 'label' in df_sl.columns:
        agg_dict['label'] = 'first'
    df_sl_agg = df_sl.groupby('tunnel_id').agg(agg_dict).reset_index()
    df_sl_agg['tunnel_id'] = df_sl_agg['tunnel_id'].astype(str) + f"_s{seed}"
    df_sf['tunnel_id'] = df_sf['tunnel_id'].astype(str) + f"_s{seed}"

    df = pd.merge(df_sl_agg, df_sf, on='tunnel_id', how='inner', suffixes=('', '_sf'))
    df = df.reindex(columns=ALL_FEATS + ['label'], fill_value=0)
    df['label'] = df['label'].fillna(1).astype(int)
    df[ALL_FEATS] = df[ALL_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0)

    X = scaler.transform(df[ALL_FEATS].values)
    y = df['label'].values
    return X, y


def compute_per_seed_recalls(seeds, models_dict, scaler):
    """
    Para cada modelo y cada seed: calcula recall de la clase ataque.

    Returns:
        recalls: dict {model_name: np.array(n_seeds)}
    """
    recalls = {m: [] for m in models_dict}

    for seed in seeds:
        print(f"   Evaluando seed {seed}...", end=' ')
        X_seed, y_seed = load_single_seed_data(seed, scaler)
        X_3d = X_seed.reshape(X_seed.shape[0], X_seed.shape[1], 1)

        for model_name, model in models_dict.items():
            if model is None:
                recalls[model_name].append(np.nan)
                continue
            if model_name in ('CNN', 'LSTM'):
                preds = (model.predict(X_3d, verbose=0) > 0.5).astype(int).flatten()
            else:
                # Tree models usan X sin escalar (mismo scaler que entrenamiento)
                # LogReg usa X escalado
                preds = model.predict(X_seed if model_name != 'LogisticRegression'
                                      else X_seed)
            recalls[model_name].append(recall_score(y_seed, preds, zero_division=0))

        recalled = {m: f"{recalls[m][-1]:.3f}" for m in models_dict if models_dict[m]}
        print(" | ".join(f"{m}={v}" for m, v in recalled.items()))

    return {m: np.array(v) for m, v in recalls.items()}


# =============================================================================
# TAREA 2 — TESTS ESTADÍSTICOS (paired t-test + Cohen's d)
# =============================================================================

def cohens_d_paired(a, b):
    """Cohen's d para muestras pareadas: d = mean(diff) / std(diff)."""
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def run_statistical_tests(recalls_dict, save_path=None):
    """
    Comparaciones pareadas entre todos los pares de modelos.
    Test: ttest_rel (pareado, más poderoso porque son los mismos seeds).
    Efecto: Cohen's d pareado.

    También compara WITHIN cada modelo: el recall vs 0 (baseline).
    """
    models    = [m for m, v in recalls_dict.items() if not np.all(np.isnan(v))]
    n_seeds   = len(next(iter(recalls_dict.values())))

    print("\n" + "="*70)
    print(f"TESTS ESTADÍSTICOS — {n_seeds} seeds (paired t-test)")
    print("="*70)

    print("\n📊 Recall por seed (media ± std):")
    for m in models:
        v = recalls_dict[m]
        print(f"   {m:22s}: {v.mean():.4f} ± {v.std():.4f}  "
              f"[{v.min():.3f} – {v.max():.3f}]")

    # ── Comparaciones entre pares ────────────────────────────────────────────
    results = []
    print(f"\n📊 Comparaciones pareadas (α=0.05):")
    print(f"   {'Model A':22s}  {'Model B':22s}  {'Δmean':>8}  {'t':>7}  {'p':>8}  {'d':>6}  Sig")
    print("   " + "─"*80)

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a_name, b_name = models[i], models[j]
            a = recalls_dict[a_name]
            b = recalls_dict[b_name]

            # Manejo de NaN
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 3:
                continue
            a_c, b_c = a[mask], b[mask]

            delta = a_c.mean() - b_c.mean()

            # Si ambas distribuciones tienen varianza cero (ej. XGB=0 vs LGB=0)
            # el t-test devuelve NaN — detectar antes de llamarlo
            if np.var(a_c - b_c) < 1e-12:
                t_stat, p_val, d, sig = np.nan, np.nan, 0.0, "†"
            else:
                t_stat, p_val = ttest_rel(a_c, b_c)
                d             = cohens_d_paired(a_c, b_c)
                sig           = "***" if p_val < 0.001 else ("**" if p_val < 0.01
                                else ("*" if p_val < 0.05 else "ns"))

            results.append({
                'model_a'  : a_name,
                'model_b'  : b_name,
                'mean_a'   : a_c.mean(),
                'mean_b'   : b_c.mean(),
                'delta'    : delta,
                't_stat'   : t_stat,
                'p_value'  : p_val,
                'cohens_d' : d,
                'sig'      : sig,
                'n'        : int(mask.sum()),
            })
            t_str = f"{t_stat:7.3f}" if not np.isnan(t_stat) else "    ---"
            p_str = f"{p_val:8.4f}" if not np.isnan(p_val) else "     ---"
            print(f"   {a_name:22s}  {b_name:22s}  "
                  f"{delta:+8.4f}  {t_str}  {p_str}  {d:6.2f}  {sig}")

    df_stats = pd.DataFrame(results)

    # ── Tabla LaTeX de tests ──────────────────────────────────────────────────
    latex_stats = _build_stats_latex_table(df_stats, n_seeds)
    print("\n" + "="*70)
    print("TABLA LaTeX — Tests estadísticos (copiar al paper):")
    print("="*70)
    print(latex_stats)

    if save_path:
        df_stats.to_csv(save_path, index=False)
        print(f"\n   ✅ CSV guardado: {save_path}")
        tex_path = save_path.replace('.csv', '.tex')
        with open(tex_path, 'w') as f:
            f.write(latex_stats)
        print(f"   ✅ LaTeX guardado: {tex_path}")

    return df_stats


def _build_stats_latex_table(df_stats, n_seeds):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Pairwise paired $t$-tests on per-seed recall")
    lines.append(r"  ($n=" + str(n_seeds) + r"$ seeds, Mutant Payload).")
    lines.append(r"  $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$, ns: not significant.}")
    lines.append(r"  Cohen's~$d$ for paired differences: small $|d|{<}0.5$,")
    lines.append(r"  medium $0.5{\le}|d|{<}0.8$, large $|d|{\ge}0.8$.}")
    lines.append(r"  \label{tab:stat_tests}")
    lines.append(r"  \begin{tabular}{llcccc}")
    lines.append(r"  \toprule")
    lines.append(r"  \textbf{Model A} & \textbf{Model B} & "
                 r"$\overline{r}_A$ & $\overline{r}_B$ & "
                 r"$p$-value & Cohen's~$d$ \\")
    lines.append(r"  \midrule")

    for _, row in df_stats.iterrows():
        sig = row['sig']
        ma  = row['model_a'].replace('_', r'\_').replace('LogisticRegression', 'LogReg')
        mb  = row['model_b'].replace('_', r'\_').replace('LogisticRegression', 'LogReg')

        # NaN p-value: ambos modelos tienen recall idéntico (cero varianza)
        if pd.isna(row['p_value']):
            p_cell = r"\multicolumn{1}{c}{---\textsuperscript{$\dagger$}}"
        elif row['p_value'] < 0.0001:
            p_cell = f"$<0.0001^{{{sig}}}$"
        else:
            p_cell = f"${row['p_value']:.4f}^{{{sig}}}$"

        lines.append(
            f"  {ma} & {mb} & "
            f"{row['mean_a']:.4f} & {row['mean_b']:.4f} & "
            f"{p_cell} & ${row['cohens_d']:+.2f}$ \\\\"
        )

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# SCENARIO B vs C — Caída de recall por modelo (CIC-2021 → Mutant Payload)
# =============================================================================

def load_cic_attack_data(scaler, n_bootstrap=9, bootstrap_size=1500, random_state=42):
    """
    Carga los flows de ATAQUE del dataset CIC-2021 (carpeta Attacks/).
    Combina stateless (media por tunnel) + stateful (fila).
    Genera n_bootstrap subconjuntos aleatorios para emparejamiento con los seeds.

    Returns:
        X_boots  : list of n_bootstrap arrays (bootstrap_size, 23)
        y_boots  : list of n_bootstrap arrays (todos 1 = ataque)
    """
    import glob as _glob

    print(f"\n   🔍 Buscando archivos CIC attack en {BASE_DIR}...")

    def _is_attack(path):
        return os.path.basename(os.path.dirname(path)).lower() == "attacks"

    sf_all = [f for f in _glob.glob(
        os.path.join(BASE_DIR, "**", "stateful_features*.csv"), recursive=True)
        if _is_attack(f)]
    sl_all = [f for f in _glob.glob(
        os.path.join(BASE_DIR, "**", "stateless_features*.csv"), recursive=True)
        if _is_attack(f)]

    print(f"   CIC attack files — stateful: {len(sf_all)}, stateless: {len(sl_all)}")
    if not sf_all:
        raise FileNotFoundError("No CIC attack stateful files found.")

    # Emparejar por nombre de archivo
    sf_map = {os.path.basename(p).replace("stateful", "PLACEHOLDER"): p for p in sf_all}
    sl_map = {os.path.basename(p).replace("stateless", "PLACEHOLDER"): p for p in sl_all}
    paired = [(sf_map[k], sl_map[k]) for k in sorted(set(sf_map) & set(sl_map))]
    if not paired:
        paired = list(zip(sorted(sf_all), sorted(sl_all)))
        print(f"   ⚠️  Emparejamiento por nombre fallido, usando orden ({len(paired)} pares).")
    else:
        print(f"   ✅ {len(paired)} pares emparejados.")

    rows_X = []
    for sf_path, sl_path in paired:
        try:
            df_sf = pd.read_csv(sf_path)
            df_sl = pd.read_csv(sl_path)

            # FIX: verificar existencia antes de to_numeric (evita 'int'.fillna())
            for c in FEATS_SL:
                if c in df_sl.columns:
                    df_sl[c] = pd.to_numeric(df_sl[c], errors='coerce').fillna(0)
                else:
                    df_sl[c] = 0.0
            for c in FEATS_SF:
                if c in df_sf.columns:
                    df_sf[c] = pd.to_numeric(df_sf[c], errors='coerce').fillna(0)
                else:
                    df_sf[c] = 0.0

            # Calcular total_queries
            freq_cols = [c for c in df_sf.columns if '_frequency' in c]
            if freq_cols:
                df_sf['total_queries'] = df_sf[freq_cols].sum(axis=1)
            elif 'rr_count' in df_sf.columns:
                df_sf['total_queries'] = df_sf['rr_count']
            else:
                df_sf['total_queries'] = 1

            # Inyectar medias SL en cada fila SF
            for feat in FEATS_SL:
                df_sf[feat] = df_sl[feat].mean() if feat in df_sl.columns else 0.0

            rows_X.append(df_sf[ALL_FEATS].fillna(0).values)
        except Exception as e:
            print(f"   ⚠️  Error en {os.path.basename(sf_path)}: {e}")
            continue

    if not rows_X:
        raise ValueError("No CIC rows loaded.")

    X_all = np.vstack(rows_X)
    X_all = scaler.transform(X_all)
    y_all = np.ones(len(X_all), dtype=int)   # todos son ataques
    print(f"   ✅ CIC attack total: {len(X_all):,} flows")

    # Generar n_bootstrap subconjuntos (matching size de seeds de Mutant)
    rng = np.random.default_rng(random_state)
    X_boots, y_boots = [], []
    for i in range(n_bootstrap):
        idx = rng.choice(len(X_all), size=min(bootstrap_size, len(X_all)), replace=False)
        X_boots.append(X_all[idx])
        y_boots.append(y_all[idx])

    return X_boots, y_boots


def compare_scenario_b_vs_c(models_dict, recalls_c, scaler,
                             n_bootstrap=9, bootstrap_size=1500,
                             save_path=None):
    """
    Scenario B: recall en CIC-2021 ataque (bootstrap)
    Scenario C: recall en Mutant Payload (por seed) — ya calculado

    Para cada modelo: paired t-test B vs C + Cohen's d.
    Pregunta: "¿Es la caída de recall al mutar el ataque estadísticamente significativa?"
    """
    print("\n" + "="*70)
    print("SCENARIO B vs C — Caída de recall por modelo (CIC-2021 → Mutant)")
    print("="*70)

    try:
        X_boots, y_boots = load_cic_attack_data(
            scaler, n_bootstrap=n_bootstrap, bootstrap_size=bootstrap_size)
    except Exception as e:
        print(f"   ❌ No se pudo cargar CIC data: {e}")
        return None

    # Calcular recalls B (CIC bootstrap) para cada modelo
    recalls_b = {m: [] for m in models_dict}
    for i, (X_b, y_b) in enumerate(zip(X_boots, y_boots)):
        X_b3 = X_b.reshape(X_b.shape[0], X_b.shape[1], 1)
        for model_name, model in models_dict.items():
            if model is None:
                recalls_b[model_name].append(np.nan)
                continue
            if model_name in ('CNN', 'LSTM'):
                preds = (model.predict(X_b3, verbose=0) > 0.5).astype(int).flatten()
            else:
                preds = model.predict(X_b)
            recalls_b[model_name].append(recall_score(y_b, preds, zero_division=0))

    recalls_b = {m: np.array(v) for m, v in recalls_b.items()}

    # Tabla comparativa y tests
    results = []
    print(f"\n   {'Model':22s}  {'B (CIC)':>10}  {'C (Mutant)':>12}  {'Δ':>8}  {'t':>7}  {'p':>8}  {'d':>6}  Sig")
    print("   " + "─"*85)

    for model_name in models_dict:
        b = recalls_b[model_name]
        c = np.array(recalls_c.get(model_name, [np.nan] * n_bootstrap))
        mask = ~(np.isnan(b) | np.isnan(c))
        if mask.sum() < 3:
            continue
        b_c, c_c = b[mask], c[mask]
        delta = b_c.mean() - c_c.mean()   # positivo = B > C (caída esperada)

        if np.var(b_c - c_c) < 1e-12:
            t_stat, p_val, d, sig = np.nan, np.nan, 0.0, "†"
        else:
            t_stat, p_val = ttest_rel(b_c, c_c)
            d   = cohens_d_paired(b_c, c_c)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01
                  else ("*" if p_val < 0.05 else "ns"))

        results.append({'model': model_name,
                        'recall_B': b_c.mean(), 'recall_C': c_c.mean(),
                        'delta': delta, 't_stat': t_stat, 'p_value': p_val,
                        'cohens_d': d, 'sig': sig})
        t_str = f"{t_stat:7.3f}" if not np.isnan(t_stat) else "    ---"
        p_str = f"{p_val:8.4f}" if not np.isnan(p_val) else "     ---"
        print(f"   {model_name:22s}  {b_c.mean():10.4f}  {c_c.mean():12.4f}  "
              f"{delta:+8.4f}  {t_str}  {p_str}  {d:6.2f}  {sig}")

    df_bc = pd.DataFrame(results)

    # Tabla LaTeX
    latex_bc = _build_bc_latex_table(df_bc)
    print("\n" + "="*70)
    print("TABLA LaTeX — Scenario B vs C (copiar al paper):")
    print("="*70)
    print(latex_bc)

    if save_path:
        df_bc.to_csv(save_path, index=False)
        tex_path = save_path.replace('.csv', '.tex')
        with open(tex_path, 'w') as f:
            f.write(latex_bc)
        print(f"\n   ✅ Guardado: {save_path}  +  {tex_path}")

    return df_bc


def _build_bc_latex_table(df_bc):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Statistical significance of recall drop from")
    lines.append(r"  Scenario~B (standard CIC-2021 attack, bootstrap $n{=}9$)")
    lines.append(r"  to Scenario~C (Mutant Payload, $n{=}9$ seeds).")
    lines.append(r"  Paired $t$-test; $\dagger$indeterminate (zero variance).")
    lines.append(r"  $^{***}p{<}0.001$.}")
    lines.append(r"  \label{tab:scenario_bc}")
    lines.append(r"  \begin{tabular}{lcccccc}")
    lines.append(r"  \toprule")
    lines.append(r"  \textbf{Model} & $\overline{r}_B$ & $\overline{r}_C$ & "
                 r"$\Delta r$ & $t$ & $p$-value & $d$ \\")
    lines.append(r"  \midrule")
    for _, row in df_bc.iterrows():
        mname = row['model'].replace('LogisticRegression', 'LogReg')
        if pd.isna(row['p_value']):
            p_cell = r"---\textsuperscript{$\dagger$}"
            t_cell = "---"
        else:
            p_cell = (f"$<0.0001^{{{row['sig']}}}$"
                      if row['p_value'] < 0.0001
                      else f"${row['p_value']:.4f}^{{{row['sig']}}}$")
            t_cell = f"${row['t_stat']:.2f}$"
        lines.append(
            f"  {mname} & {row['recall_B']:.4f} & {row['recall_C']:.4f} & "
            f"${row['delta']:+.4f}$ & {t_cell} & {p_cell} & ${row['cohens_d']:+.2f}$ \\\\"
        )
    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

def run_all(seeds=None):
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS + LaTeX TABLES")
    print("="*70)

    # ── TAREA 1: Tabla de feature importance ──────────────────────────────────
    if not os.path.exists(CSV_CONS):
        print(f"⚠️  CSV consolidado no encontrado: {CSV_CONS}")
        print("   Ejecuta feature_importance_v2.py primero.")
    else:
        generate_latex_table(
            CSV_CONS,
            top_n=10,
            save_path=os.path.join(MODEL_DIR, "table_feature_importance.tex")
        )

    # ── TAREA 2: Tests estadísticos ────────────────────────────────────────────
    print("\n" + "="*70)
    print("CARGANDO MODELOS PARA TESTS POR SEED...")
    print("="*70)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_sota.joblib"))

    # Detectar seeds disponibles automáticamente
    if seeds is None:
        pattern = re.compile(r'stateless_features-bridge\.pcap_(\d+)\.csv')
        avail_files = os.listdir(ARAGAT_DIR)
        seeds = sorted({int(m.group(1))
                        for f in avail_files
                        if (m := pattern.match(f)) and
                        os.path.exists(os.path.join(
                            ARAGAT_DIR,
                            f"stateful_features-bridge.pcap_{m.group(1)}.csv"))})
    print(f"   Seeds: {seeds}")

    # Cargar modelos
    models_dict = {}
    model_files = {
        'RandomForest'      : 'RandomForest_sota.joblib',
        'XGBoost'           : 'XGBoost_sota.joblib',
        'LightGBM'          : 'LightGBM_sota.joblib',
        'LogisticRegression': 'LogisticRegression_sota.joblib',
        'CNN'               : 'CNN_sota.keras',
        'LSTM'              : 'LSTM_sota.keras',
    }
    for name, fname in model_files.items():
        fpath = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fpath):
            print(f"   ⚠️  {name} no encontrado, omitido.")
            models_dict[name] = None
            continue
        if fname.endswith('.keras'):
            models_dict[name] = keras.models.load_model(fpath)
        else:
            models_dict[name] = joblib.load(fpath)
        print(f"   ✅ {name} cargado.")

    # Recall por seed
    print(f"\n📊 Calculando recall por seed ({len(seeds)} seeds × {len(models_dict)} modelos)...")
    recalls = compute_per_seed_recalls(seeds, models_dict, scaler)

    # Guardar recalls en CSV (útil para inspección manual)
    df_recalls = pd.DataFrame(recalls, index=[f"seed_{s}" for s in seeds])
    recalls_csv = os.path.join(MODEL_DIR, "recalls_per_seed.csv")
    df_recalls.to_csv(recalls_csv)
    print(f"\n   ✅ Recalls por seed guardados: {recalls_csv}")
    print("\n" + df_recalls.to_string())

    # Tests estadísticos modelo vs modelo
    df_stats = run_statistical_tests(
        recalls,
        save_path=os.path.join(MODEL_DIR, "statistical_tests.csv")
    )

    # ── SCENARIO B vs C: caída de recall CIC → Mutant por modelo ──────────────
    df_bc = compare_scenario_b_vs_c(
        models_dict    = models_dict,
        recalls_c      = recalls,           # Scenario C = lo que ya calculamos
        scaler         = scaler,
        n_bootstrap    = len(seeds),        # mismo n que seeds de Mutant
        bootstrap_size = 1500,
        save_path      = os.path.join(MODEL_DIR, "scenario_b_vs_c.csv")
    )

    return df_recalls, df_stats, df_bc


# =============================================================================
# EJECUCIÓN COLAB
# =============================================================================

if __name__ == "__main__":
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    df_recalls, df_stats, df_bc = run_all()
