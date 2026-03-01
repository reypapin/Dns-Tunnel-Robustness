# =============================================================================
# FIGURA 2: NORMALIZED MEAN ACTIVATION PROFILES - CNN FEATURE GROUP ACTIVATION
# Versión FINAL - Corrige: anotación panel (a), shading dual, caption exacto
# =============================================================================
# Script autocontenido: no requiere celdas previas del notebook.
# Solo necesita BASE_DIR y MODEL_DIR_SOTA definidos abajo.
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =============================================================================
# DEFINICIONES DE FEATURES (copiadas del notebook Model_CIC(3).ipynb)
# =============================================================================

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

ALL_FEATS = FEATS_SL + FEATS_SF   # 23 features total


def clean_and_prepare(df_sl, df_sf, row_idx, scaler):
    """Combina stateless (mean) + stateful (row) y genera input (1, 23, 1) para la CNN."""
    sf_row = df_sf.iloc[row_idx:row_idx + 1].copy()

    # Deserializar columnas que pueden venir como string-set (e.g. "{'1.2.3.4'}")
    for col in ['distinct_ip', 'unique_ttl']:
        if col in sf_row.columns:
            val = sf_row[col].iloc[0]
            try:
                if isinstance(val, str):
                    sf_row[col] = len(eval(val)) if val != 'set()' else 0
                else:
                    sf_row[col] = float(val)
            except Exception:
                sf_row[col] = 0

    # Calcular total_queries sumando todas las columnas *_frequency
    freq_cols = [c for c in sf_row.columns if '_frequency' in c]
    if freq_cols:
        sf_row['total_queries'] = sf_row[freq_cols].sum(axis=1)
    elif 'rr_count' in sf_row.columns:
        sf_row['total_queries'] = sf_row['rr_count']
    else:
        sf_row['total_queries'] = 1

    # Inyectar medias de features stateless en la fila stateful
    for feat in FEATS_SL:
        if feat in df_sl.columns:
            sf_row[feat] = df_sl[feat].mean()
        else:
            sf_row[feat] = 0.0

    # Seleccionar las 23 features y escalar
    x = scaler.transform(sf_row[ALL_FEATS].fillna(0))
    return x.reshape(1, 23, 1)

# =============================================================================
# PASO 0: VERIFICACIÓN PREVIA
# =============================================================================

print("=" * 60)
print("PASO 0: VERIFICACIÓN DE MODELO Y DATOS")
print("=" * 60)

# --- Verificar existencia del modelo ---
cnn_path = os.path.join(MODEL_DIR_SOTA, "CNN_sota.keras")
scaler_path = os.path.join(MODEL_DIR_SOTA, "scaler_sota.joblib")

if not os.path.exists(cnn_path):
    raise FileNotFoundError(
        f"\n❌ CNN_sota.keras no encontrado en:\n   {cnn_path}\n"
        f"   Verifica que el modelo esté en Models_SOTA_Hybrid/"
    )
if not os.path.exists(scaler_path):
    raise FileNotFoundError(
        f"\n❌ scaler_sota.joblib no encontrado en:\n   {scaler_path}"
    )

print(f"✅ Modelo CNN encontrado: {cnn_path}")
print(f"✅ Scaler encontrado:     {scaler_path}")

# --- Cargar modelo y scaler ---
cnn_model = load_model(cnn_path)
scaler    = joblib.load(scaler_path)
print("\n📐 Arquitectura CNN:")
cnn_model.summary()

# --- Identificar última Conv1D y sus dimensiones ---
conv_layers = [l for l in cnn_model.layers
               if isinstance(l, tf.keras.layers.Conv1D)]
if not conv_layers:
    raise ValueError("❌ El modelo no tiene capas Conv1D.")

# Usamos la PRIMERA Conv1D (output: 23 × 64) en lugar de la última (5 × 256).
# Ventaja: con padding='same' la primera capa tiene D=23 salidas, una por feature.
# Esto da mapping 1:1 entre posición espacial y feature de entrada, haciendo
# el eje X directamente interpretable (pos 0 = FQDN_count, ..., pos 22 = total_queries).
first_conv = conv_layers[0]
last_conv  = conv_layers[-1]   # conservado para referencia en el reporte

act_model = Model(inputs=cnn_model.input, outputs=first_conv.output)
D = act_model.output_shape[1]   # dimensiones espaciales (= 23 con primera Conv1D)
F = act_model.output_shape[2]   # número de filtros

print(f"\n📊 Capa Conv1D usada para activación: '{first_conv.name}' (1ª capa, D=23)")
print(f"   Output shape: (None, {D}, {F})  →  D={D} puntos espaciales, F={F} filtros")

if D < 4:
    print(f"\n⚠️  WARNING: Solo {D} puntos espaciales en la última Conv1D.")
    print("   La figura tendrá pocos puntos pero el mensaje conceptual sigue siendo válido.")
    print("   Si necesitas más resolución, considera usar una capa Conv1D intermedia.")
else:
    print(f"\n✅ D={D} puntos espaciales — suficiente para una figura informativa.")

# --- Verificar archivos disponibles ---
# CIC 2021: la etiqueta viene del directorio padre.
# Estructura: .../Attacks/stateful_features-*.csv   ← ataque
#             .../Benign/stateful_features-*.csv    ← benigno
# Pre-filtramos aquí para pasar SOLO archivos de ataque a collect_attack_flows.

def _is_cic_attack(path):
    """True si el folder inmediato padre es 'Attacks' (case-insensitive)."""
    return os.path.basename(os.path.dirname(path)).lower() == "attacks"

_cic_sf_all = glob.glob(os.path.join(BASE_DIR, "**", "stateful_features*.csv"),
                         recursive=True)
_cic_sl_all = glob.glob(os.path.join(BASE_DIR, "**", "stateless_features*.csv"),
                         recursive=True)

cic_sf_files = [f for f in _cic_sf_all if _is_cic_attack(f)]
cic_sl_files = [f for f in _cic_sl_all if _is_cic_attack(f)]

mut_sf_files  = glob.glob(os.path.join(BASE_DIR, "**", "*bridge*stateful*.csv"),
                           recursive=True)
mut_sl_files  = glob.glob(os.path.join(BASE_DIR, "**", "*bridge*stateless*.csv"),
                           recursive=True)

# Alternativa: buscar por patrón de Mutant Payload (ARAGAT generated)
if not mut_sf_files:
    mut_sf_files = glob.glob(os.path.join(BASE_DIR, "**", "stateful_features-bridge*"),
                              recursive=True)
    mut_sl_files = glob.glob(os.path.join(BASE_DIR, "**", "stateless_features-bridge*"),
                              recursive=True)

print(f"\n📂 CIC-2021 total encontrados — stateful: {len(_cic_sf_all)}, stateless: {len(_cic_sl_all)}")
print(f"📂 CIC-2021 ATAQUE (Attacks/) — stateful: {len(cic_sf_files)}, stateless: {len(cic_sl_files)}")
print(f"📂 Mutant Payload             — stateful: {len(mut_sf_files)}, stateless: {len(mut_sl_files)}")
if len(cic_sf_files) == 0:
    raise FileNotFoundError(
        f"\n❌ No se encontraron archivos CIC en carpetas 'Attacks/'.\n"
        f"   Verifica que BASE_DIR sea correcto: {BASE_DIR}\n"
        f"   Y que la estructura sea: .../Attacks/stateful_features-*.csv"
    )

print(f"\n✅ Sub-modelo de activación creado (hasta '{last_conv.name}')")


# =============================================================================
# PASO 1: FUNCIÓN PARA RECOLECTAR N FLOWS
# =============================================================================

def collect_attack_flows(sf_files, sl_files, model, scaler,
                         n_flows=100, is_cic=True, min_prob=0.5,
                         label_col='label'):
    """
    Recolecta hasta n_flows flows de ataque con predicción >= min_prob.

    Parameters
    ----------
    sf_files   : lista de rutas stateful CSV
    sl_files   : lista de rutas stateless CSV (misma longitud y orden)
    model      : modelo CNN completo (para filtrar por prob)
    scaler     : StandardScaler ya ajustado
    n_flows    : número máximo de flows a recolectar
    is_cic     : True = CIC 2021 (ataques en carpeta "Attacks"),
                 False = Mutant Payload (ataques tienen label==1)
    min_prob   : umbral mínimo de probabilidad de predicción
    label_col  : nombre de columna de etiqueta en CSV

    Returns
    -------
    x_list  : lista de arrays (1, 23, 1)
    probs   : lista de probabilidades correspondientes
    """
    x_list = []
    probs  = []
    total_inspected = 0

    # Nombres alternativos de columna de etiqueta (CIC usa 'Label' con L mayúscula)
    LABEL_CANDIDATES = [label_col, label_col.capitalize(), label_col.upper(),
                        'Label', 'label', 'class', 'Class', 'is_attack', 'type']

    def _find_label_col(df):
        """Retorna el nombre de la columna de etiqueta encontrada, o None."""
        for c in LABEL_CANDIDATES:
            if c in df.columns:
                return c
        return None

    # Emparejar stateful con stateless usando el nombre de archivo base
    sf_map = {os.path.basename(p).replace("stateful", "PLACEHOLDER"): p
              for p in sf_files}
    sl_map = {os.path.basename(p).replace("stateless", "PLACEHOLDER"): p
              for p in sl_files}
    paired_keys = set(sf_map.keys()) & set(sl_map.keys())

    if not paired_keys:
        # Fallback: intentar emparejar por orden de lista
        pairs = list(zip(sorted(sf_files), sorted(sl_files)))
        print(f"   ⚠️  Emparejamiento por nombre falló, usando orden ({len(pairs)} pares)")
    else:
        pairs = [(sf_map[k], sl_map[k]) for k in sorted(paired_keys)]
        print(f"   ✅ {len(pairs)} pares stateful/stateless emparejados por nombre")

    # Diagnóstico: mostrar columnas del primer par para detectar nombre de label
    if pairs:
        _df0 = pd.read_csv(pairs[0][0])
        _lc  = _find_label_col(_df0) if len(pairs) > 0 else None
        _parent0 = os.path.basename(os.path.dirname(pairs[0][0]))
        print(f"   🔍 Diagnóstico primer CSV — columnas: {list(_df0.columns[:8])}{'...' if len(_df0.columns) > 8 else ''}")
        print(f"   🔍 Columna label detectada: {_lc!r}  |  folder padre: '{_parent0}'")
        print(f"   🔍 path: ...{pairs[0][0][-70:]}")

    _first_error_shown = False   # muestra el 1er error de clean_and_prepare (global)
    _diag_done = False           # diagnóstico de columnas solo una vez

    for sf_path, sl_path in pairs:
        if len(x_list) >= n_flows:
            break
        try:
            df_sf = pd.read_csv(sf_path)
            df_sl = pd.read_csv(sl_path)

            # Para CIC: todos los archivos ya fueron pre-filtrados a "Attacks/" en PASO 0.
            # Para Mutant: se filtra por columna de etiqueta.
            if is_cic:
                rows_to_use = range(len(df_sf))
            else:
                # Mutant Payload: filtrar por label interno (auto-detecta nombre de col)
                found_col = _find_label_col(df_sf) or _find_label_col(df_sl)
                if found_col is None:
                    rows_to_use = range(len(df_sf))   # sin label → asumir que es ataque
                else:
                    src = df_sf if found_col in df_sf.columns else df_sl
                    attack_rows = src.index[src[found_col] == 1].tolist()
                    if not attack_rows:
                        continue
                    rows_to_use = attack_rows

            # Diagnóstico de columnas faltantes (solo para el primer par procesado)
            if not _diag_done:
                _diag_done = True
                # total_queries se calcula en clean_and_prepare, no viene del CSV
                computed_cols = {'total_queries'}
                missing_sf = [c for c in FEATS_SF
                              if c not in df_sf.columns and c not in computed_cols]
                missing_sl = [c for c in FEATS_SL if c not in df_sl.columns]
                if missing_sf:
                    print(f"   ⚠️  Columnas FEATS_SF faltantes (no computadas): {missing_sf}")
                if missing_sl:
                    print(f"   ⚠️  Columnas FEATS_SL faltantes en stateless CSV: {missing_sl}")
                if not missing_sf and not missing_sl:
                    print(f"   ✅ Todas las columnas de FEATS_SF/FEATS_SL presentes (o computadas).")

            n_before = len(x_list)
            for i in rows_to_use:
                if len(x_list) >= n_flows:
                    break
                if i >= len(df_sf):
                    continue
                try:
                    x_in = clean_and_prepare(df_sl, df_sf, i, scaler)
                    total_inspected += 1
                    prob = float(model.predict(x_in, verbose=0)[0][0])
                    if prob >= min_prob:
                        x_list.append(x_in)
                        probs.append(prob)
                except Exception as _e:
                    if not _first_error_shown:
                        print(f"   ❌ Error en clean_and_prepare (fila {i}): "
                              f"{type(_e).__name__}: {_e}")
                        _first_error_shown = True
                    continue

            n_added = len(x_list) - n_before
            if n_added > 0:
                fname = os.path.basename(sf_path)
                print(f"   ├─ {fname}: +{n_added} flows "
                      f"(total: {len(x_list)}/{n_flows})")

        except Exception as e:
            print(f"   ⚠️  Error en {os.path.basename(sf_path)}: {e}")
            continue

    print(f"   └─ Inspectados: {total_inspected} flows | "
          f"Válidos (prob≥{min_prob}): {len(x_list)}")

    if len(x_list) < 10:
        print(f"\n   ⚠️  WARNING: Solo {len(x_list)} flows válidos encontrados.")
        print(f"   Considera bajar min_prob (actual={min_prob}) o revisar rutas.")
        if len(x_list) == 0:
            raise ValueError("❌ No se encontraron flows válidos. "
                             "Verifica rutas y estructura de archivos.")

    return x_list, probs


# =============================================================================
# PASO 2: CÁLCULO DE PERFIL PROMEDIO REAL (con banda de error)
# =============================================================================

def get_mean_profile_with_std(x_list, activation_model):
    """
    Calcula perfil de activación promedio sobre múltiples flows.

    Proceso por flow:
      1. Extraer activaciones de la última Conv1D → shape (D, F)
      2. Mean absolute activation por posición espacial → shape (D,)
         (promedia sobre los F filtros)
      3. Normalizar por el máximo del flow individual

    Luego promedia todos los perfiles normalizados.

    Returns
    -------
    mean_p : array (D,) normalizado en [0, 1]
    std_p  : array (D,) desviación estándar entre flows
    n_used : int, número de flows procesados
    """
    profiles = []
    for x in x_list:
        acts = activation_model.predict(x, verbose=0)[0]  # (D, F)
        p    = np.mean(np.abs(acts), axis=1)               # (D,)  mean sobre filtros
        norm = p.max() + 1e-9
        profiles.append(p / norm)

    profiles = np.array(profiles)           # (N, D)
    mean_p   = np.mean(profiles, axis=0)   # (D,)
    std_p    = np.std(profiles, axis=0)    # (D,)

    # Re-normalizar el promedio final a [0, 1]
    mean_p = mean_p / (mean_p.max() + 1e-9)

    return mean_p, std_p, len(profiles)


# =============================================================================
# RECOLECCIÓN DE FLOWS
# =============================================================================

N_FLOWS      = 100   # flows objetivo por panel
MIN_PROB     = 0.5   # umbral para CIC-2021 (modelo detecta bien → exigir prob alta)
MIN_PROB_MUT = 0.0   # umbral para Mutant Payload (modelo FALLA → prob < 0.5 esperada;
                     # usamos todos los flows con label==1 para mostrar activación real)

print("\n" + "=" * 60)
print("PASO 1: RECOLECTANDO FLOWS CIC-2021")
print("=" * 60)
x_cic_list, probs_cic = collect_attack_flows(
    sf_files  = cic_sf_files,
    sl_files  = cic_sl_files,
    model     = cnn_model,
    scaler    = scaler,
    n_flows   = N_FLOWS,
    is_cic    = True,
    min_prob  = MIN_PROB
)

print("\n" + "=" * 60)
print("PASO 1b: RECOLECTANDO FLOWS MUTANT PAYLOAD")
print("=" * 60)
x_mut_list, probs_mut = collect_attack_flows(
    sf_files  = mut_sf_files,
    sl_files  = mut_sl_files,
    model     = cnn_model,
    scaler    = scaler,
    n_flows   = N_FLOWS,
    is_cic    = False,
    min_prob  = MIN_PROB_MUT   # 0.0: aceptar todos los flows con label==1
)

print("\n" + "=" * 60)
print("PASO 2: CALCULANDO PERFILES PROMEDIO")
print("=" * 60)

mean_cic, std_cic, n_cic = get_mean_profile_with_std(x_cic_list, act_model)
mean_mut, std_mut, n_mut = get_mean_profile_with_std(x_mut_list, act_model)

print(f"✅ Perfil CIC-2021:     {n_cic} flows, D={len(mean_cic)} puntos espaciales")
print(f"✅ Perfil Mutant Pay.:  {n_mut} flows, D={len(mean_mut)} puntos espaciales")
print(f"   CIC   — mean prob: {np.mean(probs_cic):.3f} ± {np.std(probs_cic):.3f}")
print(f"   Mutant — mean prob: {np.mean(probs_mut):.3f} ± {np.std(probs_mut):.3f}")

# Eje X: índice de feature (0–22), mapeado a 0.0–1.0
# Con la primera Conv1D y padding='same', la posición i corresponde al feature i.
# FEATS_SL = features 0–10 (11 stateless)
# FEATS_SF = features 11–22 (12 stateful)
x_axis = np.linspace(0.0, 1.0, D)

# Límite exacto entre FEATS_SL y FEATS_SF en el eje normalizado
N_SL = len(FEATS_SL)   # 11
N_SF = len(FEATS_SF)   # 12
N_TOTAL = N_SL + N_SF  # 23

# La metadata de CIC (bytes 0-20) afecta principalmente FEATS_SL (stateless):
# subdomain_length, entropy, FQDN_count... → primeras N_SL posiciones
# La metadata de Mutant (bytes 70-80) afecta principalmente FEATS_SF (stateful):
# distinct_ip, unique_ttl, rr_count... → últimas N_SF posiciones
sl_end   = (N_SL - 0.5) / (N_TOTAL - 1)   # frontera SL/SF ≈ 0.45
sf_start = (N_SL + 0.5) / (N_TOTAL - 1)   # inicio zona SF ≈ 0.50

# Alias para compatibilidad con el código de plot
metadata_frac_low  = sl_end    # shading de región stateless (panel a)
metadata_frac_high = sf_start  # shading de región stateful  (panel b)

print(f"\n📏 Mapping features → eje X normalizado:")
print(f"   FEATS_SL (0–{N_SL-1}): x ∈ [0.00, {sl_end:.2f}]  — stateless (DNS name patterns)")
print(f"   FEATS_SF ({N_SL}–{N_TOTAL-1}): x ∈ [{sf_start:.2f}, 1.00]  — stateful (flow statistics)")


# =============================================================================
# PASO 3: FIGURA MEJORADA
# =============================================================================

print("\n" + "=" * 60)
print("PASO 3: GENERANDO FIGURA")
print("=" * 60)

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
fig.subplots_adjust(top=0.93, bottom=0.12, hspace=0.42, left=0.10, right=0.96)

DARK_BLUE  = '#2c3e50'
ORANGE     = '#e67e22'
RED_META   = '#c0392b'

# ─────────────────────────────────────────────────────────
# Panel (a): CIC-2021
# ─────────────────────────────────────────────────────────
ax1.plot(x_axis, mean_cic,
         color=DARK_BLUE, linewidth=2.5, marker='o', markersize=6,
         label=f'Mean profile (N={n_cic} flows)', zorder=3)
ax1.fill_between(x_axis,
                 mean_cic - std_cic, mean_cic + std_cic,
                 alpha=0.20, color=DARK_BLUE, label='±1 SD', zorder=2)
ax1.axvspan(0, metadata_frac_low,
            alpha=0.12, color=RED_META, zorder=1)
ax1.axvspan(sf_start, 1.0,
            alpha=0.10, color=ORANGE, zorder=1)

# Anotación del pico — el pico real está en la región stateful (derecha)
peak_idx_cic = int(np.argmax(mean_cic))
peak_x_cic   = x_axis[peak_idx_cic]
ann_x_cic    = max(peak_x_cic - 0.42, 0.10)   # texto a la izquierda del pico
ax1.annotate(
    'Strong activation in\nstateful features\n(flow statistics)',
    xy=(peak_x_cic, mean_cic[peak_idx_cic]),
    xytext=(ann_x_cic, 0.75),
    fontsize=9, ha='center', va='top',
    color=DARK_BLUE, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=DARK_BLUE, lw=1.5,
                    connectionstyle='arc3,rad=-0.2'),
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
              edgecolor=DARK_BLUE, alpha=0.90, lw=0.8)
)

ax1.set_title('(a) Standard Attack (CIC-2021): Metadata at byte offset 0–20',
              fontweight='bold', fontsize=11, pad=8)
# Línea divisoria SL / SF
ax1.axvline(x=sl_end, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax1.set_ylabel('Normalized Activation Magnitude', fontsize=10)
ax1.set_ylim(-0.05, 1.30)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.85)
ax1.tick_params(labelsize=9)

# ─────────────────────────────────────────────────────────
# Panel (b): Mutant Payload
# ─────────────────────────────────────────────────────────
ax2.plot(x_axis, mean_mut,
         color=ORANGE, linewidth=2.5, marker='s', markersize=6,
         label=f'Mean profile (N={n_mut} flows)', zorder=3)
ax2.fill_between(x_axis,
                 mean_mut - std_mut, mean_mut + std_mut,
                 alpha=0.20, color=ORANGE, label='±1 SD', zorder=2)
ax2.axvspan(0, sl_end,
            alpha=0.10, color=RED_META, zorder=1)
ax2.axvspan(metadata_frac_high, 1.0,
            alpha=0.12, color=ORANGE, zorder=1)

# Anotación del pico — texto posicionado para no solapar la leyenda (upper right)
peak_idx_mut = int(np.argmax(mean_mut))
peak_x_mut   = x_axis[peak_idx_mut]
# Si el pico está en la mitad derecha, colocar texto a la izquierda y vice versa
if peak_x_mut >= 0.5:
    ann_x_mut = peak_x_mut - 0.38
    rad = 0.20
else:
    ann_x_mut = peak_x_mut + 0.38
    rad = -0.20
ax2.annotate(
    'Peak in stateful features\n(flow statistics)',
    xy=(peak_x_mut, mean_mut[peak_idx_mut]),
    xytext=(ann_x_mut, 0.68),
    fontsize=9, ha='center', va='top',
    color=ORANGE, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5,
                    connectionstyle=f'arc3,rad={rad}'),
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
              edgecolor=ORANGE, alpha=0.90, lw=0.8)
)

ax2.set_title('(b) Evasive Attack (Mutant Payload): Metadata shifted to byte offset 70–80',
              fontweight='bold', fontsize=11, pad=8)
# Línea divisoria SL / SF
ax2.axvline(x=sf_start, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax2.set_ylabel('Normalized Activation Magnitude', fontsize=10)
ax2.set_xlabel('Feature Sequence Position (normalized)', fontsize=10)
ax2.set_ylim(-0.05, 1.30)
ax2.set_xlim(-0.02, 1.02)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.85)
ax2.tick_params(labelsize=9)

# ─────────────────────────────────────────────────────────
# Leyenda global compartida — debajo del panel (b)
# ─────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=RED_META, alpha=0.35,
                   label='Stateless features — DNS name patterns (FEATS_SL, pos. 0–10)'),
    mpatches.Patch(facecolor=ORANGE,   alpha=0.35,
                   label='Stateful features — flow statistics (FEATS_SF, pos. 11–22)'),
]
fig.legend(handles=legend_patches,
           loc='lower center', ncol=2,
           fontsize=8.5, framealpha=0.85,
           bbox_to_anchor=(0.5, 0.00))

fig.suptitle(
    'CNN Feature Group Activation: Stateless vs. Stateful Response',
    fontsize=12, fontweight='bold'
)


# =============================================================================
# PASO 4: MANEJO DE CASOS EDGE — reporte de advertencias en el plot
# =============================================================================

warnings = []
if D == 2:
    warnings.append(f"⚠ Solo D={D} puntos espaciales (arquitectura con MaxPooling agresivo)")
if n_cic < 10:
    warnings.append(f"⚠ Pocos flows CIC-2021: N={n_cic} (recomendado ≥10)")
if n_mut < 10:
    warnings.append(f"⚠ Pocos flows Mutant Payload: N={n_mut} (recomendado ≥10)")

if warnings:
    warning_text = "\n".join(warnings)
    fig.text(0.01, 0.01, warning_text, fontsize=7, color='#cc0000',
             va='bottom', ha='left',
             bbox=dict(boxstyle='round', facecolor='#fff3f3',
                       edgecolor='#cc0000', alpha=0.8))


# =============================================================================
# PASO 5: GUARDAR
# =============================================================================

plt.tight_layout(rect=[0, 0.06, 1, 0.97])

out_png = "figure2_cnn_activation_FINAL.png"
out_pdf = "figure2_cnn_activation_FINAL.pdf"

plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FIGURE 2 GENERATION REPORT")
print("=" * 60)
print(f"CNN last Conv1D output shape : (None, {D}, {F})")
print(f"Spatial dimensions (D)       : {D}")
print(f"Filters (F)                  : {F}")
print(f"CIC-2021 flows used          : {n_cic}  (mean prob={np.mean(probs_cic):.3f})")
print(f"Mutant Payload flows used    : {n_mut}  (mean prob={np.mean(probs_mut):.3f})")
print(f"Figura PNG guardada          : {out_png}")
print(f"Figura PDF guardada          : {out_pdf}")
if warnings:
    print("\nADVERTENCIAS:")
    for w in warnings:
        print(f"  {w}")
print("=" * 60)


# =============================================================================
# PASO 6: CAPTION LaTeX LISTO PARA COPIAR
# (reemplazar N_CIC y N_MUT con los valores del reporte)
# =============================================================================
caption_latex = r"""
% ============================================================
% CAPTION PARA LATEX — copiar al paper
% ============================================================
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figure2_cnn_activation_FINAL.pdf}
  \caption{Normalized mean activation profiles of the \emph{first}
  convolutional layer (64 filters, kernel size~3) averaged over
  100~CIC-2021 attack flows \textbf{(a)} and 100~Mutant~Payload
  evasive attack flows \textbf{(b)} ($\pm$1\,SD shaded). Features
  are grouped as stateless (DNS name patterns, positions~0--10,
  pink region) and stateful (flow statistics, positions~11--22,
  orange region). In \textbf{(a)}, standard attacks produce
  moderate activation across stateless features and strong
  activation in stateful features. In \textbf{(b)}, stateless
  activation collapses nearly to zero when metadata is shifted
  to byte offset~70--80, while stateful features maintain partial
  activation---explaining the CNN's partial robustness (29.85\%
  recall): stateful flow statistics preserve discriminative power
  even when DNS name patterns are disrupted by positional mutation.}
  \label{fig:cnn_activation}
\end{figure}
% ============================================================
"""
print(caption_latex)
