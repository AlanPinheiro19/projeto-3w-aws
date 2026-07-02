"""
shap_analysis.py
================
Análise de interpretabilidade SHAP para o modelo LightGBM baseline (3W Petrobras).

O que gera:
  shap_global_summary.png        — beeswarm global (top 20 features, todas as classes)
  shap_global_bar.png            — bar chart de importância média |SHAP| por feature
  shap_class_<N>_<nome>.png      — beeswarm por classe de evento (top 15 features)
  shap_sensor_importance.png     — importância agregada por sensor original (8 sensores)
  shap_sensor_per_class.png      — heatmap: sensor × classe (importância relativa)
  shap_results.json              — tabela de top features por classe (para monografia)

Estratégia de amostragem:
  SHAP com TreeExplainer é O(n_samples × n_trees). Para 5M amostras,
  usamos amostra estratificada de background_per_class × n_classes amostras,
  garantindo representação de todas as classes, inclusive DHSV Closure (rara).

Uso:
    python scripts/shap_analysis.py
    python scripts/shap_analysis.py --n-background 500 --n-explain 2000 --verbose
    python scripts/shap_analysis.py --model lgb    # padrão
    python scripts/shap_analysis.py --model xgb    # também suporta XGBoost

Dependências:
    pip install shap --break-system-packages
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

# ── Resolve PROJECT_DIR ────────────────────────────────────────────────────────
def _find_project_dir() -> Path:
    if 'PROJECT_DIR' in os.environ:
        return Path(os.environ['PROJECT_DIR'])
    candidates = [
        r'D:\Alan\Alan\Pos GRADUAÇÂO USP\eEDB-007 - Trabalho de Conclusão\projeto-3w-aws',
        r'C:\projeto-3w-aws',
        Path(__file__).parent.parent,
        Path.cwd(),
    ]
    for c in candidates:
        p = Path(c)
        if (p / 'data' / 'gold').exists():
            return p
    raise FileNotFoundError('Não foi possível localizar PROJECT_DIR com data/gold/')

PROJECT_DIR = _find_project_dir()
GOLD_DIR    = PROJECT_DIR / 'data' / 'gold'
MODEL_DIR   = PROJECT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LABELS = {
    0: 'Normal',           1: 'BSW Abrupt',       2: 'DHSV Closure',
    3: 'Severe Slugging',  4: 'Flow Instability',  5: 'Prod. Loss',
    6: 'PCK Restriction',  7: 'Scaling',           8: 'Hydrate',
}

SENSOR_COLUMNS = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP',
                  'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL', 'QGL']

# Paleta por classe
CLASS_COLORS = {
    0: '#4C9BE8', 1: '#F4A261', 2: '#E63946', 3: '#457B9D',
    4: '#2A9D8F', 5: '#E9C46A', 6: '#8338EC', 7: '#06D6A0', 8: '#FB8500',
}

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='SHAP analysis — 3W LightGBM baseline')
    p.add_argument('--model',         type=str, default='lgb',
                   choices=['lgb', 'xgb', 'rf'],
                   help='Modelo a analisar (default: lgb)')
    p.add_argument('--n-background',  type=int, default=100,
                   help='Amostras de background por classe (default: 100)')
    p.add_argument('--n-explain',     type=int, default=200,
                   help='Amostras a explicar por classe (default: 200)')
    p.add_argument('--top-features',  type=int, default=15,
                   help='Top N features nos gráficos por classe (default: 15)')
    p.add_argument('--verbose',       action='store_true')
    return p.parse_args()

# ── Carregamento de dados ──────────────────────────────────────────────────────
def load_data():
    """Carrega apenas test.parquet (amostra estratificada) para analise SHAP.
    Train e val NAO sao necessarios (apenas X_test e retornado).
    Leitura em chunks evita OOM — SHAP precisa de no maximo ~2000 amostras.
    """
    import pyarrow.parquet as pq

    MAX_TEST   = 20_000   # muito acima dos n_per_class=200 usados no SHAP
    CHUNK_SIZE = 30_000

    print('Carregando dados Gold (apenas test.parquet, amostra para SHAP)...')
    pf    = pq.ParquetFile(str(GOLD_DIR / 'test.parquet'))
    total = pf.metadata.num_rows
    rate  = (MAX_TEST / total) if total > MAX_TEST else 1.0
    chunks, rows = [], 0
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk = batch.to_pandas()
        if rate < 1.0 and 'class' in chunk.columns:
            chunk = (chunk.groupby('class', group_keys=False)
                          .apply(lambda g: g.sample(
                              n=max(1, round(len(g) * rate)), random_state=42))
                          .reset_index(drop=True))
        chunks.append(chunk)
        rows += len(chunk)
        if rows >= MAX_TEST:
            break
    test = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f'  test={len(test):,} linhas carregadas (de {total:,} totais)')

    feat_cols = [c for c in test.columns if c.endswith('_norm')]
    if not feat_cols:
        feat_cols = [c for c in SENSOR_COLUMNS if c in test.columns]

    if 'class' not in test.columns:
        raise KeyError("Coluna 'class' nao encontrada em test.parquet")

    X_test = test[feat_cols].values.astype('float32')
    y_test = test['class'].values.astype('int32')

    classes = sorted(np.unique(y_test))
    print(f'  Classes no teste: {classes}')

    def _clean(X):
        bad = int((~np.isfinite(X)).sum())
        if bad:
            print(f'  [AVISO] {bad} NaN/Inf → substituidos por 0')
        return np.where(np.isfinite(X), X, np.float32(0.0)).astype('float32')

    return (_clean(X_test), y_test, feat_cols)   # SHAP no conjunto de teste

# ── Amostragem estratificada ───────────────────────────────────────────────────
def stratified_sample(X, y, n_per_class, seed=42):
    """Seleciona até n_per_class amostras por classe, garantindo representação igual."""
    rng    = np.random.default_rng(seed)
    idx    = []
    classes = sorted(np.unique(y))
    for cls in classes:
        mask    = np.where(y == cls)[0]
        n       = min(n_per_class, len(mask))
        chosen  = rng.choice(mask, size=n, replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    return X[idx], y[idx], idx

# ── Nomes limpos para os gráficos ──────────────────────────────────────────────
def clean_feature_name(fname):
    """Remove sufixo _norm e abrevia para legibilidade."""
    name = fname.replace('_norm', '')
    # Substitui nomes longos de sensores
    abbrev = {
        'P-PDG':      'P-PDG',
        'P-TPT':      'P-TPT',
        'T-TPT':      'T-TPT',
        'P-MON-CKP':  'P-MON-CKP',
        'T-JUS-CKP':  'T-JUS-CKP',
        'P-JUS-CKGL': 'P-JUS-CKGL',
        'T-JUS-CKGL': 'T-JUS-CKGL',
        'QGL':        'QGL',
    }
    for sensor, short in abbrev.items():
        if name.startswith(sensor):
            suffix = name[len(sensor):]
            return f'{short}{suffix}'
    return name

# ── Extração do sensor base de uma feature ────────────────────────────────────
def feature_to_sensor(fname):
    """Mapeia feature → sensor base (ex: P_PDG_roll_mean_10_norm → P-PDG).
    Features rolling usam underscore (P_PDG_...) enquanto os nomes dos sensores
    usam hífen (P-PDG). Normaliza ambos para underscore na comparação.
    """
    fname_norm = fname.replace('-', '_')
    for sensor in SENSOR_COLUMNS:
        sensor_norm = sensor.replace('-', '_')
        if fname_norm.startswith(sensor_norm):
            return sensor
    return 'outro'

# ── Plot: summary global (beeswarm) ───────────────────────────────────────────
def plot_global_summary(shap_values, X_sample, feat_names, top_n=20):
    """shap_values: (n_samples, n_features, n_classes) para multiclass."""
    try:
        import shap
    except ImportError:
        return

    # Importância média |SHAP| por feature (média sobre classes e amostras)
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))   # (n_features,)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    X_disp    = X_sample[:, top_idx]
    sv_disp   = shap_values[:, top_idx, :].mean(axis=2)   # média das classes
    fnames    = [clean_feature_name(feat_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv_disp, X_disp, feature_names=fnames,
                      show=False, plot_size=None, color_bar=True)
    ax = plt.gca()
    ax.set_title('SHAP — Importância Global (média sobre todas as classes)', fontsize=13, pad=12)
    plt.tight_layout()
    out = MODEL_DIR / 'shap_global_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Salvo: {out.name}')

# ── Plot: bar chart global ─────────────────────────────────────────────────────
def plot_global_bar(shap_values, feat_names, top_n=20):
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    vals     = mean_abs[top_idx]
    fnames   = [clean_feature_name(feat_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
    bars = ax.barh(range(top_n)[::-1], vals, color=colors)
    ax.set_yticks(range(top_n)[::-1])
    ax.set_yticklabels(fnames, fontsize=10)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=11)
    ax.set_title(f'Top {top_n} Features — Importância SHAP Global', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = MODEL_DIR / 'shap_global_bar.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Salvo: {out.name}')

# ── Plot: beeswarm por classe ──────────────────────────────────────────────────
def plot_per_class(shap_values, X_sample, y_sample, feat_names, classes, top_n=15):
    try:
        import shap
    except ImportError:
        return

    for cls_idx, cls in enumerate(classes):
        cls = int(cls)
        label = EVENT_LABELS.get(cls, str(cls))

        sv_cls = shap_values[:, :, cls_idx]           # (n_samples, n_features)
        mean_abs = np.abs(sv_cls).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]

        X_disp  = X_sample[:, top_idx]
        sv_disp = sv_cls[:, top_idx]
        fnames  = [clean_feature_name(feat_names[i]) for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv_disp, X_disp, feature_names=fnames,
                          show=False, plot_size=None,
                          plot_type='dot', color_bar=True)
        plt.title(f'SHAP — Classe {cls}: {label} (top {top_n} features)', fontsize=12, pad=10)
        plt.tight_layout()
        safe_label = label.replace(' ', '_').replace('/', '-')
        out = MODEL_DIR / f'shap_class_{cls:02d}_{safe_label}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Salvo: {out.name}')

# ── Plot: importância por sensor (agregado) ────────────────────────────────────
def plot_sensor_importance(shap_values, feat_names, classes):
    """Agrega importância SHAP por sensor original (soma |SHAP| das suas features)."""
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))   # (n_features,)

    sensor_imp = {s: 0.0 for s in SENSOR_COLUMNS}
    for fi, fname in enumerate(feat_names):
        sensor = feature_to_sensor(fname)
        if sensor in sensor_imp:
            sensor_imp[sensor] += float(mean_abs[fi])

    sensors = list(sensor_imp.keys())
    vals    = [sensor_imp[s] for s in sensors]
    total   = sum(vals)
    pct     = [v / total * 100 for v in vals]

    order   = np.argsort(pct)[::-1]
    sensors_o = [sensors[i] for i in order]
    pct_o     = [pct[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ['#2A9D8F' if p >= 15 else '#4C9BE8' if p >= 8 else '#CBD5E1' for p in pct_o]
    bars = ax.bar(range(len(sensors_o)), pct_o, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(sensors_o)))
    ax.set_xticklabels(sensors_o, fontsize=11)
    ax.set_ylabel('Importância SHAP relativa (%)', fontsize=11)
    ax.set_title('Importância SHAP por Sensor — Todas as Classes', fontsize=13)
    for bar, p in zip(bars, pct_o):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{p:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='500')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(pct_o) * 1.15)
    plt.tight_layout()
    out = MODEL_DIR / 'shap_sensor_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Salvo: {out.name}')
    return sensor_imp

# ── Plot: heatmap sensor × classe ─────────────────────────────────────────────
def plot_sensor_heatmap(shap_values, feat_names, classes):
    """Heatmap mostrando quais sensores são mais importantes para cada classe."""
    n_classes = shap_values.shape[2]
    sensor_class_imp = np.zeros((len(SENSOR_COLUMNS), n_classes))

    for fi, fname in enumerate(feat_names):
        sensor = feature_to_sensor(fname)
        if sensor in SENSOR_COLUMNS:
            si = SENSOR_COLUMNS.index(sensor)
            for ci in range(n_classes):
                sensor_class_imp[si, ci] += float(np.abs(shap_values[:, fi, ci]).mean())

    # Normalizar por coluna (classe) para mostrar importância relativa
    col_sums = sensor_class_imp.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    sensor_class_pct = sensor_class_imp / col_sums * 100

    class_labels = [f'Cls {int(c)}\n{EVENT_LABELS.get(int(c), "?")[:8]}' for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.4), 5))
    im = ax.imshow(sensor_class_pct, cmap='YlOrRd', aspect='auto', vmin=0)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_yticks(range(len(SENSOR_COLUMNS)))
    ax.set_yticklabels(SENSOR_COLUMNS, fontsize=10)
    ax.set_title('Importância SHAP por Sensor × Classe (%)', fontsize=12, pad=10)

    plt.colorbar(im, ax=ax, label='% importância SHAP da classe')

    # Anotações nos valores
    for si in range(len(SENSOR_COLUMNS)):
        for ci in range(len(classes)):
            v = sensor_class_pct[si, ci]
            color = 'white' if v > 50 else 'black'
            ax.text(ci, si, f'{v:.0f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='500')

    plt.tight_layout()
    out = MODEL_DIR / 'shap_sensor_per_class.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Salvo: {out.name}')
    return sensor_class_pct

# ── Export JSON para monografia ────────────────────────────────────────────────
def export_json(shap_values, feat_names, classes, sensor_class_pct):
    """Exporta top features por classe e importância por sensor → JSON para monografia."""
    result = {
        'modelo': 'LightGBM baseline',
        'n_amostras_shap': int(shap_values.shape[0]),
        'n_features': int(shap_values.shape[1]),
        'classes': {},
        'sensor_importance_global': {},
    }

    mean_global = np.abs(shap_values).mean(axis=(0, 2))
    sensor_global = {s: 0.0 for s in SENSOR_COLUMNS}
    for fi, fname in enumerate(feat_names):
        sensor = feature_to_sensor(fname)
        if sensor in sensor_global:
            sensor_global[sensor] += float(mean_global[fi])
    total = sum(sensor_global.values()) or 1.0
    result['sensor_importance_global'] = {
        s: round(v / total * 100, 2) for s, v in
        sorted(sensor_global.items(), key=lambda x: -x[1])
    }

    for ci, cls in enumerate(classes):
        cls_int = int(cls)
        label   = EVENT_LABELS.get(cls_int, str(cls_int))
        sv_cls  = shap_values[:, :, ci]
        mean_abs = np.abs(sv_cls).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:10]

        top_features = []
        for fi in top_idx:
            top_features.append({
                'feature':  feat_names[fi],
                'sensor':   feature_to_sensor(feat_names[fi]),
                'shap_mean_abs': round(float(mean_abs[fi]), 6),
            })

        sensor_pct = {SENSOR_COLUMNS[si]: round(float(sensor_class_pct[si, ci]), 1)
                      for si in range(len(SENSOR_COLUMNS))}
        sensor_pct = dict(sorted(sensor_pct.items(), key=lambda x: -x[1]))

        result['classes'][str(cls_int)] = {
            'label':            label,
            'top_10_features':  top_features,
            'sensor_importance_pct': sensor_pct,
        }

    json_path = MODEL_DIR / 'shap_results.json'
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  JSON: {json_path}')
    return result

# ── Impressão do resumo para monografia ───────────────────────────────────────
def print_summary(result):
    print('\n' + '=' * 60)
    print('  RESUMO SHAP — TOP SENSORES POR CLASSE')
    print('=' * 60)
    for cls_str, data in result['classes'].items():
        label = data['label']
        top3  = list(data['sensor_importance_pct'].items())[:3]
        top3_str = ' | '.join(f'{s}: {v:.0f}%' for s, v in top3)
        print(f'  Classe {cls_str:2s} ({label:<20}): {top3_str}')
    print('=' * 60)
    print('\n  Importância global por sensor:')
    for sensor, pct in result['sensor_importance_global'].items():
        bar = '█' * int(pct / 2)
        print(f'  {sensor:<14}: {pct:5.1f}%  {bar}')

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    try:
        import shap
        print(f'shap version: {shap.__version__}')
    except ImportError:
        print('\nERRO: shap não instalado.')
        print('  pip install shap --break-system-packages')
        sys.exit(1)

    import joblib

    print(f'scripts/shap_analysis.py')
    print(f'PROJECT_DIR : {PROJECT_DIR}')

    args = parse_args()

    # 1. Dados
    X_test, y_test, feat_names = load_data()
    classes = np.array(sorted(int(c) for c in np.unique(y_test)))
    print(f'\n  Classes: {list(classes)}')
    print(f'  Features: {len(feat_names)}')

    # 2. Modelo
    model_file = {'lgb': 'lgb_baseline.joblib',
                  'xgb': 'xgb_baseline.joblib',
                  'rf':  'rf_baseline.joblib'}[args.model]
    model_path = MODEL_DIR / model_file
    if not model_path.exists():
        print(f'\nERRO: modelo não encontrado: {model_path}')
        sys.exit(1)
    print(f'\nCarregando {model_file}...')
    clf = joblib.load(model_path)

    # 3. Amostragem estratificada
    n_bg  = args.n_background
    n_exp = args.n_explain
    print(f'\nAmostragem estratificada:')
    print(f'  background = {n_bg}/classe × {len(classes)} classes = ~{n_bg*len(classes):,} amostras')
    print(f'  explicação = {n_exp}/classe × {len(classes)} classes = ~{n_exp*len(classes):,} amostras')

    X_bg,  y_bg,  _ = stratified_sample(X_test, y_test, n_bg,  seed=42)
    X_exp, y_exp, _ = stratified_sample(X_test, y_test, n_exp, seed=99)
    print(f'  X_background shape: {X_bg.shape}')
    print(f'  X_explain shape:    {X_exp.shape}')

    # 4. SHAP TreeExplainer
    print(f'\nCriando TreeExplainer...')
    # Não usar model_output='raw': causa incompatibilidade de escala em LGB multiclass.
    # TreeExplainer sem data é mais rápido e usa o valor esperado interno do modelo.
    explainer = shap.TreeExplainer(clf)

    print(f'Calculando SHAP values para {len(X_exp):,} amostras...')
    print(f'  (pode levar alguns minutos)')
    t0 = __import__('time').time()

    # check_additivity=False: necessário para LGB multiclass com alguns backends
    shap_vals = explainer.shap_values(X_exp, check_additivity=False)
    elapsed = __import__('time').time() - t0
    print(f'  Concluído em {elapsed:.1f}s')

    # shap_vals pode ser lista [array_cls0, ...] ou array 3D
    if isinstance(shap_vals, list):
        # lista de (n_samples, n_features) → stack para (n_samples, n_features, n_classes)
        shap_3d = np.stack(shap_vals, axis=2)
    elif shap_vals.ndim == 3:
        shap_3d = shap_vals                  # já (n_samples, n_features, n_classes)
    else:
        # 2D: modelo binário ou single-output — expande
        shap_3d = shap_vals[:, :, np.newaxis]

    # Para XGBoost com remapeamento: alinha n_classes do modelo com classes dos dados
    if args.model == 'xgb' and shap_3d.shape[2] != len(classes):
        n_model_cls = shap_3d.shape[2]
        if n_model_cls < len(classes):
            # Remapeamento: os índices do modelo correspondem à ordem crescente das classes
            print(f'  [XGB] {n_model_cls} classes no modelo → {len(classes)} originais (remapeamento)')
            # shap_3d já está correto; apenas usamos classes como labels
            classes = classes[:n_model_cls]   # ajusta para não indexar além

    print(f'  SHAP values shape: {shap_3d.shape}  '
          f'(amostras × features × {shap_3d.shape[2]} classes)')

    # 5. Gráficos
    print('\nGerando gráficos...')
    plot_global_summary(shap_3d, X_exp, feat_names, top_n=args.top_features)
    plot_global_bar(shap_3d, feat_names, top_n=args.top_features)
    plot_per_class(shap_3d, X_exp, y_exp, feat_names, classes, top_n=args.top_features)
    sensor_imp  = plot_sensor_importance(shap_3d, feat_names, classes)
    sensor_pct  = plot_sensor_heatmap(shap_3d, feat_names, classes)

    # 6. JSON + resumo
    print('\nExportando resultados...')
    result = export_json(shap_3d, feat_names, classes, sensor_pct)
    print_summary(result)

    print('\nSHAP analysis concluída!')
    print(f'Arquivos salvos em: {MODEL_DIR}')

if __name__ == '__main__':
    main()
