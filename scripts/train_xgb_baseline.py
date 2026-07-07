"""
train_xgb_baseline.py
=====================
Treina XGBoost baseline para classificação de eventos em poços 3W Petrobras.

Diferenciais vs Random Forest:
  - Gradient boosting: menos overfitting em dados desbalanceados
  - sample_weight proporcional ao inverso da frequência de cada classe
  - Early stopping via eval set (para quando val loss não melhora)
  - Feature importance por 'gain' (mais interpretável que 'weight')

Uso:
    python scripts/train_xgb_baseline.py
    python scripts/train_xgb_baseline.py --n-estimators 500 --max-depth 6 --verbose

Saídas (em models/):
    xgb_baseline.joblib              modelo treinado
    xgb_baseline_report.txt          classification report (val + test)
    xgb_baseline_confusion.png       matriz de confusão (test)
    xgb_baseline_importance.png      top-30 features por ganho
    xgb_baseline_metrics.json        métricas resumidas
"""

import os
import sys
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Resolve PROJECT_DIR ───────────────────────────────────────────────────────
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
    0: 'Normal',        1: 'BSW Abrupt',      2: 'DHSV Closure',
    3: 'Severe Slugging', 4: 'Flow Instability', 5: 'Prod. Loss',
    6: 'PCK Restriction', 7: 'Scaling',         8: 'Hydrate',
}

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='XGBoost baseline — 3W dataset')
    p.add_argument('--n-estimators',   type=int,   default=500,
                   help='Número máximo de rounds (default: 500; early stopping pode reduzir)')
    p.add_argument('--max-depth',      type=int,   default=6,
                   help='Profundidade máxima da árvore (default: 6)')
    p.add_argument('--learning-rate',  type=float, default=0.05,
                   help='Taxa de aprendizado eta (default: 0.05)')
    p.add_argument('--subsample',      type=float, default=0.8,
                   help='Fração de amostras por árvore (default: 0.8)')
    p.add_argument('--colsample',      type=float, default=0.8,
                   help='Fração de features por árvore (default: 0.8)')
    p.add_argument('--early-stopping', type=int,   default=30,
                   help='Rounds sem melhora para parar (default: 30)')
    p.add_argument('--n-jobs',         type=int,   default=-1)
    p.add_argument('--max-samples',    type=int,   default=500_000,
                   help='Máx amostras de treino via chunks estratificados (default: 500000). '
                        '0 = carrega tudo. Igualado ao RF para comparação justa.')
    p.add_argument('--verbose',        action='store_true')
    return p.parse_args()

# ── Carregamento com chunked sampling (igual ao RF) ───────────────────────────
def load_data(max_samples: int = 500_000):
    """Carrega splits Gold em chunks com amostragem estratificada por classe.

    Idêntico ao train_rf_baseline.py para garantir comparação justa entre modelos:
    todos treinam no mesmo volume de dados e mesma distribuição de classes.

    max_samples: limite de linhas para train (val e test usam max_samples//3).
    0 = carrega o dataset completo sem amostragem.
    """
    import pyarrow.parquet as pq

    print('Carregando dados Gold (chunked sampling estratificado)...')
    print(f'  Diretório Gold: {GOLD_DIR}')

    CHUNK_SIZE = 30_000   # linhas por batch de leitura

    def _read_chunked(path, max_rows, class_col='class'):
        """Lê parquet em chunks e amostra estratificadamente para caber em max_rows."""
        pf    = pq.ParquetFile(str(path))
        total = pf.metadata.num_rows
        rate  = (max_rows / total) if (max_rows and total > max_rows) else 1.0
        chunks, rows = [], 0
        for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
            chunk = batch.to_pandas()
            if rate < 1.0 and class_col in chunk.columns:
                chunk = (chunk.groupby(class_col, group_keys=False)
                              .apply(lambda g: g.sample(
                                  n=max(1, round(len(g) * rate)),
                                  random_state=42))
                              .reset_index(drop=True))
            elif rate < 1.0:
                chunk = chunk.sample(n=max(1, round(len(chunk) * rate)),
                                     random_state=42)
            chunks.append(chunk)
            rows += len(chunk)
            if max_rows and rows >= max_rows:
                break
        if not chunks:
            return pd.DataFrame()
        result = pd.concat(chunks, ignore_index=True)
        tag = f'{total:,} → {len(result):,} (estratificado)' if total > max_rows and max_rows else f'{len(result):,}'
        print(f'  {path.name}: {tag}')
        return result

    val_max  = max_samples // 3 if max_samples else 0
    test_max = max_samples // 3 if max_samples else 0

    val   = _read_chunked(GOLD_DIR / 'val.parquet',   val_max)
    test  = _read_chunked(GOLD_DIR / 'test.parquet',  test_max)
    train = _read_chunked(GOLD_DIR / 'train.parquet', max_samples)
    print(f'  train={len(train):,}  val={len(val):,}  test={len(test):,}')

    total = len(train) + len(val) + len(test)
    if total == 0:
        raise FileNotFoundError(
            f'Todos os arquivos Gold estão vazios em {GOLD_DIR}.\n'
            'Re-execute o pipeline Gold: dag_gold_rebuild no Airflow.'
        )

    feat_cols = [c for c in train.columns if c.endswith('_norm')]
    if not feat_cols:
        sensor_cols = ['P-PDG','P-TPT','T-TPT','P-MON-CKP','T-JUS-CKP','P-JUS-CKGL','T-JUS-CKGL','QGL']
        feat_cols = [c for c in sensor_cols if c in train.columns]
        print(f'  [AVISO] Sem features _norm — usando {len(feat_cols)} sensores brutos')
    print(f'  Features: {len(feat_cols)}')

    # ── Split estratificado se classes ausentes em algum split ────────────────
    from sklearn.model_selection import train_test_split
    all_data    = pd.concat([train, val, test], ignore_index=True)
    classes_all = set(all_data['class'].unique())
    missing     = (classes_all - set(train['class'].unique())
                   | classes_all - set(val['class'].unique())
                   | classes_all - set(test['class'].unique()))

    if missing:
        print(f'\n  [AVISO] Classes ausentes em algum split: '
              f'{sorted(int(c) for c in missing)}')
        print('  → Aplicando split ESTRATIFICADO global (15% val, 15% test)...')
        y_all = all_data['class'].values.astype('int32')
        X_all = all_data[feat_cols].values.astype('float32')
        X_tmp, X_test_s, y_tmp, y_test_s = train_test_split(
            X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
        X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
            X_tmp, y_tmp, test_size=round(0.15/0.85, 6), random_state=42, stratify=y_tmp)
        print(f'  Split estratificado: train={len(y_train_s):,}  '
              f'val={len(y_val_s):,}  test={len(y_test_s):,}')
    else:
        X_train_s = train[feat_cols].values.astype('float32')
        y_train_s = train['class'].values.astype('int32')
        X_val_s   = val[feat_cols].values.astype('float32')
        y_val_s   = val['class'].values.astype('int32')
        X_test_s  = test[feat_cols].values.astype('float32')
        y_test_s  = test['class'].values.astype('int32')

    # ── Sanitiza NaN / Inf ────────────────────────────────────────────────────
    def _clean(X, name):
        bad = int((~np.isfinite(X)).sum())
        if bad:
            print(f'  [AVISO] {name}: {bad} NaN/Inf → substituídos por 0')
        return np.where(np.isfinite(X), X, np.float32(0.0)).astype('float32')

    X_train_s = _clean(X_train_s, 'X_train')
    X_val_s   = _clean(X_val_s,   'X_val')
    X_test_s  = _clean(X_test_s,  'X_test')

    print(f'\n  Classes no treino: {sorted(int(c) for c in np.unique(y_train_s))}')
    dist = pd.Series(y_train_s).value_counts().sort_index()
    for cls, cnt in dist.items():
        print(f'    Classe {int(cls)} ({EVENT_LABELS.get(int(cls),"?")}): '
              f'{cnt:,} ({cnt/len(y_train_s)*100:.2f}%)')

    return X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, feat_cols

# ── Pesos por amostra (inverso da frequência) ─────────────────────────────────
def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Calcula peso por amostra proporcional ao inverso da frequência da classe.
    Equivale a class_weight='balanced' do sklearn, mas compatível com XGBoost nativo.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n_total = len(y)
    n_classes = len(classes)
    weights = np.array([
        n_total / (n_classes * freq[c]) for c in y
    ], dtype='float32')
    return weights

# ── Treinamento ───────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, args):
    try:
        import xgboost as xgb
    except ImportError:
        print('\nERRO: xgboost não instalado.')
        print('  pip install xgboost --break-system-packages')
        print('  ou dentro do container: pip install xgboost')
        sys.exit(1)

    # Remapeia labels para [0, N-1] — usa TODAS as classes (train+val+test)
    # para evitar KeyError quando alguma classe aparece só no val/test
    classes_all = sorted(set(np.unique(y_train)) | set(np.unique(y_val)))
    label_map  = {int(c): i for i, c in enumerate(classes_all)}
    label_rmap = {i: c for c, i in label_map.items()}
    y_train_xgb = np.array([label_map[int(c)] for c in y_train], dtype='int32')
    y_val_xgb   = np.array([label_map[int(c)] for c in y_val],   dtype='int32')

    sample_weights = compute_sample_weights(y_train_xgb)
    n_classes      = len(classes_all)

    print(f'\nTreinando XGBoost ({args.n_estimators} rounds max, '
          f'max_depth={args.max_depth}, lr={args.learning_rate})...')
    print(f'  {n_classes} classes | early_stopping={args.early_stopping} rounds')

    params = {
        'objective':        'multi:softprob',
        'num_class':        n_classes,
        'eval_metric':      'mlogloss',
        'max_depth':        args.max_depth,
        'learning_rate':    args.learning_rate,
        'subsample':        args.subsample,
        'colsample_bytree': args.colsample,
        'min_child_weight': 5,
        'gamma':            0.1,
        'reg_alpha':        0.1,      # L1 — reduz overfitting
        'reg_lambda':       1.0,      # L2
        'n_jobs':           args.n_jobs,
        'random_state':     42,
        'tree_method':      'hist',   # mais rápido que 'exact'
        'verbosity':        1 if args.verbose else 0,
    }

    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping,
        **params,
    )

    t0 = time.time()
    try:
        clf.fit(
            X_train, y_train_xgb,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val_xgb)],
            verbose=args.verbose,
        )
    except Exception as e:
        import traceback
        print(f'\nERRO NO FIT: {type(e).__name__}: {e}')
        traceback.print_exc()
        raise

    elapsed = time.time() - t0
    best_round = clf.best_iteration + 1 if hasattr(clf, 'best_iteration') else args.n_estimators
    print(f'  Treinamento concluído em {elapsed:.1f}s | melhor round: {best_round}')

    return clf, label_map, label_rmap

# ── Avaliação ─────────────────────────────────────────────────────────────────
def evaluate(clf, X, y_true_orig, label_map, split_name, classes_orig):
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    y_true_xgb = np.array([label_map[int(c)] for c in y_true_orig], dtype='int32')
    y_pred_xgb = clf.predict(X)

    acc    = accuracy_score(y_true_xgb, y_pred_xgb)
    f1_w   = f1_score(y_true_xgb, y_pred_xgb, average='weighted', zero_division=0)
    f1_mac = f1_score(y_true_xgb, y_pred_xgb, average='macro',    zero_division=0)

    labels_idx = list(range(len(classes_orig)))
    report = classification_report(
        y_true_xgb, y_pred_xgb,
        labels=labels_idx,
        target_names=[EVENT_LABELS.get(c, f'Cls{c}') for c in classes_orig],
        zero_division=0,
    )
    print(f'\n── {split_name} ──')
    print(f'  Acurácia:    {acc:.4f}')
    print(f'  F1 weighted: {f1_w:.4f}')
    print(f'  F1 macro:    {f1_mac:.4f}')
    print(report)
    return y_pred_xgb, acc, f1_w, f1_mac, report

# ── Matriz de Confusão ────────────────────────────────────────────────────────
def plot_confusion(y_true_xgb, y_pred_xgb, classes_orig, out_path):
    from sklearn.metrics import confusion_matrix

    labels_idx = list(range(len(classes_orig)))
    cm     = confusion_matrix(y_true_xgb, y_pred_xgb, labels=labels_idx)
    cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    labels = [EVENT_LABELS.get(c, f'Cls{c}') for c in classes_orig]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Matriz de Confusão — XGBoost Baseline (Teste)', fontsize=13, fontweight='bold')

    for ax, data, fmt, title in zip(
        axes, [cm, cm_pct], ['d', '.1f'],
        ['Contagem Absoluta', 'Percentual por Classe Real (%)']
    ):
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Predito'); ax.set_ylabel('Real')
        thresh = data.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{data[i,j]:{fmt}}', ha='center', va='center',
                        color='white' if data[i,j] > thresh else 'black', fontsize=7)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Salvo: {out_path.name}')

# ── Feature Importance (gain) ─────────────────────────────────────────────────
def plot_importance(clf, feat_cols, out_path, top_n=30):
    importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1], color='#E67E22', edgecolor='white')
    ax.set_title(f'Top {top_n} Features — XGBoost Baseline (gain)', fontweight='bold')
    ax.set_xlabel('Importância (Gain)')
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Salvo: {out_path.name}')
    return importances

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    import xgboost as xgb
    import joblib
    from sklearn.metrics import f1_score, accuracy_score

    print(f'PROJECT_DIR : {PROJECT_DIR}')
    print(f'xgboost     : {xgb.__version__}')

    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = load_data(
        max_samples=args.max_samples
    )
    clf, label_map, label_rmap = train_model(X_train, y_train, X_val, y_val, args)

    classes_orig = sorted(np.unique(np.concatenate([y_train, y_val, y_test])))

    y_val_pred,  acc_v, f1w_v, f1m_v, rep_v = evaluate(clf, X_val,  y_val,  label_map, 'VALIDAÇÃO', classes_orig)
    y_test_pred, acc_t, f1w_t, f1m_t, rep_t = evaluate(clf, X_test, y_test, label_map, 'TESTE',     classes_orig)

    # Report
    report_path = MODEL_DIR / 'xgb_baseline_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== XGBOOST BASELINE — 3W DATASET ===\n\n')
        f.write(f'n_estimators (max)   : {args.n_estimators}\n')
        f.write(f'best_iteration       : {clf.best_iteration + 1 if hasattr(clf,"best_iteration") else "N/A"}\n')
        f.write(f'max_depth            : {args.max_depth}\n')
        f.write(f'learning_rate        : {args.learning_rate}\n')
        f.write(f'subsample            : {args.subsample}\n')
        f.write(f'colsample_bytree     : {args.colsample}\n')
        f.write(f'pesos de classe      : inverso da frequência\n\n')
        f.write('── VALIDAÇÃO ──\n')
        f.write(f'Acurácia   : {acc_v:.4f}\n')
        f.write(f'F1 weighted: {f1w_v:.4f}\n')
        f.write(f'F1 macro   : {f1m_v:.4f}\n\n')
        f.write(rep_v)
        f.write('\n── TESTE ──\n')
        f.write(f'Acurácia   : {acc_t:.4f}\n')
        f.write(f'F1 weighted: {f1w_t:.4f}\n')
        f.write(f'F1 macro   : {f1m_t:.4f}\n\n')
        f.write(rep_t)
    print(f'\nReport salvo: {report_path}')

    print('\nGerando gráficos...')
    y_true_xgb  = np.array([label_map[c] for c in y_test],  dtype='int32')
    plot_confusion(y_true_xgb, y_test_pred, classes_orig, MODEL_DIR / 'xgb_baseline_confusion.png')
    importances = plot_importance(clf, feat_cols, MODEL_DIR / 'xgb_baseline_importance.png')

    model_path = MODEL_DIR / 'xgb_baseline.joblib'
    joblib.dump(clf, model_path, compress=3)
    print(f'  Modelo salvo: {model_path} ({model_path.stat().st_size // 1024} KB)')

    metrics = {
        'model':            'XGBClassifier',
        'n_estimators_max': args.n_estimators,
        'best_iteration':   int(clf.best_iteration) + 1 if hasattr(clf, 'best_iteration') else None,
        'max_depth':        args.max_depth,
        'learning_rate':    args.learning_rate,
        'class_weight':     'inverse_frequency',
        'val':  {'accuracy': round(acc_v,4), 'f1_weighted': round(f1w_v,4), 'f1_macro': round(f1m_v,4)},
        'test': {'accuracy': round(acc_t,4), 'f1_weighted': round(f1w_t,4), 'f1_macro': round(f1m_t,4)},
        'top5_features': importances.head(5).index.tolist(),
    }
    metrics_path = MODEL_DIR / 'xgb_baseline_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f'  Métricas JSON: {metrics_path}')

    print('\n' + '='*55)
    print('  RESUMO FINAL — XGBOOST BASELINE')
    print('='*55)
    print(f'  Validação  | Acurácia={acc_v:.4f}  F1w={f1w_v:.4f}  F1mac={f1m_v:.4f}')
    print(f'  Teste      | Acurácia={acc_t:.4f}  F1w={f1w_t:.4f}  F1mac={f1m_t:.4f}')
    print('='*55)
    print(f'\nArquivos em: {MODEL_DIR}')

if __name__ == '__main__':
    main()
