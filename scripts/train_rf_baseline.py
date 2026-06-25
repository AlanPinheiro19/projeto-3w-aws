"""
train_rf_baseline.py
====================
Treina Random Forest baseline usando a camada Gold do pipeline 3W.

Uso:
    python scripts/train_rf_baseline.py
    python scripts/train_rf_baseline.py --n-estimators 200 --verbose

Saídas (em models/):
    rf_baseline.joblib          modelo treinado
    rf_baseline_report.txt      classification report (val + test)
    rf_baseline_confusion.png   matriz de confusão (test)
    rf_baseline_importance.png  top-30 features por importância
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
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ── Resolve PROJECT_DIR ───────────────────────────────────────────────────────
def _find_project_dir() -> Path:
    if 'PROJECT_DIR' in os.environ:
        return Path(os.environ['PROJECT_DIR'])
    candidates = [
        r'D:\Alan\Alan\Pos GRADUAÇÂO USP\eEDB-007 - Trabalho de Conclusão\projeto-3w-aws',
        r'C:\projeto-3w-aws',
        Path(__file__).parent.parent,   # scripts/ -> projeto-3w-aws/
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
    0: 'Normal',
    1: 'BSW Abrupt',
    2: 'DHSV Closure',
    3: 'Severe Slugging',
    4: 'Flow Instability',
    5: 'Prod. Loss',
    6: 'PCK Restriction',
    7: 'Scaling',
    8: 'Hydrate',
}

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Random Forest baseline — 3W dataset')
    p.add_argument('--n-estimators', type=int, default=150,
                   help='Número de árvores (default: 150)')
    p.add_argument('--max-depth', type=int, default=None,
                   help='Profundidade máxima (default: None = irrestrito)')
    p.add_argument('--min-samples-leaf', type=int, default=5,
                   help='Min samples por folha (default: 5)')
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='Threads paralelas (default: -1 = todos os cores)')
    p.add_argument('--max-samples', type=int, default=500_000,
                   help='Máx amostras de treino (default: 500000; 0 = sem limite)')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()

# ── Carregamento ──────────────────────────────────────────────────────────────
def load_data(max_samples: int = 500_000):
    import pyarrow.parquet as pq

    print('Carregando dados Gold...')
    print(f'  Diretório Gold: {GOLD_DIR}')

    def _read_limited(path, max_rows):
        """Lê apenas max_rows linhas do parquet sem carregar o arquivo inteiro na RAM."""
        if max_rows:
            pf = pq.ParquetFile(str(path))
            total = pf.metadata.num_rows
            if total > max_rows:
                batch = next(pf.iter_batches(batch_size=max_rows))
                df = batch.to_pandas()
                print(f'  [SAMPLING] {path.name}: {total:,} → {len(df):,} linhas lidas do disco')
                return df
        return pd.read_parquet(path)

    train = _read_limited(GOLD_DIR / 'train.parquet', max_samples)
    val   = pd.read_parquet(GOLD_DIR / 'val.parquet')
    test  = pd.read_parquet(GOLD_DIR / 'test.parquet')
    print(f'  train={len(train):,}  val={len(val):,}  test={len(test):,}')

    # ── Fallback: se splits estão vazios, faz split manual ───────────────────
    total = len(train) + len(val) + len(test)
    if total == 0:
        raise FileNotFoundError(
            f'Todos os arquivos Gold estão vazios em {GOLD_DIR}.\n'
            'Re-execute o pipeline Gold no Airflow ou rode:\n'
            '  python scripts/etl_bronze_silver_gold.py --step gold_final'
        )

    if len(train) == 0:
        print('\n  [AVISO] train.parquet vazio — fazendo split manual a partir do Gold completo...')
        all_data = pd.concat([train, val, test], ignore_index=True)
        if len(all_data) == 0:
            # Tenta carregar do Silver como último recurso
            print('  [AVISO] Gold vazio — tentando silver...')
            silver_dir = PROJECT_DIR / 'data' / 'silver'
            files = sorted(silver_dir.glob('*.parquet'))
            if not files:
                raise FileNotFoundError(f'Silver também vazio em {silver_dir}')
            all_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            print(f'  Silver carregado: {len(all_data):,} linhas')

        n = len(all_data)
        i70, i85 = int(n * 0.70), int(n * 0.85)
        train = all_data.iloc[:i70]
        val   = all_data.iloc[i70:i85]
        test  = all_data.iloc[i85:]
        print(f'  Split manual: train={len(train):,}  val={len(val):,}  test={len(test):,}')

    from sklearn.model_selection import train_test_split

    feat_cols = [c for c in train.columns if c.endswith('_norm')]
    if not feat_cols:
        sensor_cols = ['P-PDG','P-TPT','T-TPT','P-MON-CKP','T-JUS-CKP','P-JUS-CKGL','T-JUS-CKGL','QGL']
        feat_cols = [c for c in sensor_cols if c in train.columns]
        print(f'  [AVISO] Sem features _norm — usando {len(feat_cols)} sensores brutos')
    print(f'  Features: {len(feat_cols)}')

    # ── Fix: split estratificado quando o split temporal deixa classes fora do treino ──
    all_data      = pd.concat([train, val, test], ignore_index=True)
    classes_all   = set(all_data['class'].unique())
    classes_train = set(train['class'].unique())
    missing       = classes_all - classes_train

    if missing:
        print(f'\n  [AVISO] Split temporal excluiu {len(missing)} classe(s) do treino: '
              f'{sorted(int(c) for c in missing)}')
        print('  → Aplicando split ESTRATIFICADO por classe (15% val, 15% test)...')
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
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        if nan_count or inf_count:
            print(f'  [AVISO] {name}: {nan_count} NaN, {inf_count} Inf → substituídos por 0')
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

# ── Treinamento ───────────────────────────────────────────────────────────────
def train_model(X_train, y_train, args):
    from sklearn.ensemble import RandomForestClassifier

    print(f'\nTreinando Random Forest ({args.n_estimators} árvores, class_weight=balanced)...')
    t0 = time.time()

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=args.n_jobs,
        verbose=1 if args.verbose else 0,
    )
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        import traceback
        print(f'\nERRO NO FIT: {type(e).__name__}: {e}')
        traceback.print_exc()
        raise
    elapsed = time.time() - t0
    print(f'  Treinamento concluído em {elapsed:.1f}s')
    return clf

# ── Avaliação ─────────────────────────────────────────────────────────────────
def evaluate(clf, X, y, split_name, classes):
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    y_pred = clf.predict(X)
    acc    = accuracy_score(y, y_pred)
    f1_w   = f1_score(y, y_pred, average='weighted', zero_division=0)
    f1_mac = f1_score(y, y_pred, average='macro', zero_division=0)
    report = classification_report(
        y, y_pred,
        labels=classes,
        target_names=[EVENT_LABELS.get(c, f'Cls{c}') for c in classes],
        zero_division=0,
    )
    print(f'\n── {split_name} ──')
    print(f'  Acurácia:        {acc:.4f}')
    print(f'  F1 weighted:     {f1_w:.4f}')
    print(f'  F1 macro:        {f1_mac:.4f}')
    print(report)
    return y_pred, acc, f1_w, f1_mac, report

# ── Matriz de Confusão ────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, classes, out_path):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    labels = [EVENT_LABELS.get(c, f'Cls{c}') for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Matriz de Confusão — Random Forest Baseline (Teste)', fontsize=13, fontweight='bold')

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ['d', '.1f'],
        ['Contagem Absoluta', 'Percentual por Classe Real (%)']
    ):
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        thresh = data.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = f'{data[i,j]:{fmt}}'
                ax.text(j, i, val, ha='center', va='center',
                        color='white' if data[i,j] > thresh else 'black', fontsize=7)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Salvo: {out_path.name}')

# ── Feature Importance ────────────────────────────────────────────────────────
def plot_importance(clf, feat_cols, out_path, top_n=30):
    importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1], color='#1F5C99', edgecolor='white')
    ax.set_title(f'Top {top_n} Features — Random Forest Baseline', fontweight='bold')
    ax.set_xlabel('Importância (Mean Decrease Impurity)')
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Salvo: {out_path.name}')
    return importances

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    try:
        import sklearn, joblib
    except ImportError:
        print('ERRO: instale scikit-learn e joblib:')
        print('  pip install scikit-learn joblib')
        sys.exit(1)

    print(f'PROJECT_DIR: {PROJECT_DIR}')
    print(f'scikit-learn: {sklearn.__version__}')

    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = load_data(
        max_samples=args.max_samples
    )
    clf = train_model(X_train, y_train, args)

    classes = sorted(np.unique(np.concatenate([y_train, y_val, y_test])))

    # Avalia em val e test
    y_val_pred,  acc_v, f1w_v, f1m_v, rep_v = evaluate(clf, X_val,  y_val,  'VALIDAÇÃO', classes)
    y_test_pred, acc_t, f1w_t, f1m_t, rep_t = evaluate(clf, X_test, y_test, 'TESTE',     classes)

    # Salva report
    report_path = MODEL_DIR / 'rf_baseline_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== RANDOM FOREST BASELINE — 3W DATASET ===\n\n')
        f.write(f'n_estimators     : {args.n_estimators}\n')
        f.write(f'max_depth        : {args.max_depth}\n')
        f.write(f'min_samples_leaf : {args.min_samples_leaf}\n')
        f.write(f'class_weight     : balanced\n\n')
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

    # Gráficos
    print('\nGerando gráficos...')
    plot_confusion(y_test, y_test_pred, classes, MODEL_DIR / 'rf_baseline_confusion.png')
    importances = plot_importance(clf, feat_cols, MODEL_DIR / 'rf_baseline_importance.png')

    # Salva modelo
    model_path = MODEL_DIR / 'rf_baseline.joblib'
    joblib.dump(clf, model_path, compress=3)
    print(f'  Modelo salvo: {model_path} ({model_path.stat().st_size//1024} KB)')

    # Salva métricas resumidas em JSON
    metrics = {
        'model': 'RandomForestClassifier',
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'class_weight': 'balanced',
        'val':  {'accuracy': round(acc_v,4), 'f1_weighted': round(f1w_v,4), 'f1_macro': round(f1m_v,4)},
        'test': {'accuracy': round(acc_t,4), 'f1_weighted': round(f1w_t,4), 'f1_macro': round(f1m_t,4)},
        'top5_features': importances.head(5).index.tolist(),
    }
    metrics_path = MODEL_DIR / 'rf_baseline_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f'  Métricas JSON: {metrics_path}')

    print('\n' + '='*55)
    print('  RESUMO FINAL — RANDOM FOREST BASELINE')
    print('='*55)
    print(f'  Validação  | Acurácia={acc_v:.4f}  F1w={f1w_v:.4f}  F1mac={f1m_v:.4f}')
    print(f'  Teste      | Acurácia={acc_t:.4f}  F1w={f1w_t:.4f}  F1mac={f1m_t:.4f}')
    print('='*55)
    print(f'\nArquivos em: {MODEL_DIR}')

if __name__ == '__main__':
    main()
