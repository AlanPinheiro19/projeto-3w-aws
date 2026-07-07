"""
train_lgb_baseline.py
=====================
Treina LightGBM baseline para classificação de eventos em poços 3W Petrobras.

Diferenciais vs Random Forest e XGBoost:
  - Crescimento de árvore leaf-wise (vs level-wise do XGBoost): mais preciso em dados densos
  - is_unbalance=True: ajuste automático de pesos sem precisar calcular manualmente
  - Treino mais rápido (~5-10x vs XGBoost em datasets grandes)
  - Menos hiperparâmetros críticos para ajustar no baseline

Uso:
    python scripts/train_lgb_baseline.py
    python scripts/train_lgb_baseline.py --n-estimators 1000 --num-leaves 63 --verbose

Saídas (em models/):
    lgb_baseline.joblib              modelo treinado
    lgb_baseline_report.txt          classification report (val + test)
    lgb_baseline_confusion.png       matriz de confusão (test)
    lgb_baseline_importance.png      top-30 features por ganho
    lgb_baseline_metrics.json        métricas resumidas
    lgb_baseline_learning_curve.png  curva de aprendizado (train vs val loss)
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
    p = argparse.ArgumentParser(description='LightGBM baseline — 3W dataset')
    p.add_argument('--n-estimators',   type=int,   default=1000,
                   help='Número máximo de rounds (default: 1000; early stopping reduz)')
    p.add_argument('--num-leaves',     type=int,   default=31,
                   help='Número máximo de folhas por árvore (default: 31)')
    p.add_argument('--max-depth',      type=int,   default=-1,
                   help='Profundidade máx (-1 = irrestrito, default: -1)')
    p.add_argument('--learning-rate',  type=float, default=0.05,
                   help='Taxa de aprendizado (default: 0.05)')
    p.add_argument('--min-child-samples', type=int, default=20,
                   help='Min amostras por folha (default: 20; aumentar reduz overfitting)')
    p.add_argument('--subsample',      type=float, default=0.8,
                   help='Fração de amostras por árvore / bagging_fraction (default: 0.8)')
    p.add_argument('--colsample',      type=float, default=0.8,
                   help='Fração de features por árvore / feature_fraction (default: 0.8)')
    p.add_argument('--early-stopping', type=int,   default=50,
                   help='Rounds sem melhora para parar (default: 50)')
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

# ── Treinamento ───────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, args):
    try:
        import lightgbm as lgb
    except ImportError:
        print('\nERRO: lightgbm não instalado.')
        print('  pip install lightgbm --break-system-packages')
        print('  ou dentro do container: pip install lightgbm')
        sys.exit(1)

    n_classes = len(np.unique(y_train))

    print(f'\nTreinando LightGBM ({args.n_estimators} rounds max, '
          f'num_leaves={args.num_leaves}, lr={args.learning_rate})...')
    print(f'  {n_classes} classes | early_stopping={args.early_stopping} rounds | '
          f'is_unbalance=True')

    clf = lgb.LGBMClassifier(
        objective         = 'multiclass',
        num_class         = n_classes,
        metric            = 'multi_logloss',
        n_estimators      = args.n_estimators,
        num_leaves        = args.num_leaves,
        max_depth         = args.max_depth,
        learning_rate     = args.learning_rate,
        min_child_samples = args.min_child_samples,
        subsample         = args.subsample,
        subsample_freq    = 1,            # necessário para ativar bagging
        colsample_bytree  = args.colsample,
        reg_alpha         = 0.1,          # L1
        reg_lambda        = 1.0,          # L2
        is_unbalance      = True,         # peso automático por classe
        n_jobs            = args.n_jobs,
        random_state      = 42,
        verbosity         = 1 if args.verbose else -1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=args.verbose),
        lgb.log_evaluation(period=50 if args.verbose else -1),
    ]

    t0 = time.time()
    try:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    except Exception as e:
        import traceback
        print(f'\nERRO NO FIT: {type(e).__name__}: {e}')
        traceback.print_exc()
        raise

    elapsed = time.time() - t0
    best_iter = clf.best_iteration_ if hasattr(clf, 'best_iteration_') else args.n_estimators
    print(f'  Treinamento concluído em {elapsed:.1f}s | melhor iteração: {best_iter}')

    return clf

# ── Avaliação ─────────────────────────────────────────────────────────────────
def evaluate(clf, X, y, split_name, classes):
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    y_pred = clf.predict(X)
    acc    = accuracy_score(y, y_pred)
    f1_w   = f1_score(y, y_pred, average='weighted', zero_division=0)
    f1_mac = f1_score(y, y_pred, average='macro',    zero_division=0)
    report = classification_report(
        y, y_pred,
        labels=classes,
        target_names=[EVENT_LABELS.get(c, f'Cls{c}') for c in classes],
        zero_division=0,
    )
    print(f'\n── {split_name} ──')
    print(f'  Acurácia:    {acc:.4f}')
    print(f'  F1 weighted: {f1_w:.4f}')
    print(f'  F1 macro:    {f1_mac:.4f}')
    print(report)
    return y_pred, acc, f1_w, f1_mac, report

# ── Curva de Aprendizado ──────────────────────────────────────────────────────
def plot_learning_curve(clf, out_path):
    """Plota train vs val logloss ao longo das iterações."""
    try:
        results = clf.evals_result_
    except AttributeError:
        print('  [AVISO] evals_result_ não disponível — pulando curva de aprendizado')
        return

    if not results:
        return

    train_key = 'training'    if 'training'    in results else list(results.keys())[0]
    val_key   = 'valid_0'     if 'valid_0'     in results else list(results.keys())[-1]
    metric    = 'multi_logloss' if 'multi_logloss' in results.get(train_key, {}) else \
                list(results.get(train_key, {}).keys())[0] if results.get(train_key) else None

    if metric is None:
        return

    train_loss = results[train_key].get(metric, [])
    val_loss   = results[val_key].get(metric, [])

    if not train_loss:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss, label='Treino', color='#1F5C99', linewidth=1.2)
    if val_loss:
        ax.plot(val_loss, label='Validação', color='#E67E22', linewidth=1.2)
        if hasattr(clf, 'best_iteration_') and clf.best_iteration_:
            ax.axvline(clf.best_iteration_, color='green', linestyle='--',
                       linewidth=1, label=f'Melhor iter: {clf.best_iteration_}')
    ax.set_title('Curva de Aprendizado — LightGBM Baseline', fontweight='bold')
    ax.set_xlabel('Iteração')
    ax.set_ylabel(f'{metric}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Salvo: {out_path.name}')

# ── Matriz de Confusão ────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, classes, out_path):
    from sklearn.metrics import confusion_matrix

    cm     = confusion_matrix(y_true, y_pred, labels=classes)
    cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    labels = [EVENT_LABELS.get(c, f'Cls{c}') for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Matriz de Confusão — LightGBM Baseline (Teste)', fontsize=13, fontweight='bold')

    for ax, data, fmt, title in zip(
        axes, [cm, cm_pct], ['d', '.1f'],
        ['Contagem Absoluta', 'Percentual por Classe Real (%)']
    ):
        im = ax.imshow(data, interpolation='nearest', cmap='Greens')
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

# ── Feature Importance ────────────────────────────────────────────────────────
def plot_importance(clf, feat_cols, out_path, top_n=30):
    importances = pd.Series(
        clf.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)
    top = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1], color='#1E8B4C', edgecolor='white')
    ax.set_title(f'Top {top_n} Features — LightGBM Baseline (split gain)', fontweight='bold')
    ax.set_xlabel('Importância (Split Gain)')
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Salvo: {out_path.name}')
    return importances

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    import lightgbm as lgb
    import joblib

    print(f'PROJECT_DIR : {PROJECT_DIR}')
    print(f'lightgbm    : {lgb.__version__}')

    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = load_data(
        max_samples=args.max_samples
    )
    clf = train_model(X_train, y_train, X_val, y_val, args)

    classes = sorted(np.unique(np.concatenate([y_train, y_val, y_test])))

    y_val_pred,  acc_v, f1w_v, f1m_v, rep_v = evaluate(clf, X_val,  y_val,  'VALIDAÇÃO', classes)
    y_test_pred, acc_t, f1w_t, f1m_t, rep_t = evaluate(clf, X_test, y_test, 'TESTE',     classes)

    # Report
    report_path = MODEL_DIR / 'lgb_baseline_report.txt'
    best_iter = clf.best_iteration_ if hasattr(clf, 'best_iteration_') else args.n_estimators
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== LIGHTGBM BASELINE — 3W DATASET ===\n\n')
        f.write(f'n_estimators (max)   : {args.n_estimators}\n')
        f.write(f'best_iteration       : {best_iter}\n')
        f.write(f'num_leaves           : {args.num_leaves}\n')
        f.write(f'max_depth            : {args.max_depth}\n')
        f.write(f'learning_rate        : {args.learning_rate}\n')
        f.write(f'min_child_samples    : {args.min_child_samples}\n')
        f.write(f'subsample            : {args.subsample}\n')
        f.write(f'colsample_bytree     : {args.colsample}\n')
        f.write(f'is_unbalance         : True\n\n')
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
    plot_confusion(y_test, y_test_pred, classes, MODEL_DIR / 'lgb_baseline_confusion.png')
    importances = plot_importance(clf, feat_cols, MODEL_DIR / 'lgb_baseline_importance.png')
    plot_learning_curve(clf, MODEL_DIR / 'lgb_baseline_learning_curve.png')

    model_path = MODEL_DIR / 'lgb_baseline.joblib'
    joblib.dump(clf, model_path, compress=3)
    print(f'  Modelo salvo: {model_path} ({model_path.stat().st_size // 1024} KB)')

    metrics = {
        'model':            'LGBMClassifier',
        'n_estimators_max': args.n_estimators,
        'best_iteration':   int(best_iter) if best_iter else None,
        'num_leaves':       args.num_leaves,
        'max_depth':        args.max_depth,
        'learning_rate':    args.learning_rate,
        'class_weight':     'is_unbalance=True',
        'val':  {'accuracy': round(acc_v,4), 'f1_weighted': round(f1w_v,4), 'f1_macro': round(f1m_v,4)},
        'test': {'accuracy': round(acc_t,4), 'f1_weighted': round(f1w_t,4), 'f1_macro': round(f1m_t,4)},
        'top5_features': importances.head(5).index.tolist(),
    }
    metrics_path = MODEL_DIR / 'lgb_baseline_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f'  Métricas JSON: {metrics_path}')

    print('\n' + '='*55)
    print('  RESUMO FINAL — LIGHTGBM BASELINE')
    print('='*55)
    print(f'  Validação  | Acurácia={acc_v:.4f}  F1w={f1w_v:.4f}  F1mac={f1m_v:.4f}')
    print(f'  Teste      | Acurácia={acc_t:.4f}  F1w={f1w_t:.4f}  F1mac={f1m_t:.4f}')
    print('='*55)
    print(f'\nArquivos em: {MODEL_DIR}')

if __name__ == '__main__':
    main()
