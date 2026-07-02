"""
tune_thresholds.py
==================
Threshold tuning por classe para os modelos baseline 3W Petrobras.

Estratégia:
  - Carrega RF, XGBoost e LightGBM já treinados (.joblib)
  - Obtém probabilidades predict_proba() no conjunto de VALIDAÇÃO
  - Para cada modelo, faz grid search de threshold por classe
    minimizando F1 Macro (busca o ponto ótimo precisão/recall por classe)
  - Aplica thresholds otimizados no TESTE e compara com baseline argmax

Regra de decisão com threshold:
  Para cada amostra, calcula score[c] = P(c) / threshold[c]
  Prediz classe = argmax(score)  (equivale a: qual classe "excedeu" mais seu limiar)

  Intuição: se threshold[DHSV]=0.10 e P(DHSV)=0.12 → score=1.2 → classe selecionada
            se threshold[Normal]=0.90 e P(Normal)=0.80 → score=0.89 → não selecionada

Uso:
    python scripts/tune_thresholds.py
    python scripts/tune_thresholds.py --steps 50 --model lgb
    python scripts/tune_thresholds.py --steps 100 --verbose

Saídas (em models/):
    threshold_tuning_report.txt     relatório completo antes/depois por modelo
    threshold_tuning_results.json   thresholds ótimos + métricas em JSON
    threshold_tuning_curves.png     curva F1 Macro × threshold para classe problemática (DHSV)
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

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Threshold tuning por classe — 3W dataset')
    p.add_argument('--steps',   type=int,   default=50,
                   help='Resolução do grid search (default: 50 → passos de 0.02)')
    p.add_argument('--model',   type=str,   default='all',
                   choices=['all', 'rf', 'xgb', 'lgb'],
                   help='Modelo a tunar (default: all)')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()

# ── Carregamento de dados ──────────────────────────────────────────────────────
def load_data():
    """Carrega splits Gold em chunks para evitar OOM.
    Aplica split estratificado quando val/test estao com classes ausentes
    (split temporal concentra eventos em janelas especificas).
    """
    from sklearn.model_selection import train_test_split
    import pyarrow.parquet as pq

    print('Carregando dados Gold...')
    print(f'  Diretorio Gold: {GOLD_DIR}')

    CHUNK_SIZE = 30_000
    MAX_TRAIN  = 300_000
    MAX_VAL    = MAX_TRAIN // 3
    MAX_TEST   = MAX_TRAIN // 3

    def _read_chunked(path, max_rows, class_col='class'):
        pf    = pq.ParquetFile(str(path))
        total = pf.metadata.num_rows
        rate  = (max_rows / total) if (max_rows and total > max_rows) else 1.0
        chunks, rows = [], 0
        for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
            chunk = batch.to_pandas()
            if rate < 1.0 and class_col in chunk.columns:
                chunk = (chunk.groupby(class_col, group_keys=False)
                              .apply(lambda g: g.sample(
                                  n=max(1, round(len(g) * rate)), random_state=42))
                              .reset_index(drop=True))
            elif rate < 1.0:
                chunk = chunk.sample(n=max(1, round(len(chunk) * rate)), random_state=42)
            chunks.append(chunk)
            rows += len(chunk)
            if max_rows and rows >= max_rows:
                break
        result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        tag = f'{total:,} -> {len(result):,} (amostrado)' if rate < 1.0 else f'{len(result):,}'
        print(f'  {path.name}: {tag} linhas')
        return result

    val   = _read_chunked(GOLD_DIR / 'val.parquet',   MAX_VAL)
    test  = _read_chunked(GOLD_DIR / 'test.parquet',  MAX_TEST)
    train = _read_chunked(GOLD_DIR / 'train.parquet', MAX_TRAIN)
    print(f'  train={len(train):,}  val={len(val):,}  test={len(test):,}')

    feat_cols = [c for c in train.columns if c.endswith('_norm')]
    if not feat_cols:
        sensor_cols = ['P-PDG','P-TPT','T-TPT','P-MON-CKP','T-JUS-CKP','P-JUS-CKGL','T-JUS-CKGL','QGL']
        feat_cols = [c for c in sensor_cols if c in train.columns]
    print(f'  Features: {len(feat_cols)}')

    # Split estratificado quando val/test nao tem todas as classes
    all_data      = pd.concat([train, val, test], ignore_index=True)
    classes_all   = set(all_data['class'].unique())
    classes_train = set(train['class'].unique())
    classes_val   = set(val['class'].unique())
    classes_test  = set(test['class'].unique())
    missing = ((classes_all - classes_train)
               | (classes_all - classes_val)
               | (classes_all - classes_test))

    if missing:
        faltando = sorted(int(c) for c in missing)
        print(f'  [INFO] Aplicando split ESTRATIFICADO (classes ausentes: {faltando})...')
        y_all = all_data['class'].values.astype('int32')
        X_all = all_data[feat_cols].values.astype('float32')

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=round(0.15/0.85, 6), random_state=42, stratify=y_tmp)
        print(f'  Estratificado: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}')
    else:
        X_train = train[feat_cols].values.astype('float32')
        y_train = train['class'].values.astype('int32')
        X_val   = val[feat_cols].values.astype('float32')
        y_val   = val['class'].values.astype('int32')
        X_test  = test[feat_cols].values.astype('float32')
        y_test  = test['class'].values.astype('int32')

    def _clean(X):
        bad = int((~np.isfinite(X)).sum())
        if bad:
            print(f'  [AVISO] {bad} NaN/Inf substituidos por 0')
        return np.where(np.isfinite(X), X, np.float32(0.0)).astype('float32')

    return (_clean(X_train), y_train,
            _clean(X_val),   y_val,
            _clean(X_test),  y_test,
            feat_cols)

# ── Carregamento de modelos ────────────────────────────────────────────────────
def load_models(model_filter):
    import joblib
    models = {}
    specs  = [
        ('rf',  'rf_baseline.joblib'),
        ('xgb', 'xgb_baseline.joblib'),
        ('lgb', 'lgb_baseline.joblib'),
    ]
    for key, fname in specs:
        if model_filter != 'all' and key != model_filter:
            continue
        path = MODEL_DIR / fname
        if not path.exists():
            print(f'  [AVISO] Modelo não encontrado: {path} — pulando')
            continue
        print(f'  Carregando {fname}...')
        models[key] = joblib.load(path)
    return models

# ── Regra de decisão com threshold ────────────────────────────────────────────
def predict_with_thresholds(proba, thresholds, classes):
    """
    proba      : (n_samples, n_classes) — probabilidades brutas do modelo
    thresholds : dict {class_int: threshold_float}
    classes    : lista ordenada das classes (índice → classe real)

    Regra: score[c] = P(c) / threshold[c]
           predição = classes[argmax(score)]
    """
    t = np.array([thresholds.get(int(c), 1.0) for c in classes], dtype='float32')
    # evita divisão por zero
    t = np.clip(t, 1e-6, 1.0)
    scores = proba / t[np.newaxis, :]
    idx    = np.argmax(scores, axis=1)
    return np.array([int(classes[i]) for i in idx], dtype='int32')

# ── Grid search de threshold ───────────────────────────────────────────────────
def tune_thresholds(proba, y_true, classes, n_steps=50, verbose=False):
    """
    Otimiza threshold por classe usando grid search sequencial.
    Para cada classe, varre [0.01, 1.0] e escolhe o threshold que
    maximiza o F1 Macro global (mantendo os demais fixos).

    Retorna: dict {class_int: best_threshold}
    """
    from sklearn.metrics import f1_score

    # Inicia com threshold = fração base de cada classe (prior calculado de y_true)
    unique_y, counts_y = np.unique(y_true.astype('int32'), return_counts=True)
    total_y   = float(counts_y.sum())
    prior_map = {int(u): float(c / total_y) for u, c in zip(unique_y, counts_y)}
    # Fallback uniforme para classes que o modelo conhece mas não estão em y_true
    best_t = {int(c): prior_map.get(int(c), 1.0 / len(classes)) for c in classes}

    grid = np.linspace(0.01, 1.0, n_steps)

    for i, cls in enumerate(classes):
        cls = int(cls)
        best_f1  = -1.0
        best_val = best_t[cls]

        for t in grid:
            trial = dict(best_t)
            trial[cls] = float(t)
            y_pred = predict_with_thresholds(proba, trial, classes)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_val = float(t)

        if verbose:
            label = EVENT_LABELS.get(cls, str(cls))
            print(f'    Classe {cls} ({label:<18}): threshold={best_val:.3f}  '
                  f'→ F1mac={best_f1:.4f}')
        best_t[cls] = best_val

    return best_t

# ── Avaliação comparativa ──────────────────────────────────────────────────────
def evaluate_pair(proba, y_true, thresholds, classes, split_name, lines):
    from sklearn.metrics import (classification_report, f1_score,
                                 accuracy_score, confusion_matrix)

    # Baseline: argmax puro
    y_base = np.array([int(classes[i]) for i in np.argmax(proba, axis=1)], dtype='int32')
    # Com thresholds
    y_tuned = predict_with_thresholds(proba, thresholds, classes)

    acc_b  = accuracy_score(y_true, y_base)
    f1m_b  = f1_score(y_true, y_base,  average='macro',    zero_division=0)
    f1w_b  = f1_score(y_true, y_base,  average='weighted', zero_division=0)
    acc_t  = accuracy_score(y_true, y_tuned)
    f1m_t  = f1_score(y_true, y_tuned, average='macro',    zero_division=0)
    f1w_t  = f1_score(y_true, y_tuned, average='weighted', zero_division=0)

    delta_f1m = f1m_t - f1m_b

    hdr = f'\n── {split_name} ──'
    lines.append(hdr)
    print(hdr)

    row = (f'  {"":20s}  {"Acc":>7}  {"F1w":>7}  {"F1mac":>7}')
    lines.append(row); print(row)
    row = (f'  {"Baseline (argmax)":20s}  {acc_b:7.4f}  {f1w_b:7.4f}  {f1m_b:7.4f}')
    lines.append(row); print(row)
    row = (f'  {"Threshold tuning":20s}  {acc_t:7.4f}  {f1w_t:7.4f}  {f1m_t:7.4f}  '
           f'(Δ F1mac={delta_f1m:+.4f})')
    lines.append(row); print(row)

    lines.append(f'\n  Classification report — Threshold tuning ({split_name}):')
    print(f'\n  Classification report — Threshold tuning ({split_name}):')
    cr = classification_report(
        y_true, y_tuned,
        labels=list(classes),
        target_names=[EVENT_LABELS.get(int(c), str(c)) for c in classes],
        zero_division=0,
    )
    for l in cr.splitlines():
        lines.append('  ' + l); print('  ' + l)

    return {
        'baseline': {'acc': acc_b, 'f1w': f1w_b, 'f1mac': f1m_b},
        'tuned':    {'acc': acc_t, 'f1w': f1w_t, 'f1mac': f1m_t},
        'delta_f1mac': delta_f1m,
        'confusion_matrix': confusion_matrix(y_true, y_tuned,
                                             labels=list(classes)).tolist(),
    }

# ── Curva threshold × F1 para DHSV (classe 2) ─────────────────────────────────
def plot_dhsv_curve(model_results):
    """Plota como o threshold da DHSV afeta Precisão, Recall e F1 para cada modelo."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    fig, axes = plt.subplots(1, len(model_results), figsize=(5 * len(model_results), 4),
                             squeeze=False)
    grid = np.linspace(0.01, 1.0, 80)

    for ax, (mname, data) in zip(axes[0], model_results.items()):
        proba      = data['proba_val']
        y_val      = data['y_val']
        classes    = data['classes']
        best_t     = data['thresholds']
        dhsv_idx   = list(classes).index(2) if 2 in classes else None

        if dhsv_idx is None:
            ax.set_title(f'{mname.upper()}\n(DHSV não encontrada)')
            continue

        precs, recs, f1s = [], [], []
        for t in grid:
            trial = dict(best_t)
            trial[2] = float(t)
            y_pred = predict_with_thresholds(proba, trial, classes)
            precs.append(precision_score(y_val, y_pred, labels=[2], average='macro', zero_division=0))
            recs.append(recall_score(y_val,  y_pred, labels=[2], average='macro', zero_division=0))
            f1s.append(f1_score(y_val,    y_pred, labels=[2], average='macro', zero_division=0))

        ax.plot(grid, precs, label='Precisão', color='#1f77b4')
        ax.plot(grid, recs,  label='Recall',   color='#ff7f0e')
        ax.plot(grid, f1s,   label='F1',       color='#2ca02c', linewidth=2)
        ax.axvline(best_t.get(2, 0.5), color='red', linestyle='--', label=f"Ótimo ({best_t.get(2, 0.5):.2f})")
        ax.set_title(f'{mname.upper()} — DHSV Closure (classe 2)')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Curva Threshold × Métricas — DHSV Closure', fontsize=13, y=1.02)
    plt.tight_layout()
    out = MODEL_DIR / 'threshold_dhsv_curve.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Salvo: {out.name}')

# ── Matriz de confusão comparativa ────────────────────────────────────────────
def plot_confusion_comparison(model_results, split='test'):
    """Plota matrizes de confusão lado a lado: baseline vs. tuned para cada modelo."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    n_models = len(model_results)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (mname, data) in enumerate(model_results.items()):
        proba   = data[f'proba_{split}']
        y_true  = data[f'y_{split}']
        classes = data['classes']
        labels  = [EVENT_LABELS.get(int(c), str(c)) for c in classes]

        y_base  = np.array([int(classes[i]) for i in np.argmax(proba, axis=1)])
        y_tuned = predict_with_thresholds(proba, data['thresholds'], classes)

        for col, (y_pred, title) in enumerate([
            (y_base,  f'{mname.upper()} — Baseline'),
            (y_tuned, f'{mname.upper()} — Tuned'),
        ]):
            cm = confusion_matrix(y_true, y_pred, labels=list(classes))
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
                ax=axes[row, col], colorbar=False, xticks_rotation=45)
            axes[row, col].set_title(title, fontsize=11)

    plt.tight_layout()
    out = MODEL_DIR / f'threshold_confusion_{split}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Salvo: {out.name}')

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f'scripts/tune_thresholds.py')
    print(f'PROJECT_DIR : {PROJECT_DIR}')

    args = parse_args()

    # 1. Dados
    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = load_data()

    # 2. Modelos
    print('\nCarregando modelos...')
    models = load_models(args.model)
    if not models:
        print('\nERRO: Nenhum modelo encontrado em models/. Execute os scripts de treino primeiro.')
        sys.exit(1)

    lines       = []
    all_results = {}
    model_data  = {}   # para os gráficos

    for mname, clf in models.items():
        sep = '=' * 58
        header = f'\n{sep}\n  MODELO: {mname.upper()}\n{sep}'
        lines.append(header); print(header)

        # 3. Probabilidades
        print('  Calculando predict_proba...')
        try:
            proba_val  = clf.predict_proba(X_val)
            proba_test = clf.predict_proba(X_test)
        except Exception as e:
            print(f'  ERRO ao calcular proba: {e}')
            continue

        # Classes conhecidas pelo modelo
        if hasattr(clf, 'classes_'):
            classes = clf.classes_
        else:
            classes = np.arange(proba_val.shape[1])

        # Para XGBoost: detectar remapeamento de labels contiguos -> originais
        # O XGB treina com y remapeado para [0..n-1]; clf.classes_ pode nao refletir
        # os labels originais. So remapeia quando o NUMERO de classes for igual
        # (se diferente, o modelo foi treinado com menos classes que os dados atuais).
        if mname == 'xgb':
            clf_classes_sorted = np.array(sorted(int(c) for c in classes))
            data_classes       = np.array(sorted(int(c) for c in np.unique(y_val)))
            if (len(clf_classes_sorted) == len(data_classes)
                    and not np.array_equal(clf_classes_sorted, data_classes)):
                classes = data_classes
                print(f'  [XGB] Remapeamento detectado: modelo={list(clf_classes_sorted)} '
                      f'-> originais={list(classes)}')
            elif len(clf_classes_sorted) != len(data_classes):
                print(f'  [XGB] Modelo tem {len(clf_classes_sorted)} classes, '
                      f'dados tem {len(data_classes)} - usando classes do modelo')

        print(f'  Classes: {list(classes)}')

        # 4. Tuning no conjunto de validação
        print(f'\n  Otimizando thresholds na validação ({args.steps} passos)...')
        best_t = tune_thresholds(
            proba_val, y_val, classes, n_steps=args.steps, verbose=args.verbose)

        t_row = '  Thresholds ótimos:'
        lines.append(t_row); print(t_row)
        for cls, t in sorted(best_t.items()):
            label = EVENT_LABELS.get(int(cls), str(cls))
            row = f'    Classe {cls:1d} ({label:<18}): {t:.3f}'
            lines.append(row); print(row)

        # 5. Avaliação
        res_val  = evaluate_pair(proba_val,  y_val,  best_t, classes, 'VALIDAÇÃO', lines)
        res_test = evaluate_pair(proba_test, y_test, best_t, classes, 'TESTE',     lines)

        all_results[mname] = {
            'thresholds': best_t,
            'validation': res_val,
            'test':       res_test,
        }
        model_data[mname] = {
            'proba_val':  proba_val,
            'proba_test': proba_test,
            'y_val':      y_val,
            'y_test':     y_test,
            'classes':    classes,
            'thresholds': best_t,
        }

    # 6. Resumo final
    sep = '=' * 58
    lines.append(f'\n{sep}')
    lines.append('  RESUMO — DELTA F1 MACRO (tuned − baseline)')
    lines.append(sep)
    print(f'\n{sep}')
    print('  RESUMO — DELTA F1 MACRO (tuned − baseline)')
    print(sep)

    hdr = f'  {"Modelo":8s}  {"Val base":>9}  {"Val tuned":>9}  {"ΔVal":>7}  '
    hdr += f'{"Test base":>9}  {"Test tuned":>9}  {"ΔTest":>7}'
    lines.append(hdr); print(hdr)

    for mname, res in all_results.items():
        v = res['validation']
        t = res['test']
        row = (f'  {mname.upper():8s}  '
               f'{v["baseline"]["f1mac"]:9.4f}  {v["tuned"]["f1mac"]:9.4f}  '
               f'{v["delta_f1mac"]:+7.4f}  '
               f'{t["baseline"]["f1mac"]:9.4f}  {t["tuned"]["f1mac"]:9.4f}  '
               f'{t["delta_f1mac"]:+7.4f}')
        lines.append(row); print(row)
    lines.append(sep); print(sep)

    # 7. Salvar relatório e JSON
    report_path = MODEL_DIR / 'threshold_tuning_report.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n  Relatório: {report_path}')

    json_path = MODEL_DIR / 'threshold_tuning_results.json'
    # Serializa numpy arrays
    def np_serial(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):    return obj.tolist()
        raise TypeError(f'{type(obj)}')
    json_path.write_text(
        json.dumps(all_results, indent=2, default=np_serial), encoding='utf-8')
    print(f'  JSON:      {json_path}')

    # 8. Gráficos
    print('\nGerando gráficos...')
    if model_data:
        plot_dhsv_curve(model_data)
        plot_confusion_comparison(model_data, split='test')

    print('\nThreshold tuning concluído!')

if __name__ == '__main__':
    main()
