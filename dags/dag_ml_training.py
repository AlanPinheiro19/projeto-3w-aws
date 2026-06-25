"""
dag_ml_training.py
==================
DAG de treinamento completo dos modelos ML para detecção de eventos 3W Petrobras.

Fluxo:
    verificar_gold
        ├── treinar_rf  ──┐
        ├── treinar_xgb ──┼── threshold_tuning ── shap_analysis ── relatorio_ml
        └── treinar_lgb ──┘

RF, XGB e LGB treinam em PARALELO (pool separado por modelo).
Threshold tuning aguarda os 3 modelos.
SHAP analysis roda após LGB (melhor modelo) E após threshold tuning.

Trigger:    manual (schedule=None)
Pool:       ml_training_pool  (max 2 slots — evita OOM com XGB + LGB simultâneos)

Para executar:
    Airflow UI → DAGs → dag_ml_training → Trigger DAG
    ou: airflow dags trigger dag_ml_training
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Configuração ───────────────────────────────────────────────────────────────
DEFAULT_PROJECT_DIR = (
    r'D:\Alan\Alan\Pos GRADUAÇÂO USP\eEDB-007 - Trabalho de Conclusão\projeto-3w-aws'
)
PROJECT_DIR = os.environ.get('PROJECT_DIR', DEFAULT_PROJECT_DIR)
SCRIPTS_DIR = str(Path(PROJECT_DIR) / 'scripts')
MODELS_DIR  = str(Path(PROJECT_DIR) / 'models')
GOLD_DIR    = str(Path(PROJECT_DIR) / 'data' / 'gold')

PYTHON_BIN  = os.environ.get('PYTHON_BIN', 'python')   # ou 'python3' / caminho absoluto

DEFAULT_ARGS = {
    'owner':            'alan',
    'depends_on_past':  False,
    'retries':          0,
    'retry_delay':      timedelta(minutes=5),
    'email_on_failure': False,
}

# ── DAG ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id          = 'dag_ml_training',
    description     = 'Treinamento completo RF + XGB + LGB + Threshold Tuning + SHAP',
    default_args    = DEFAULT_ARGS,
    start_date      = datetime(2026, 1, 1),
    schedule        = None,          # somente trigger manual
    catchup         = False,
    max_active_runs = 1,
    tags            = ['ml', 'training', '3w'],
    params          = {
        'modelos':        'all',     # 'all' | 'rf' | 'xgb' | 'lgb'
        'n_estimators_rf':   100,    # reduzido para t3.large (8GB RAM)
        'n_estimators_xgb':  200,    # early stopping para mais cedo (era 500)
        'n_estimators_lgb':  500,    # early stopping reduz na prática (era 1000)
        'threshold_steps':    50,
        'shap_n_background':  50,    # reduzido para economizar RAM (era 100)
        'shap_n_explain':    100,    # reduzido para economizar RAM (era 200)
        'force_retrain':    False,   # True → retreina mesmo se .joblib existir
    },
) as dag:

    dag.doc_md = """
## DAG de Treinamento ML — 3W Petrobras

Treina Random Forest, XGBoost e LightGBM nos dados Gold, otimiza
thresholds por classe e gera análise SHAP de interpretabilidade.

### Pré-requisito
A camada Gold deve estar populada. Rode `dag_gold_rebuild` antes se necessário.

### Parâmetros configuráveis (Trigger DAG → Config)
| Parâmetro | Descrição | Default |
|---|---|---|
| `modelos` | Modelos a treinar: `all`, `rf`, `xgb`, `lgb` | `all` |
| `n_estimators_rf` | Número de árvores RF | 150 |
| `n_estimators_xgb` | Rounds máximos XGBoost | 500 |
| `n_estimators_lgb` | Rounds máximos LightGBM | 1000 |
| `threshold_steps` | Passos do grid search de threshold | 50 |
| `shap_n_background` | Amostras background por classe (SHAP) | 100 |
| `shap_n_explain` | Amostras a explicar por classe (SHAP) | 200 |
| `force_retrain` | Retreinar mesmo com .joblib existente | False |

### Tempo estimado
- RF: ~5 min | XGB: ~50 min | LGB: ~10 min (paralelo)
- Threshold tuning: ~5 min
- SHAP analysis: ~3 min
- **Total: ~60–70 min**
    """

    # ── 1. Verificar Gold ──────────────────────────────────────────────────────
    def _verificar_gold(**ctx):
        import pandas as pd

        gold = Path(GOLD_DIR)
        splits = ['train.parquet', 'val.parquet', 'test.parquet']
        resultados = {}

        for split in splits:
            path = gold / split
            if not path.exists():
                raise FileNotFoundError(
                    f'Arquivo Gold ausente: {path}\n'
                    'Execute dag_gold_rebuild antes de rodar este DAG.'
                )
            df = pd.read_parquet(path, columns=['class'])
            resultados[split] = len(df)

        total = sum(resultados.values())
        if total == 0:
            raise ValueError('Gold está vazio. Execute dag_gold_rebuild.')

        print('Gold verificado:')
        for s, n in resultados.items():
            print(f'  {s}: {n:,} linhas')
        print(f'  TOTAL: {total:,} linhas')

        # Verifica features — lê só o schema (sem carregar dados na RAM)
        import pyarrow.parquet as pq
        schema = pq.read_schema(str(gold / 'train.parquet'))
        feat_cols = [c for c in schema.names if c.endswith('_norm')]
        print(f'  Features _norm: {len(feat_cols)}')
        if len(feat_cols) < 10:
            raise ValueError(f'Poucas features _norm ({len(feat_cols)}). Gold pode estar corrompido.')

        return resultados

    verificar_gold = PythonOperator(
        task_id         = 'verificar_gold',
        python_callable = _verificar_gold,
    )

    # ── 2. Treinamento dos modelos (paralelo) ──────────────────────────────────
    def _cmd_treinar(script_name, extra_args='', **ctx):
        """Executa um script de treino via subprocess e reloga o output."""
        script = str(Path(SCRIPTS_DIR) / script_name)
        cmd    = f'{PYTHON_BIN} "{script}" {extra_args}'.strip()
        print(f'Executando: {cmd}')

        env = os.environ.copy()
        env['PROJECT_DIR'] = PROJECT_DIR

        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, env=env,
        )
        for line in proc.stdout:
            print(line, end='')
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f'{script_name} falhou com código {proc.returncode}')

    # Sequencial para economizar RAM no t3.large (8GB):
    # RF → XGB → LGB, cada um libera memória antes do próximo iniciar.
    treinar_rf = PythonOperator(
        task_id           = 'treinar_rf',
        python_callable   = _cmd_treinar,
        op_kwargs         = {
            'script_name': 'train_rf_baseline.py',
            'extra_args':  '--n-estimators {{ params.n_estimators_rf }} --n-jobs 1',
        },
        execution_timeout = timedelta(hours=1),
    )

    treinar_xgb = PythonOperator(
        task_id           = 'treinar_xgb',
        python_callable   = _cmd_treinar,
        op_kwargs         = {
            'script_name': 'train_xgb_baseline.py',
            'extra_args':  '--n-estimators {{ params.n_estimators_xgb }}',
        },
        execution_timeout = timedelta(hours=2),
    )

    treinar_lgb = PythonOperator(
        task_id           = 'treinar_lgb',
        python_callable   = _cmd_treinar,
        op_kwargs         = {
            'script_name': 'train_lgb_baseline.py',
            'extra_args':  '--n-estimators {{ params.n_estimators_lgb }} --n-jobs 1',
        },
        execution_timeout = timedelta(hours=1),
    )

    # ── 3. Threshold tuning (aguarda os 3 modelos) ────────────────────────────
    threshold_tuning = PythonOperator(
        task_id         = 'threshold_tuning',
        python_callable = _cmd_treinar,
        op_kwargs       = {
            'script_name': 'tune_thresholds.py',
            'extra_args':  '--steps {{ params.threshold_steps }}',
        },
    )

    # ── 4. SHAP analysis (aguarda LGB + threshold tuning) ─────────────────────
    shap_analysis = PythonOperator(
        task_id         = 'shap_analysis',
        python_callable = _cmd_treinar,
        op_kwargs       = {
            'script_name': 'shap_analysis.py',
            'extra_args': (
                '--model lgb '
                '--n-background {{ params.shap_n_background }} '
                '--n-explain {{ params.shap_n_explain }}'
            ),
        },
        execution_timeout = timedelta(minutes=30),
    )

    # ── 5. Relatório final ─────────────────────────────────────────────────────
    def _relatorio_ml(**ctx):
        """Consolida métricas JSON de todos os modelos e imprime resumo."""
        models_path = Path(MODELS_DIR)
        report_lines = [
            '=' * 62,
            '  RELATÓRIO FINAL — ML TRAINING DAG',
            f'  Executado em: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            '=' * 62,
        ]

        # Métricas baseline
        model_files = {
            'RF':  'rf_baseline_metrics.json',
            'XGB': 'xgb_baseline_metrics.json',
            'LGB': 'lgb_baseline_metrics.json',
        }
        report_lines.append('\n  BASELINES:')
        report_lines.append(f'  {"Modelo":6s}  {"Acc Test":>9}  {"F1w Test":>9}  {"F1mac Test":>10}')
        for mname, fname in model_files.items():
            fpath = models_path / fname
            if fpath.exists():
                with open(fpath) as f:
                    m = json.load(f)
                test = m.get('test', m)
                acc  = test.get('accuracy', test.get('acc',  0))
                f1w  = test.get('f1_weighted', test.get('f1w', 0))
                f1m  = test.get('f1_macro',    test.get('f1mac', 0))
                report_lines.append(
                    f'  {mname:6s}  {acc:9.4f}  {f1w:9.4f}  {f1m:10.4f}')
            else:
                report_lines.append(f'  {mname:6s}  (métricas não encontradas)')

        # Threshold tuning
        tuning_path = models_path / 'threshold_tuning_results.json'
        if tuning_path.exists():
            with open(tuning_path) as f:
                tuning = json.load(f)
            report_lines.append('\n  THRESHOLD TUNING (Δ F1 Macro):')
            for mname, res in tuning.items():
                delta = res.get('test', {}).get('delta_f1mac', 0)
                tuned = res.get('test', {}).get('tuned', {}).get('f1mac', 0)
                report_lines.append(
                    f'  {mname.upper():6s}  F1mac tuned={tuned:.4f}  Δ={delta:+.4f}')

        # SHAP
        shap_path = models_path / 'shap_results.json'
        if shap_path.exists():
            with open(shap_path) as f:
                shap_data = json.load(f)
            report_lines.append('\n  SHAP — Top sensores globais:')
            for sensor, pct in list(shap_data.get('sensor_importance_global', {}).items())[:3]:
                report_lines.append(f'    {sensor}: {pct:.1f}%')

        report_lines.append('\n' + '=' * 62)
        report_text = '\n'.join(report_lines)

        # Salvar
        out = models_path / 'ml_training_dag_report.txt'
        out.write_text(report_text, encoding='utf-8')

        print(report_text)
        print(f'\n  Relatório salvo: {out}')

    relatorio_ml = PythonOperator(
        task_id         = 'relatorio_ml',
        python_callable = _relatorio_ml,
        trigger_rule    = TriggerRule.ALL_DONE,   # roda mesmo se SHAP falhar
    )

    # ── Dependências ───────────────────────────────────────────────────────────
    #
    #   verificar_gold → treinar_rf → treinar_xgb → treinar_lgb
    #                                                    ↓
    #                                           threshold_tuning
    #                                                    ↓
    #                                            shap_analysis
    #                                                    ↓
    #                                            relatorio_ml
    #
    # Sequencial: cada modelo libera RAM antes do próximo iniciar (t3.large 8GB).
    #
    verificar_gold >> treinar_rf >> treinar_xgb >> treinar_lgb
    treinar_lgb >> threshold_tuning >> shap_analysis >> relatorio_ml
