"""
Script: etl_bronze_silver_gold.py
Descricao: Pipeline ETL local com tres camadas de maturidade de dados.

  RAW  -> BRONZE : tipagem, schema padrao, metadados de origem
  BRONZE -> SILVER: limpeza de nulos, remocao de duplicatas, validacao de ranges
  SILVER -> GOLD  : rolling features, lag, normalizacao Z-score, divisao treino/val/teste

Uso (dentro do container Docker ou com pandas instalado):
  python scripts/etl_bronze_silver_gold.py [--step all|bronze|silver|gold] [--verbose]

Dependencias:
  pip install pandas pyarrow numpy scikit-learn
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.spark_config import (
    RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
    setup_directories, logger,
)

# ---------------------------------------------------------------------------
# Constantes do dataset 3W
# ---------------------------------------------------------------------------

SENSOR_COLUMNS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP",
    "T-JUS-CKP", "P-JUS-CKGL", "T-JUS-CKGL", "QGL",
]

TARGET_COLUMN = "class"
TIMESTAMP_COLUMN = "timestamp"

EVENT_LABELS = {
    0: "Normal",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Undefined",
}

# Ranges fisicos validos dos sensores (baseado na documentacao do 3W)
# Valores fora desses limites sao marcados como nulos na camada Silver
SENSOR_PHYSICAL_RANGES = {
    "P-PDG":        (0.0,   1000.0),
    "P-TPT":        (0.0,   1000.0),
    "T-TPT":        (-10.0, 200.0),
    "P-MON-CKP":    (0.0,   1000.0),
    "T-JUS-CKP":    (-10.0, 200.0),
    "P-JUS-CKGL":   (0.0,   1000.0),
    "T-JUS-CKGL":   (-10.0, 200.0),
    "QGL":          (0.0,   5000.0),
}

ROLLING_WINDOWS = [5, 10, 30]
LAG_STEPS = [1, 3, 5]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ---------------------------------------------------------------------------
# Utilitarios comuns
# ---------------------------------------------------------------------------

def load_parquets_from_dir(directory: Path, exclude_names=None) -> pd.DataFrame:
    """Carrega todos os .parquet de um diretorio em um unico DataFrame."""
    exclude_names = exclude_names or []
    files = [
        f for f in directory.glob("*.parquet")
        if f.name not in exclude_names
    ]
    if not files:
        logger.warning("Nenhum arquivo Parquet encontrado em: %s", directory)
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as exc:
            logger.error("Falha ao ler %s: %s", f.name, exc)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Carregados %d arquivos | %d linhas x %d colunas de %s",
                len(files), *df.shape, directory)
    return df


def save_parquet(df: pd.DataFrame, path: Path, name: str) -> None:
    """Salva DataFrame como Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Salvo: %s | %d linhas | %.2f MB", name, len(df), size_mb)


def print_class_distribution(df: pd.DataFrame, label: str) -> None:
    """Exibe distribuicao de classes no log."""
    if TARGET_COLUMN not in df.columns and "classe_evento" not in df.columns:
        return
    col = TARGET_COLUMN if TARGET_COLUMN in df.columns else "classe_evento"
    dist = df[col].value_counts().sort_index()
    logger.info("Distribuicao de classes em [%s]:", label)
    for cls, cnt in dist.items():
        nome = EVENT_LABELS.get(int(cls), "?")
        pct = cnt / len(df) * 100
        logger.info("  Classe %d (%s): %d linhas (%.1f%%)", cls, nome, cnt, pct)


# ---------------------------------------------------------------------------
# CAMADA BRONZE
# Objetivo: schema padrao, tipagem correta, metadados de origem
# Fonte: data/processed/ (arquivos ja com ID_Poco e classe_evento)
# ---------------------------------------------------------------------------

def run_bronze() -> pd.DataFrame:
    """
    Leitura da camada Processed e criacao da Bronze.

    Transformacoes:
    - Garante que timestamp e do tipo datetime
    - Converte sensores para float64
    - Garante que class/classe_evento e int8
    - Remove colunas redundantes
    - Adiciona coluna de origem e data de processamento
    """
    logger.info("=" * 60)
    logger.info("BRONZE - iniciando")
    logger.info("=" * 60)

    from config.spark_config import PROCESSED_DIR
    df = load_parquets_from_dir(
        PROCESSED_DIR,
        exclude_names=["all_classes_combined.parquet"]
    )

    if df.empty:
        logger.error("Sem dados em processed/ para criar Bronze.")
        return df

    # Normaliza coluna de timestamp
    for ts_col in ["timestamp", "index", "Timestamp"]:
        if ts_col in df.columns:
            df = df.rename(columns={ts_col: TIMESTAMP_COLUMN})
            break
    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    # Garante colunas de sensor existem e sao float
    for col in SENSOR_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            df[col] = np.nan

    # Normaliza coluna de classe
    if "classe_evento" in df.columns:
        df[TARGET_COLUMN] = df["classe_evento"].fillna(-1).astype("int8")
    elif TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(-1).astype("int8")

    # Remove classes invalidas (-1)
    before = len(df)
    df = df[df[TARGET_COLUMN] >= 0]
    removed = before - len(df)
    if removed > 0:
        logger.info("Bronze: removidos %d registros sem classe valida.", removed)

    # Adiciona label textual se nao existir
    if "nome_evento" not in df.columns:
        df["nome_evento"] = df[TARGET_COLUMN].map(EVENT_LABELS).fillna("Desconhecido")

    # Seleciona apenas colunas necessarias
    keep_cols = [TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN, "nome_evento"] + SENSOR_COLUMNS
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Metadados
    df["bronze_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Bronze: %d linhas x %d colunas", *df.shape)
    print_class_distribution(df, "Bronze")

    out = BRONZE_DIR / "bronze_3w_all.parquet"
    save_parquet(df, out, "bronze_3w_all.parquet")
    return df


# ---------------------------------------------------------------------------
# CAMADA SILVER
# Objetivo: dados limpos e confiaveis para analise
# Fonte: data/bronze/
# ---------------------------------------------------------------------------

def run_silver(df_bronze: pd.DataFrame = None) -> pd.DataFrame:
    """
    Limpeza e validacao dos dados da camada Bronze.

    Transformacoes:
    - Forward fill + backward fill por poco (por classe de evento)
    - Invalida leituras fora dos ranges fisicos dos sensores
    - Remove duplicatas de timestamp por poco
    - Remove registros onde todos os sensores sao nulos
    """
    logger.info("=" * 60)
    logger.info("SILVER - iniciando")
    logger.info("=" * 60)

    if df_bronze is None or df_bronze.empty:
        df_bronze = load_parquets_from_dir(BRONZE_DIR)

    if df_bronze.empty:
        logger.error("Sem dados na camada Bronze.")
        return pd.DataFrame()

    df = df_bronze.copy()

    # 1. Invalida leituras fora dos ranges fisicos
    for sensor, (vmin, vmax) in SENSOR_PHYSICAL_RANGES.items():
        if sensor not in df.columns:
            continue
        mask_invalid = (df[sensor] < vmin) | (df[sensor] > vmax)
        invalid_count = mask_invalid.sum()
        if invalid_count > 0:
            logger.info("Silver: %d leituras invalidas em %s -> substituidas por NaN",
                        invalid_count, sensor)
            df.loc[mask_invalid, sensor] = np.nan

    # 2. Forward fill + backward fill por poco e classe
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]
    sort_cols = [TIMESTAMP_COLUMN] if TIMESTAMP_COLUMN in df.columns else []

    if partition_cols and sort_cols:
        df = df.sort_values(partition_cols + sort_cols)
        for sensor in SENSOR_COLUMNS:
            if sensor not in df.columns:
                continue
            df[sensor] = (
                df.groupby(partition_cols)[sensor]
                .transform(lambda s: s.ffill().bfill())
            )

    # 3. Remove duplicatas de timestamp por poco
    if TIMESTAMP_COLUMN in df.columns and "ID_Poco" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["ID_Poco", TIMESTAMP_COLUMN], keep="first")
        dup_removed = before - len(df)
        if dup_removed > 0:
            logger.info("Silver: %d duplicatas de timestamp removidas.", dup_removed)

    # 4. Remove linhas onde TODOS os sensores ainda sao nulos
    sensor_cols_present = [c for c in SENSOR_COLUMNS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=sensor_cols_present, how="all")
    all_null_removed = before - len(df)
    if all_null_removed > 0:
        logger.info("Silver: %d linhas com todos os sensores nulos removidas.",
                    all_null_removed)

    # 5. Preenche nulos residuais com mediana global por classe (fallback)
    for sensor in sensor_cols_present:
        if df[sensor].isna().sum() > 0:
            if TARGET_COLUMN in df.columns:
                df[sensor] = df.groupby(TARGET_COLUMN)[sensor].transform(
                    lambda s: s.fillna(s.median())
                )
            df[sensor] = df[sensor].fillna(df[sensor].median())

    # Metadados
    df["silver_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Silver: %d linhas x %d colunas", *df.shape)
    print_class_distribution(df, "Silver")

    out = SILVER_DIR / "silver_3w_all.parquet"
    save_parquet(df, out, "silver_3w_all.parquet")
    return df


# ---------------------------------------------------------------------------
# CAMADA GOLD
# Objetivo: dataset ML-ready com features engineered e normalizado
# Fonte: data/silver/
# ---------------------------------------------------------------------------

def _safe_col(name: str) -> str:
    """Converte nome de sensor para nome de coluna valido (sem hifens)."""
    return name.replace("-", "_")


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, std, min, max por janela e por poco."""
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]

    for sensor in SENSOR_COLUMNS:
        if sensor not in df.columns:
            continue
        safe = _safe_col(sensor)
        for w in ROLLING_WINDOWS:
            grp = df.groupby(partition_cols)[sensor] if partition_cols else df[sensor]
            df[f"{safe}_roll_mean_{w}"] = grp.transform(
                lambda s, _w=w: s.rolling(_w, min_periods=1).mean()
            )
            df[f"{safe}_roll_std_{w}"] = grp.transform(
                lambda s, _w=w: s.rolling(_w, min_periods=1).std().fillna(0)
            )
            df[f"{safe}_roll_min_{w}"] = grp.transform(
                lambda s, _w=w: s.rolling(_w, min_periods=1).min()
            )
            df[f"{safe}_roll_max_{w}"] = grp.transform(
                lambda s, _w=w: s.rolling(_w, min_periods=1).max()
            )
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variaveis de lag temporal por poco."""
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]

    for sensor in SENSOR_COLUMNS:
        if sensor not in df.columns:
            continue
        safe = _safe_col(sensor)
        for lag in LAG_STEPS:
            if partition_cols:
                df[f"{safe}_lag_{lag}"] = df.groupby(partition_cols)[sensor].shift(lag)
            else:
                df[f"{safe}_lag_{lag}"] = df[sensor].shift(lag)
    return df


def create_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa de variacao (delta) entre instantes consecutivos por poco."""
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]

    for sensor in SENSOR_COLUMNS:
        if sensor not in df.columns:
            continue
        safe = _safe_col(sensor)
        if partition_cols:
            df[f"{safe}_delta"] = df.groupby(partition_cols)[sensor].diff()
        else:
            df[f"{safe}_delta"] = df[sensor].diff()
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list, train_mask: pd.Series) -> tuple:
    """
    Normalizacao Z-score calculada APENAS no conjunto de treino.
    Retorna (df_normalizado, dict_stats) para uso posterior na inferencia.
    """
    stats = {}
    df_norm = df.copy()

    for col in feature_cols:
        if col not in df_norm.columns:
            continue
        mean_val = df_norm.loc[train_mask, col].mean()
        std_val = df_norm.loc[train_mask, col].std()
        if pd.isna(std_val) or std_val == 0:
            std_val = 1.0
        stats[col] = {"mean": float(mean_val), "std": float(std_val)}
        df_norm[f"{col}_norm"] = (df_norm[col] - mean_val) / std_val

    return df_norm, stats


def run_gold(df_silver: pd.DataFrame = None) -> pd.DataFrame:
    """
    Engenharia de features e preparacao do dataset ML-ready.

    Transformacoes:
    - Rolling statistics (mean, std, min, max) janelas 5, 10, 30
    - Variaveis de lag (1, 3, 5 passos)
    - Taxa de variacao (delta) por sensor
    - Normalizacao Z-score (stats calculadas so no treino)
    - Divisao temporal: treino 70% / validacao 15% / teste 15%
    - Salva train.parquet, val.parquet, test.parquet em data/gold/
    """
    logger.info("=" * 60)
    logger.info("GOLD - iniciando")
    logger.info("=" * 60)

    if df_silver is None or df_silver.empty:
        df_silver = load_parquets_from_dir(SILVER_DIR)

    if df_silver.empty:
        logger.error("Sem dados na camada Silver.")
        return pd.DataFrame()

    df = df_silver.copy()

    # Ordena por poco e timestamp para garantir sequencia correta
    sort_cols = [c for c in ["ID_Poco", TIMESTAMP_COLUMN] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    logger.info("Gold: criando rolling features...")
    df = create_rolling_features(df)

    logger.info("Gold: criando lag features...")
    df = create_lag_features(df)

    logger.info("Gold: criando delta features...")
    df = create_delta_features(df)

    # Remove linhas com nulos gerados pelas janelas iniciais
    meta_cols = {TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN,
                 "nome_evento", "bronze_at", "silver_at"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    before = len(df)
    df = df.dropna(subset=feature_cols, how="any")
    logger.info("Gold: %d linhas removidas por nulos de janela inicial.", before - len(df))

    # Divisao temporal (sem embaralhamento para preservar ordem cronologica)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_mask = df.index < train_end

    logger.info("Gold: normalizando features (stats do treino)...")
    df, scaler_stats = normalize_features(df, feature_cols, train_mask)

    # Salva as stats de normalizacao para uso na inferencia futura
    stats_path = GOLD_DIR / "scaler_stats.json"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(scaler_stats, fh, indent=2)
    logger.info("Scaler stats salvas em: %s", stats_path)

    # Seleciona apenas colunas normalizadas + metadados
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    final_cols = [TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN, "nome_evento"] + norm_cols
    final_cols = [c for c in final_cols if c in df.columns]
    df_final = df[final_cols].copy()
    df_final["gold_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Divisao e escrita
    df_train = df_final.iloc[:train_end]
    df_val   = df_final.iloc[train_end:val_end]
    df_test  = df_final.iloc[val_end:]

    save_parquet(df_train, GOLD_DIR / "train.parquet", "train.parquet")
    save_parquet(df_val,   GOLD_DIR / "val.parquet",   "val.parquet")
    save_parquet(df_test,  GOLD_DIR / "test.parquet",  "test.parquet")

    logger.info("Gold: divisao temporal:")
    logger.info("  Treino:    %d linhas (%.0f%%)", len(df_train), TRAIN_RATIO * 100)
    logger.info("  Validacao: %d linhas (%.0f%%)", len(df_val),   VAL_RATIO * 100)
    logger.info("  Teste:     %d linhas (%.0f%%)", len(df_test),  TEST_RATIO * 100)
    logger.info("  Features:  %d colunas", len(norm_cols))

    print_class_distribution(df_train, "Gold - Treino")
    print_class_distribution(df_val,   "Gold - Validacao")
    print_class_distribution(df_test,  "Gold - Teste")

    return df_final


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(step: str = "all", verbose: bool = False) -> None:
    """
    Executa o pipeline ETL completo ou uma etapa especifica.

    Args:
        step: "all" | "bronze" | "silver" | "gold"
        verbose: se True, exibe mais detalhes no log
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_directories()

    start = datetime.now(timezone.utc)
    logger.info("Pipeline ETL iniciado | etapa: %s | %s", step, start.strftime("%Y-%m-%d %H:%M:%S"))

    df_bronze = pd.DataFrame()
    df_silver = pd.DataFrame()

    if step in ("all", "bronze"):
        df_bronze = run_bronze()

    if step in ("all", "silver"):
        df_silver = run_silver(df_bronze if not df_bronze.empty else None)

    if step in ("all", "gold"):
        run_gold(df_silver if not df_silver.empty else None)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info("Pipeline ETL concluido em %.1f segundos.", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Bronze/Silver/Gold - Dataset 3W")
    parser.add_argument(
        "--step",
        choices=["all", "bronze", "silver", "gold"],
        default="all",
        help="Etapa a executar (default: all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Nivel de log DEBUG",
    )
    args = parser.parse_args()
    run_pipeline(step=args.step, verbose=args.verbose)
