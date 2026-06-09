# ingestao_classe0_local.py
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO

# Criar pasta local
LOCAL_DIR = Path("C:/projeto-3w-aws/data/classe0")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Obter lista de arquivos
api_url = "https://api.github.com/repos/petrobras/3W/contents/dataset/0"
files = requests.get(api_url).json()
parquet_files = [f for f in files if f['name'].endswith('.parquet')]

print(f"Encontrados {len(parquet_files)} arquivos")

# Baixar cada arquivo
for idx, file_info in enumerate(parquet_files[:10], 1):  # Apenas 10 para teste
    url = file_info['download_url']
    name = file_info['name']
    local_path = LOCAL_DIR / name
    
    print(f"[{idx}/10] Baixando: {name}")
    response = requests.get(url)
    
    with open(local_path, 'wb') as f:
        f.write(response.content)
    
    print(f"  Salvo: {local_path} ({len(response.content) / 1024:.1f} KB)")

print(f"\nDados salvos em: {LOCAL_DIR}")