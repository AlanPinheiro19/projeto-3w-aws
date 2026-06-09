# test_classe1.py
import requests
from pathlib import Path

# Testar classe 1
api_url = "https://api.github.com/repos/petrobras/3W/contents/dataset/1"
response = requests.get(api_url)
files = response.json()

well_files = [f for f in files if f['name'].startswith('WELL')]
print(f"Classe 1 - Arquivos WELL encontrados: {len(well_files)}")

for f in well_files[:5]:
    print(f"  {f['name']}")