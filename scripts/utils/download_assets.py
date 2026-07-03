"""
download_assets.py — Baixa logos PECE/USP para a pasta assets/ na raiz do projeto.
Execute uma vez antes de rodar o dashboard:
    python scripts/utils/download_assets.py
"""
import urllib.request
import pathlib

# Localiza a raiz do projeto (dois niveis acima de scripts/utils/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ASSETS = PROJECT_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

URLS = {
    "logo-pece-white.png":  "https://pecepoli.com.br/wp-content/themes/pece/dist/img/logo-pece-white.png",
    "logo-pece-blue.png":   "https://pecepoli.com.br/wp-content/themes/pece/dist/img/logo-blue.png",
    "logo-usp.png":         "https://pecepoli.com.br/wp-content/themes/pece/dist/img/logo-usp.png",
    "logo-poli.png":        "https://pecepoli.com.br/wp-content/themes/pece/dist/img/logo-escola-politecnica.png",
    "minerva.png":          "https://pecepoli.com.br/wp-content/uploads/2024/05/cropped-minerva-270x270.png",
}

# Cabecalhos para contornar bloqueio de hotlink externo
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://pecepoli.com.br/",
}

for filename, url in URLS.items():
    dest = ASSETS / filename
    if dest.exists():
        print(f"  OK {filename} (ja existe)")
        continue
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r, open(dest, "wb") as f:
            f.write(r.read())
        size = dest.stat().st_size
        print(f"  OK {filename} ({size//1024} KB)")
    except Exception as e:
        print(f"  ERRO {filename} — {e}")

print("\nPronto! Execute na raiz do projeto: streamlit run app.py")
