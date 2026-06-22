"""
Script de Ingestao - Dataset 3W Petrobras (Classe 0)
Versao para ambiente academico com permissoes restritas (voclabs)
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from io import BytesIO

import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError


# ============================================================================
# CONFIGURACAO DE LOGGING
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# ============================================================================
# CONSTANTES
# ============================================================================

DEFAULT_BUCKET = "tcc-ptbr-3w-datalake-2026"
DEFAULT_CLASSE = "0"
GITHUB_API_BASE = "https://api.github.com/repos/petrobras/3W/contents/dataset"
REQUEST_TIMEOUT = 30
CHUNK_SIZE = 8192


# ============================================================================
# CLASSES E EXCEPTIONS
# ============================================================================

class IngestionError(Exception):
    pass


class DataIngestor:
    def __init__(self, bucket_name: str, classe: str, verbose: bool = False):
        self.bucket_name = bucket_name
        self.classe = classe
        self.logger = logging.getLogger(__name__)
        
        setup_logging(verbose)
        self._validate_classe()
        self._initialize_s3_client()
    
    def _validate_classe(self) -> None:
        try:
            classe_int = int(self.classe)
            if classe_int < 0 or classe_int > 9:
                raise IngestionError(f"Classe invalida: {self.classe}. Deve estar entre 0 e 9.")
        except ValueError:
            raise IngestionError(f"Classe invalida: {self.classe}. Deve ser um numero.")
    
    def _initialize_s3_client(self) -> None:
        """Inicializa o cliente S3 sem testar ListBuckets (permissoes restritas)."""
        try:
            self.s3_client = boto3.client('s3')
            # APENAS testar se o cliente foi criado - sem listar buckets
            self.logger.info("Cliente S3 inicializado com sucesso")
            self.logger.info(f"Bucket configurado: {self.bucket_name}")
        except NoCredentialsError:
            raise IngestionError("Credenciais AWS nao encontradas. Execute 'aws configure'.")
        except Exception as e:
            raise IngestionError(f"Erro ao inicializar cliente S3: {e}")
    
    def get_file_list(self) -> List[Dict]:
        api_url = f"{GITHUB_API_BASE}/{self.classe}"
        self.logger.info(f"Buscando arquivos da classe {self.classe}: {api_url}")
        
        try:
            response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            files = response.json()
        except requests.exceptions.RequestException as e:
            raise IngestionError(f"Erro ao acessar GitHub API: {e}")
        
        parquet_files = [
            f for f in files 
            if isinstance(f, dict) and f.get('name', '').endswith('.parquet')
        ]
        
        if not parquet_files:
            raise IngestionError(f"Nenhum arquivo Parquet encontrado para a classe {self.classe}")
        
        self.logger.info(f"Encontrados {len(parquet_files)} arquivos Parquet")
        return parquet_files
    
    def download_and_upload(self, file_info: Dict) -> Tuple[bool, int]:
        file_name = file_info.get('name', 'unknown')
        file_url = file_info.get('download_url')
        s3_key = f"raw/classe{self.classe}/{file_name}"
        
        self.logger.debug(f"Processando: {file_name}")
        
        if not file_url:
            self.logger.error(f"URL nao disponivel para: {file_name}")
            return False, 0
        
        try:
            response = requests.get(file_url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            content = BytesIO()
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    content.write(chunk)
            
            file_size = content.tell()
            content.seek(0)
            
            self.s3_client.upload_fileobj(content, self.bucket_name, s3_key)
            
            self.logger.debug(f"Upload concluido: {file_name} ({file_size / 1024:.1f} KB)")
            return True, file_size
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout ao baixar: {file_name}")
            return False, 0
        except Exception as e:
            self.logger.error(f"Erro ao processar {file_name}: {str(e)[:100]}")
            return False, 0
    
    def verify_s3_upload(self) -> Dict:
        """
        Verifica os arquivos no S3 sem usar ListBuckets.
        Usa list_objects_v2 diretamente com prefixo conhecido.
        """
        prefix = f"raw/classe{self.classe}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=100
            )
            
            if 'Contents' not in response:
                self.logger.warning(f"Nenhum arquivo encontrado em s3://{self.bucket_name}/{prefix}")
                return {'count': 0, 'total_size': 0, 'files': []}
            
            files = []
            total_size = 0
            for obj in response['Contents']:
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': str(obj['LastModified'])
                })
                total_size += obj['Size']
            
            return {
                'count': len(files),
                'total_size': total_size,
                'files': files
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'AccessDenied':
                self.logger.warning("Sem permissao para listar objetos. Verifique manualmente no console.")
            else:
                self.logger.error(f"Erro ao verificar S3: {e}")
            return {'count': 0, 'total_size': 0, 'files': []}
    
    def run(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO INGESTAO DO DATASET 3W (MODO RESTRITO)")
        self.logger.info(f"Classe: {self.classe} (Operacao Normal)")
        self.logger.info(f"Bucket destino: s3://{self.bucket_name}/")
        self.logger.info(f"Data/Hora: {datetime.now().isoformat()}")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            files = self.get_file_list()
        except IngestionError as e:
            self.logger.error(f"Falha ao obter lista de arquivos: {e}")
            return {'success': False, 'error': str(e)}
        
        success_count = 0
        failed_count = 0
        total_size = 0
        failed_files = []
        
        for idx, file_info in enumerate(files, 1):
            self.logger.info(f"[{idx}/{len(files)}] Processando: {file_info['name']}")
            
            success, size = self.download_and_upload(file_info)
            
            if success:
                success_count += 1
                total_size += size
            else:
                failed_count += 1
                failed_files.append(file_info['name'])
        
        verification = self.verify_s3_upload()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = {
            'success': failed_count == 0,
            'total_files': len(files),
            'success_count': success_count,
            'failed_count': failed_count,
            'failed_files': failed_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'verification': verification,
            'elapsed_seconds': elapsed
        }
        
        self.logger.info("=" * 60)
        self.logger.info("RESUMO DA INGESTAO")
        self.logger.info(f"Total de arquivos: {result['total_files']}")
        self.logger.info(f"Uploads bem-sucedidos: {result['success_count']}")
        self.logger.info(f"Uploads com falha: {result['failed_count']}")
        self.logger.info(f"Tamanho total: {result['total_size_mb']:.2f} MB")
        self.logger.info(f"Tempo de execucao: {result['elapsed_seconds']:.2f} segundos")
        
        if failed_files:
            self.logger.warning(f"Arquivos com falha: {', '.join(failed_files[:5])}")
        
        if verification['count'] > 0:
            self.logger.info(f"S3: {verification['count']} arquivos, "
                           f"{verification['total_size'] / (1024*1024):.2f} MB")
        
        self.logger.info("=" * 60)
        self.logger.info("INGESTAO CONCLUIDA")
        
        return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Ingestao do Dataset 3W da Petrobras (modo ambiente restrito)'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default=DEFAULT_BUCKET,
        help=f'Nome do bucket S3 (default: {DEFAULT_BUCKET})'
    )
    parser.add_argument(
        '--classe',
        type=str,
        default=DEFAULT_CLASSE,
        help='Classe de evento (0-9, default: 0)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Exibe mensagens de debug'
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    
    try:
        ingestor = DataIngestor(
            bucket_name=args.bucket,
            classe=args.classe,
            verbose=args.verbose
        )
        
        result = ingestor.run()
        
        if result['success']:
            return 0
        else:
            return 1
            
    except IngestionError as e:
        logging.error(f"Erro fatal: {e}")
        return 1
    except KeyboardInterrupt:
        logging.warning("Execucao interrompida pelo usuario")
        return 130
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())