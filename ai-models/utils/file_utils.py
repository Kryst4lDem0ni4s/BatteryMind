"""
BatteryMind - File Handlers
Comprehensive file I/O, data export, and storage utilities for the BatteryMind
autonomous battery management system with support for multiple formats and cloud storage.

Features:
- Multi-format file I/O (CSV, JSON, Parquet, HDF5, Excel)
- Cloud storage integration (AWS S3, Google Cloud Storage)
- Data compression and encryption
- Batch processing and streaming I/O
- File validation and integrity checking
- Backup and versioning utilities
- Data archiving and lifecycle management

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import os
import shutil
import gzip
import zipfile
import tarfile
import json
import csv
import pickle
import hashlib
import threading
from typing import Dict, List, Optional, Any, Union, Iterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO, BytesIO
import tempfile
import mimetypes

# Data processing imports
import pandas as pd
import numpy as np

# Cloud storage imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    from google.cloud.exceptions import NotFound, Forbidden
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Compression and encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Additional format support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FileMetadata:
    """File metadata information."""
    
    file_path: str
    file_size: int
    created_at: datetime
    modified_at: datetime
    file_type: str
    mime_type: str
    checksum: str
    compression: Optional[str] = None
    encryption: bool = False
    version: str = "1.0"
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'file_path': self.file_path,
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'file_type': self.file_type,
            'mime_type': self.mime_type,
            'checksum': self.checksum,
            'compression': self.compression,
            'encryption': self.encryption,
            'version': self.version,
            'tags': self.tags
        }

@dataclass
class ExportConfig:
    """Configuration for data export operations."""
    
    format: str = "csv"
    compression: Optional[str] = None
    encryption: bool = False
    batch_size: int = 10000
    include_metadata: bool = True
    validate_data: bool = True
    create_backup: bool = False
    overwrite_existing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'format': self.format,
            'compression': self.compression,
            'encryption': self.encryption,
            'batch_size': self.batch_size,
            'include_metadata': self.include_metadata,
            'validate_data': self.validate_data,
            'create_backup': self.create_backup,
            'overwrite_existing': self.overwrite_existing
        }

class FileHandler:
    """
    Base file handler with common file operations.
    """
    
    def __init__(self, 
                 base_path: str = "./data",
                 enable_compression: bool = True,
                 enable_encryption: bool = False,
                 backup_enabled: bool = True):
        
        self.base_path = Path(base_path)
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        self.backup_enabled = backup_enabled
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Encryption setup
        self.cipher_suite = None
        if enable_encryption and ENCRYPTION_AVAILABLE:
            self._setup_encryption()
        
        # File locks for concurrent access
        self.file_locks: Dict[str, threading.Lock] = {}
        
        logger.info(f"File Handler initialized with base path: {base_path}")
    
    def _setup_encryption(self):
        """Set up file encryption."""
        try:
            # Generate or load encryption key
            key_file = self.base_path / ".encryption_key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                # Make key file read-only
                os.chmod(key_file, 0o600)
            
            self.cipher_suite = Fernet(key)
            logger.info("File encryption enabled")
            
        except Exception as e:
            logger.error(f"Error setting up encryption: {e}")
            self.enable_encryption = False
    
    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """Get or create a file lock for concurrent access."""
        if file_path not in self.file_locks:
            self.file_locks[file_path] = threading.Lock()
        return self.file_locks[file_path]
    
    def calculate_checksum(self, file_path: Union[str, Path], algorithm: str = "md5") -> str:
        """Calculate file checksum."""
        try:
            hash_algo = getattr(hashlib, algorithm)()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_algo.update(chunk)
            
            return hash_algo.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> FileMetadata:
        """Get comprehensive file metadata."""
        try:
            file_path = Path(file_path)
            stat = file_path.stat()
            
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            
            metadata = FileMetadata(
                file_path=str(file_path),
                file_size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                file_type=file_path.suffix.lower(),
                mime_type=mime_type,
                checksum=self.calculate_checksum(file_path)
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            raise
    
    def create_backup(self, file_path: Union[str, Path], backup_dir: str = None) -> str:
        """Create a backup of the file."""
        try:
            file_path = Path(file_path)
            
            if backup_dir is None:
                backup_dir = self.base_path / "backups"
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped backup name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name
            
            # Copy file to backup location
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def compress_file(self, file_path: Union[str, Path], 
                     compression_type: str = "gzip",
                     remove_original: bool = False) -> str:
        """Compress a file."""
        try:
            file_path = Path(file_path)
            
            if compression_type == "gzip":
                compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
                
                with open(file_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            elif compression_type == "zip":
                compressed_path = file_path.with_suffix(".zip")
                
                with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(file_path, file_path.name)
            
            elif compression_type == "tar.gz":
                compressed_path = file_path.with_suffix(".tar.gz")
                
                with tarfile.open(compressed_path, "w:gz") as tar:
                    tar.add(file_path, arcname=file_path.name)
            
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
            
            if remove_original:
                file_path.unlink()
            
            logger.info(f"File compressed: {compressed_path}")
            return str(compressed_path)
            
        except Exception as e:
            logger.error(f"Error compressing file: {e}")
            raise
    
    def decompress_file(self, compressed_path: Union[str, Path],
                       extract_dir: str = None) -> str:
        """Decompress a file."""
        try:
            compressed_path = Path(compressed_path)
            
            if extract_dir is None:
                extract_dir = compressed_path.parent
            else:
                extract_dir = Path(extract_dir)
            
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            if compressed_path.suffix == ".gz":
                # Handle .gz files
                output_path = extract_dir / compressed_path.stem
                
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            elif compressed_path.suffix == ".zip":
                # Handle .zip files
                with zipfile.ZipFile(compressed_path, 'r') as zipf:
                    zipf.extractall(extract_dir)
                    output_path = extract_dir / zipf.namelist()[0]
            
            elif compressed_path.suffix == ".gz" and ".tar" in compressed_path.name:
                # Handle .tar.gz files
                with tarfile.open(compressed_path, "r:gz") as tar:
                    tar.extractall(extract_dir)
                    output_path = extract_dir / tar.getnames()[0]
            
            else:
                raise ValueError(f"Unsupported compression format: {compressed_path.suffix}")
            
            logger.info(f"File decompressed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error decompressing file: {e}")
            raise
    
    def encrypt_file(self, file_path: Union[str, Path]) -> str:
        """Encrypt a file."""
        try:
            if not self.cipher_suite:
                raise RuntimeError("Encryption not available")
            
            file_path = Path(file_path)
            encrypted_path = file_path.with_suffix(file_path.suffix + ".enc")
            
            with open(file_path, 'rb') as f_in:
                data = f_in.read()
            
            encrypted_data = self.cipher_suite.encrypt(data)
            
            with open(encrypted_path, 'wb') as f_out:
                f_out.write(encrypted_data)
            
            logger.info(f"File encrypted: {encrypted_path}")
            return str(encrypted_path)
            
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            raise
    
    def decrypt_file(self, encrypted_path: Union[str, Path]) -> str:
        """Decrypt a file."""
        try:
            if not self.cipher_suite:
                raise RuntimeError("Encryption not available")
            
            encrypted_path = Path(encrypted_path)
            
            if not encrypted_path.name.endswith('.enc'):
                raise ValueError("File does not appear to be encrypted")
            
            decrypted_path = encrypted_path.with_suffix('')
            
            with open(encrypted_path, 'rb') as f_in:
                encrypted_data = f_in.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            with open(decrypted_path, 'wb') as f_out:
                f_out.write(decrypted_data)
            
            logger.info(f"File decrypted: {decrypted_path}")
            return str(decrypted_path)
            
        except Exception as e:
            logger.error(f"Error decrypting file: {e}")
            raise
    
    def validate_file_integrity(self, file_path: Union[str, Path], 
                              expected_checksum: str = None) -> bool:
        """Validate file integrity using checksum."""
        try:
            if expected_checksum is None:
                return True
            
            actual_checksum = self.calculate_checksum(file_path)
            return actual_checksum == expected_checksum
            
        except Exception as e:
            logger.error(f"Error validating file integrity: {e}")
            return False

class DataExporter:
    """
    Advanced data exporter with support for multiple formats and optimizations.
    """
    
    def __init__(self, file_handler: FileHandler = None):
        self.file_handler = file_handler or FileHandler()
        
        # Export statistics
        self.export_stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'total_records_exported': 0,
            'total_size_exported': 0
        }
        
        logger.info("Data Exporter initialized")
    
    def export_battery_data(self, 
                          data: pd.DataFrame,
                          output_path: str,
                          config: ExportConfig = None) -> Dict[str, Any]:
        """
        Export battery data in specified format.
        
        Args:
            data: Battery data DataFrame
            output_path: Output file path
            config: Export configuration
            
        Returns:
            Export result dictionary
        """
        try:
            config = config or ExportConfig()
            output_path = Path(output_path)
            
            # Validate data if requested
            if config.validate_data:
                validation_result = self._validate_battery_data(data)
                if not validation_result['valid']:
                    raise ValueError(f"Data validation failed: {validation_result['errors']}")
            
            # Create backup if requested
            if config.create_backup and output_path.exists():
                backup_path = self.file_handler.create_backup(output_path)
                logger.info(f"Backup created: {backup_path}")
            
            # Check if file exists and overwrite policy
            if output_path.exists() and not config.overwrite_existing:
                raise FileExistsError(f"Output file exists and overwrite is disabled: {output_path}")
            
            # Export based on format
            export_result = self._export_by_format(data, output_path, config)
            
            # Apply compression if requested
            if config.compression:
                compressed_path = self.file_handler.compress_file(
                    output_path, 
                    config.compression,
                    remove_original=True
                )
                export_result['file_path'] = compressed_path
            
            # Apply encryption if requested
            if config.encryption:
                encrypted_path = self.file_handler.encrypt_file(export_result['file_path'])
                if compressed_path != export_result['file_path']:
                    os.remove(export_result['file_path'])
                export_result['file_path'] = encrypted_path
            
            # Update statistics
            self.export_stats['total_exports'] += 1
            self.export_stats['successful_exports'] += 1
            self.export_stats['total_records_exported'] += len(data)
            self.export_stats['total_size_exported'] += export_result.get('file_size', 0)
            
            logger.info(f"Battery data exported successfully: {export_result['file_path']}")
            
            return export_result
            
        except Exception as e:
            self.export_stats['total_exports'] += 1
            self.export_stats['failed_exports'] += 1
            logger.error(f"Error exporting battery data: {e}")
            raise
    
    def _export_by_format(self, 
                         data: pd.DataFrame, 
                         output_path: Path,
                         config: ExportConfig) -> Dict[str, Any]:
        """Export data based on specified format."""
        try:
            start_time = datetime.now()
            
            if config.format.lower() == "csv":
                data.to_csv(output_path, index=False)
            
            elif config.format.lower() == "json":
                data.to_json(output_path, orient='records', date_format='iso')
            
            elif config.format.lower() == "parquet" and PARQUET_AVAILABLE:
                data.to_parquet(output_path, index=False)
            
            elif config.format.lower() == "hdf5" and HDF5_AVAILABLE:
                data.to_hdf(output_path, key='battery_data', mode='w', index=False)
            
            elif config.format.lower() == "excel" and EXCEL_AVAILABLE:
                data.to_excel(output_path, index=False, sheet_name='BatteryData')
            
            elif config.format.lower() == "pickle":
                data.to_pickle(output_path)
            
            else:
                raise ValueError(f"Unsupported export format: {config.format}")
            
            # Get file metadata
            metadata = self.file_handler.get_file_metadata(output_path)
            
            export_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'file_path': str(output_path),
                'format': config.format,
                'records_exported': len(data),
                'file_size': metadata.file_size,
                'export_time_seconds': export_time,
                'checksum': metadata.checksum,
                'metadata': metadata.to_dict() if config.include_metadata else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting in format {config.format}: {e}")
            raise
    
    def _validate_battery_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate battery data before export."""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check required columns
            required_columns = ['battery_id', 'timestamp', 'soh', 'soc']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                validation_result['valid'] = False
            
            # Check data types
            if 'timestamp' in data.columns:
                try:
                    pd.to_datetime(data['timestamp'])
                except:
                    validation_result['errors'].append("Invalid timestamp format")
                    validation_result['valid'] = False
            
            # Check value ranges
            if 'soh' in data.columns:
                invalid_soh = ((data['soh'] < 0) | (data['soh'] > 100)).sum()
                if invalid_soh > 0:
                    validation_result['warnings'].append(f"Found {invalid_soh} invalid SoH values")
            
            if 'soc' in data.columns:
                invalid_soc = ((data['soc'] < 0) | (data['soc'] > 100)).sum()
                if invalid_soc > 0:
                    validation_result['warnings'].append(f"Found {invalid_soc} invalid SoC values")
            
            # Check for missing data
            missing_data = data.isnull().sum().sum()
            if missing_data > 0:
                validation_result['warnings'].append(f"Found {missing_data} missing values")
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def export_fleet_summary(self, 
                           fleet_data: Dict[str, pd.DataFrame],
                           output_dir: str,
                           config: ExportConfig = None) -> Dict[str, Any]:
        """Export comprehensive fleet data summary."""
        try:
            config = config or ExportConfig()
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            export_results = {}
            
            # Export each dataset
            for dataset_name, data in fleet_data.items():
                if data.empty:
                    continue
                
                output_file = output_dir / f"{dataset_name}.{config.format}"
                result = self.export_battery_data(data, output_file, config)
                export_results[dataset_name] = result
            
            # Create summary report
            summary_report = self._generate_fleet_summary_report(fleet_data, export_results)
            summary_path = output_dir / f"fleet_summary.{config.format}"
            
            if config.format == "json":
                with open(summary_path, 'w') as f:
                    json.dump(summary_report, f, indent=2, default=str)
            else:
                summary_df = pd.DataFrame([summary_report])
                self.export_battery_data(summary_df, summary_path, config)
            
            export_results['summary'] = {
                'file_path': str(summary_path),
                'summary_data': summary_report
            }
            
            logger.info(f"Fleet summary exported: {output_dir}")
            
            return export_results
            
        except Exception as e:
            logger.error(f"Error exporting fleet summary: {e}")
            raise
    
    def _generate_fleet_summary_report(self, 
                                     fleet_data: Dict[str, pd.DataFrame],
                                     export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fleet summary report."""
        try:
            summary = {
                'export_timestamp': datetime.now().isoformat(),
                'datasets_exported': len(export_results),
                'total_records': sum(len(df) for df in fleet_data.values()),
                'total_file_size': sum(result.get('file_size', 0) for result in export_results.values()),
                'dataset_details': {}
            }
            
            for dataset_name, data in fleet_data.items():
                if dataset_name in export_results:
                    summary['dataset_details'][dataset_name] = {
                        'record_count': len(data),
                        'file_size': export_results[dataset_name].get('file_size', 0),
                        'export_time': export_results[dataset_name].get('export_time_seconds', 0),
                        'columns': list(data.columns),
                        'data_types': data.dtypes.to_dict()
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating fleet summary report: {e}")
            return {}
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        return self.export_stats.copy()

class CloudStorageHandler:
    """
    Cloud storage handler supporting multiple providers.
    """
    
    def __init__(self, 
                 provider: str = "aws",
                 bucket_name: str = None,
                 region: str = "us-east-1"):
        
        self.provider = provider.lower()
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize cloud client
        self.client = None
        self._setup_cloud_client()
        
        logger.info(f"Cloud Storage Handler initialized for {provider}")
    
    def _setup_cloud_client(self):
        """Set up cloud storage client."""
        try:
            if self.provider == "aws" and AWS_AVAILABLE:
                self.client = boto3.client('s3', region_name=self.region)
                
                # Test connection
                self.client.head_bucket(Bucket=self.bucket_name)
                logger.info("AWS S3 client initialized successfully")
            
            elif self.provider == "gcp" and GCP_AVAILABLE:
                self.client = gcs.Client()
                
                # Test connection
                bucket = self.client.bucket(self.bucket_name)
                bucket.reload()
                logger.info("Google Cloud Storage client initialized successfully")
            
            else:
                logger.warning(f"Cloud provider {self.provider} not available or supported")
                
        except Exception as e:
            logger.error(f"Error setting up cloud storage client: {e}")
            self.client = None
    
    def upload_file(self, 
                   local_path: Union[str, Path],
                   remote_key: str,
                   metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """Upload file to cloud storage."""
        try:
            if not self.client:
                raise RuntimeError("Cloud storage client not available")
            
            local_path = Path(local_path)
            
            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            start_time = datetime.now()
            
            if self.provider == "aws":
                extra_args = {}
                if metadata:
                    extra_args['Metadata'] = metadata
                
                self.client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    remote_key,
                    ExtraArgs=extra_args
                )
            
            elif self.provider == "gcp":
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(remote_key)
                
                if metadata:
                    blob.metadata = metadata
                
                blob.upload_from_filename(str(local_path))
            
            upload_time = (datetime.now() - start_time).total_seconds()
            file_size = local_path.stat().st_size
            
            result = {
                'local_path': str(local_path),
                'remote_key': remote_key,
                'bucket': self.bucket_name,
                'file_size': file_size,
                'upload_time_seconds': upload_time,
                'provider': self.provider
            }
            
            logger.info(f"File uploaded to {self.provider}: {remote_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error uploading file to cloud storage: {e}")
            raise
    
    def download_file(self, 
                     remote_key: str,
                     local_path: Union[str, Path]) -> Dict[str, Any]:
        """Download file from cloud storage."""
        try:
            if not self.client:
                raise RuntimeError("Cloud storage client not available")
            
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            start_time = datetime.now()
            
            if self.provider == "aws":
                self.client.download_file(
                    self.bucket_name,
                    remote_key,
                    str(local_path)
                )
            
            elif self.provider == "gcp":
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(remote_key)
                blob.download_to_filename(str(local_path))
            
            download_time = (datetime.now() - start_time).total_seconds()
            file_size = local_path.stat().st_size if local_path.exists() else 0
            
            result = {
                'remote_key': remote_key,
                'local_path': str(local_path),
                'bucket': self.bucket_name,
                'file_size': file_size,
                'download_time_seconds': download_time,
                'provider': self.provider
            }
            
            logger.info(f"File downloaded from {self.provider}: {remote_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading file from cloud storage: {e}")
            raise
    
    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in cloud storage."""
        try:
            if not self.client:
                raise RuntimeError("Cloud storage client not available")
            
            files = []
            
            if self.provider == "aws":
                paginator = self.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            files.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'etag': obj['ETag'].strip('"')
                            })
            
            elif self.provider == "gcp":
                bucket = self.client.bucket(self.bucket_name)
                blobs = bucket.list_blobs(prefix=prefix)
                
                for blob in blobs:
                    files.append({
                        'key': blob.name,
                        'size': blob.size,
                        'last_modified': blob.time_created.isoformat() if blob.time_created else None,
                        'etag': blob.etag
                    })
            
            logger.info(f"Listed {len(files)} files from {self.provider}")
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files from cloud storage: {e}")
            return []
    
    def delete_file(self, remote_key: str) -> bool:
        """Delete file from cloud storage."""
        try:
            if not self.client:
                raise RuntimeError("Cloud storage client not available")
            
            if self.provider == "aws":
                self.client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            
            elif self.provider == "gcp":
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(remote_key)
                blob.delete()
            
            logger.info(f"File deleted from {self.provider}: {remote_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from cloud storage: {e}")
            return False

class BatchProcessor:
    """
    Batch file processor for handling large datasets.
    """
    
    def __init__(self, 
                 batch_size: int = 10000,
                 max_workers: int = 4):
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        logger.info(f"Batch Processor initialized with batch_size={batch_size}")
    
    def process_large_file(self, 
                          file_path: Union[str, Path],
                          processor_func: Callable,
                          output_path: Union[str, Path] = None) -> Dict[str, Any]:
        """Process large file in batches."""
        try:
            file_path = Path(file_path)
            
            # Determine file format and create reader
            if file_path.suffix.lower() == '.csv':
                reader = pd.read_csv(file_path, chunksize=self.batch_size)
            elif file_path.suffix.lower() == '.json':
                # For JSON, we need to read line by line (assuming JSONL format)
                reader = self._json_batch_reader(file_path)
            else:
                raise ValueError(f"Unsupported file format for batch processing: {file_path.suffix}")
            
            results = []
            batch_count = 0
            total_records = 0
            
            start_time = datetime.now()
            
            # Process batches
            for batch in reader:
                batch_result = processor_func(batch)
                results.append(batch_result)
                batch_count += 1
                total_records += len(batch)
                
                if batch_count % 10 == 0:
                    logger.info(f"Processed {batch_count} batches, {total_records} records")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Combine results if output path is provided
            if output_path and results:
                self._combine_batch_results(results, output_path)
            
            summary = {
                'file_path': str(file_path),
                'batches_processed': batch_count,
                'total_records': total_records,
                'processing_time_seconds': processing_time,
                'output_path': str(output_path) if output_path else None,
                'results': results if not output_path else None
            }
            
            logger.info(f"Batch processing completed: {batch_count} batches, {total_records} records")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    def _json_batch_reader(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Create batch reader for JSON Lines format."""
        try:
            batch_data = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        batch_data.append(record)
                        
                        if len(batch_data) >= self.batch_size:
                            yield pd.DataFrame(batch_data)
                            batch_data = []
                    except json.JSONDecodeError:
                        continue
            
            # Yield remaining data
            if batch_data:
                yield pd.DataFrame(batch_data)
                
        except Exception as e:
            logger.error(f"Error reading JSON batch: {e}")
            raise
    
    def _combine_batch_results(self, results: List[Any], output_path: Union[str, Path]):
        """Combine batch processing results into single output."""
        try:
            output_path = Path(output_path)
            
            if all(isinstance(result, pd.DataFrame) for result in results):
                # Combine DataFrames
                combined_df = pd.concat(results, ignore_index=True)
                
                if output_path.suffix.lower() == '.csv':
                    combined_df.to_csv(output_path, index=False)
                elif output_path.suffix.lower() == '.json':
                    combined_df.to_json(output_path, orient='records')
                elif output_path.suffix.lower() == '.parquet' and PARQUET_AVAILABLE:
                    combined_df.to_parquet(output_path, index=False)
                else:
                    combined_df.to_pickle(output_path)
            
            else:
                # Save as JSON for other result types
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Batch results combined: {output_path}")
            
        except Exception as e:
            logger.error(f"Error combining batch results: {e}")
            raise

# Factory functions and utilities
def create_file_handler(base_path: str = "./data",
                       enable_compression: bool = True,
                       enable_encryption: bool = False) -> FileHandler:
    """Create a file handler instance."""
    return FileHandler(
        base_path=base_path,
        enable_compression=enable_compression,
        enable_encryption=enable_encryption
    )

def create_data_exporter(file_handler: FileHandler = None) -> DataExporter:
    """Create a data exporter instance."""
    return DataExporter(file_handler)

def create_cloud_storage_handler(provider: str = "aws",
                                bucket_name: str = None,
                                region: str = "us-east-1") -> "CloudStorageHandler":
    """Create a cloud storage handler instance."""
    return CloudStorageHandler(
        provider=provider,
        bucket_name=bucket_name,
        region=region
    )

def create_batch_processor(batch_size: int = 10000,
                          max_workers: int = 4) -> BatchProcessor:
    """Create a batch processor instance."""
    return BatchProcessor(
        batch_size=batch_size,
        max_workers=max_workers
    )

# Utility functions
def get_file_format(file_path: Union[str, Path]) -> str:
    """Detect file format from extension."""
    return Path(file_path).suffix.lower().lstrip('.')

def estimate_file_processing_time(file_path: Union[str, Path],
                                 processing_rate_mb_per_sec: float = 10.0) -> float:
    """Estimate file processing time based on size."""
    try:
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        return file_size_mb / processing_rate_mb_per_sec
    except:
        return 0.0

def validate_file_path(file_path: Union[str, Path],
                      must_exist: bool = True,
                      create_parent_dirs: bool = False) -> bool:
    """Validate file path."""
    try:
        file_path = Path(file_path)
        
        if create_parent_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if must_exist and not file_path.exists():
            return False
        
        if not must_exist and not file_path.parent.exists():
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file path: {e}")
        return False

# Log module initialization
logger.info("BatteryMind File Handlers Module v1.0.0 loaded successfully")
