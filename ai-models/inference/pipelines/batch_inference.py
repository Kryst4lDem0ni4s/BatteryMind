"""
BatteryMind - Batch Inference System

High-performance batch inference system for processing large volumes of battery
data with distributed processing, queue management, and result aggregation.

Features:
- Distributed batch processing with worker pools
- Asynchronous job queue management
- Result aggregation and analysis
- Progress tracking and monitoring
- Error handling and retry mechanisms
- Memory-efficient streaming processing

Author: BatteryMind Development Team
Version: 1.0.0
"""

import asyncio
import concurrent.futures
import logging
import time
from typing import Dict, List, Optional, Union, Any, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import traceback
from collections import defaultdict
import pickle
import uuid

# Data processing
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Async and concurrency
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiofiles
import aiohttp

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Internal imports
from .inference_pipeline import BatteryInferencePipeline, InferenceConfig, BatteryDataInput, InferenceResult
from ...utils.logging_utils import setup_logging
from ...utils.data_utils import chunk_dataframe, validate_data_schema

# Configure logging
logger = structlog.get_logger()

# Metrics
BATCH_JOBS_TOTAL = Counter('batch_jobs_total', 'Total batch jobs processed', ['status'])
BATCH_PROCESSING_TIME = Histogram('batch_processing_time_seconds', 'Batch processing time')
BATCH_SIZE_HISTOGRAM = Histogram('batch_size_distribution', 'Distribution of batch sizes')
ACTIVE_BATCH_JOBS = Gauge('active_batch_jobs', 'Number of active batch jobs')

@dataclass
class BatchJobConfig:
    """Configuration for batch processing jobs."""
    
    # Job identification
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_name: str = "battery_batch_inference"
    
    # Processing parameters
    batch_size: int = 1000
    max_workers: int = 4
    max_concurrent_batches: int = 10
    chunk_size: int = 10000
    
    # Performance settings
    enable_parallel_processing: bool = True
    enable_streaming: bool = False
    memory_limit_gb: float = 8.0
    
    # Output configuration
    output_format: str = "json"  # json, csv, parquet
    output_path: Optional[str] = None
    include_metadata: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    continue_on_error: bool = True
    error_threshold: float = 0.1  # Max error rate before stopping
    
    # Monitoring
    enable_progress_tracking: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000

@dataclass
class BatchJobStatus:
    """Status tracking for batch jobs."""
    
    job_id: str
    status: str  # queued, running, completed, failed, cancelled
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Progress tracking
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    
    # Performance metrics
    processing_rate: float = 0.0  # items per second
    estimated_completion: Optional[datetime] = None
    
    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0
    last_error: Optional[str] = None
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

@dataclass
class BatchJobResult:
    """Result of a batch processing job."""
    
    job_id: str
    status: str
    
    # Summary statistics
    total_processed: int
    successful_predictions: int
    failed_predictions: int
    
    # Performance metrics
    total_time_seconds: float
    average_processing_time_ms: float
    throughput_items_per_second: float
    
    # Quality metrics
    average_confidence: float
    average_data_quality: float
    
    # Output information
    output_files: List[str]
    result_summary: Dict[str, Any]
    
    # Error information
    error_summary: Dict[str, Any]
    
    # Metadata
    config: BatchJobConfig
    created_at: datetime
    completed_at: Optional[datetime] = None

class BatchInferenceProcessor:
    """
    High-performance batch inference processor for battery data.
    """
    
    def __init__(self, inference_config: InferenceConfig, batch_config: BatchJobConfig):
        self.inference_config = inference_config
        self.batch_config = batch_config
        
        # Initialize inference pipeline
        self.inference_pipeline = BatteryInferencePipeline(inference_config)
        
        # Job tracking
        self.active_jobs: Dict[str, BatchJobStatus] = {}
        self.job_results: Dict[str, BatchJobResult] = {}
        
        # Worker management
        self.executor = None
        self.worker_pool = None
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        
        logger.info("BatchInferenceProcessor initialized", 
                   batch_config=batch_config.__dict__)
    
    async def process_batch_from_dataframe(self, 
                                         df: pd.DataFrame, 
                                         config: Optional[BatchJobConfig] = None) -> BatchJobResult:
        """
        Process batch inference from a pandas DataFrame.
        
        Args:
            df: DataFrame with battery data
            config: Optional batch configuration override
            
        Returns:
            BatchJobResult: Results of batch processing
        """
        job_config = config or self.batch_config
        job_id = job_config.job_id
        
        logger.info("Starting batch processing from DataFrame", 
                   job_id=job_id, 
                   data_shape=df.shape)
        
        # Create job status
        job_status = BatchJobStatus(
            job_id=job_id,
            status="running",
            start_time=datetime.now(),
            total_items=len(df)
        )
        self.active_jobs[job_id] = job_status
        
        try:
            # Validate input data
            self._validate_batch_input(df)
            
            # Convert DataFrame to input objects
            input_data = self._dataframe_to_input_objects(df)
            
            # Process batch
            results = await self._process_batch_data(input_data, job_config, job_status)
            
            # Create job result
            job_result = self._create_job_result(job_config, job_status, results)
            
            # Save results
            if job_config.output_path:
                await self._save_batch_results(results, job_config)
            
            # Update status
            job_status.status = "completed"
            job_status.end_time = datetime.now()
            
            self.job_results[job_id] = job_result
            
            logger.info("Batch processing completed", 
                       job_id=job_id,
                       processed_items=job_status.processed_items,
                       success_rate=job_status.successful_items / job_status.total_items)
            
            return job_result
            
        except Exception as e:
            logger.error("Batch processing failed", 
                        job_id=job_id,
                        error=str(e))
            
            job_status.status = "failed"
            job_status.last_error = str(e)
            job_status.end_time = datetime.now()
            
            # Create error result
            error_result = BatchJobResult(
                job_id=job_id,
                status="failed",
                total_processed=job_status.processed_items,
                successful_predictions=job_status.successful_items,
                failed_predictions=job_status.failed_items,
                total_time_seconds=0.0,
                average_processing_time_ms=0.0,
                throughput_items_per_second=0.0,
                average_confidence=0.0,
                average_data_quality=0.0,
                output_files=[],
                result_summary={},
                error_summary={"error": str(e)},
                config=job_config,
                created_at=job_status.start_time or datetime.now()
            )
            
            self.job_results[job_id] = error_result
            return error_result
        
        finally:
            # Cleanup
            if job_id in self.active_jobs:
                self.active_jobs[job_id] = job_status
            
            ACTIVE_BATCH_JOBS.dec()
    
    async def process_batch_from_file(self, 
                                    file_path: str, 
                                    config: Optional[BatchJobConfig] = None) -> BatchJobResult:
        """
        Process batch inference from a file.
        
        Args:
            file_path: Path to input file (CSV, JSON, Parquet)
            config: Optional batch configuration override
            
        Returns:
            BatchJobResult: Results of batch processing
        """
        job_config = config or self.batch_config
        
        logger.info("Starting batch processing from file", 
                   file_path=file_path,
                   job_id=job_config.job_id)
        
        # Load data based on file type
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Process DataFrame
        return await self.process_batch_from_dataframe(df, job_config)
    
    async def process_streaming_batch(self, 
                                    data_stream: Iterator[BatteryDataInput],
                                    config: Optional[BatchJobConfig] = None) -> BatchJobResult:
        """
        Process batch inference from a streaming data source.
        
        Args:
            data_stream: Iterator of battery data inputs
            config: Optional batch configuration override
            
        Returns:
            BatchJobResult: Results of batch processing
        """
        job_config = config or self.batch_config
        job_id = job_config.job_id
        
        logger.info("Starting streaming batch processing", job_id=job_id)
        
        # Create job status
        job_status = BatchJobStatus(
            job_id=job_id,
            status="running",
            start_time=datetime.now(),
            total_items=0  # Unknown for streaming
        )
        self.active_jobs[job_id] = job_status
        
        all_results = []
        batch_buffer = []
        
        try:
            async for data_item in data_stream:
                batch_buffer.append(data_item)
                
                # Process when buffer is full
                if len(batch_buffer) >= job_config.batch_size:
                    batch_results = await self._process_batch_chunk(batch_buffer, job_config, job_status)
                    all_results.extend(batch_results)
                    
                    # Update progress
                    job_status.processed_items += len(batch_buffer)
                    job_status.successful_items += len([r for r in batch_results if not r.alerts])
                    job_status.failed_items += len([r for r in batch_results if r.alerts])
                    
                    # Clear buffer
                    batch_buffer = []
                    
                    # Check error rate
                    if job_status.processed_items > 0:
                        job_status.error_rate = job_status.failed_items / job_status.processed_items
                        if job_status.error_rate > job_config.error_threshold:
                            raise Exception(f"Error rate {job_status.error_rate:.2%} exceeds threshold")
            
            # Process remaining items
            if batch_buffer:
                batch_results = await self._process_batch_chunk(batch_buffer, job_config, job_status)
                all_results.extend(batch_results)
                
                job_status.processed_items += len(batch_buffer)
                job_status.successful_items += len([r for r in batch_results if not r.alerts])
                job_status.failed_items += len([r for r in batch_results if r.alerts])
            
            # Create job result
            job_result = self._create_job_result(job_config, job_status, all_results)
            
            # Save results
            if job_config.output_path:
                await self._save_batch_results(all_results, job_config)
            
            job_status.status = "completed"
            job_status.end_time = datetime.now()
            job_status.total_items = job_status.processed_items
            
            self.job_results[job_id] = job_result
            
            logger.info("Streaming batch processing completed", 
                       job_id=job_id,
                       processed_items=job_status.processed_items)
            
            return job_result
            
        except Exception as e:
            logger.error("Streaming batch processing failed", 
                        job_id=job_id,
                        error=str(e))
            
            job_status.status = "failed"
            job_status.last_error = str(e)
            job_status.end_time = datetime.now()
            
            raise
        
        finally:
            if job_id in self.active_jobs:
                self.active_jobs[job_id] = job_status
    
    def _validate_batch_input(self, df: pd.DataFrame):
        """Validate batch input data."""
        required_columns = ['battery_id', 'timestamp', 'voltage', 'current', 'temperature', 'state_of_charge']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values in required columns
        for col in required_columns:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                logger.warning(f"Column {col} has {null_count} null values")
        
        # Data type validation
        numeric_columns = ['voltage', 'current', 'temperature', 'state_of_charge']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")
        
        logger.info("Batch input validation completed", 
                   total_rows=len(df),
                   columns=list(df.columns))
    
    def _dataframe_to_input_objects(self, df: pd.DataFrame) -> List[BatteryDataInput]:
        """Convert DataFrame to BatteryDataInput objects."""
        input_objects = []
        
        for _, row in df.iterrows():
            try:
                # Required fields
                input_data = BatteryDataInput(
                    battery_id=str(row['battery_id']),
                    timestamp=pd.to_datetime(row['timestamp']),
                    voltage=float(row['voltage']),
                    current=float(row['current']),
                    temperature=float(row['temperature']),
                    state_of_charge=float(row['state_of_charge']),
                    cycle_count=int(row.get('cycle_count', 0)),
                    age_days=int(row.get('age_days', 0)),
                    
                    # Optional fields
                    state_of_health=float(row['state_of_health']) if 'state_of_health' in row and pd.notna(row['state_of_health']) else None,
                    capacity=float(row['capacity']) if 'capacity' in row and pd.notna(row['capacity']) else None,
                    internal_resistance=float(row['internal_resistance']) if 'internal_resistance' in row and pd.notna(row['internal_resistance']) else None,
                    
                    # Metadata
                    battery_type=str(row.get('battery_type', 'li_ion')),
                    manufacturer=str(row.get('manufacturer', 'unknown')),
                    model_name=str(row.get('model_name', 'unknown'))
                )
                
                input_objects.append(input_data)
                
            except Exception as e:
                logger.warning(f"Failed to convert row to BatteryDataInput", 
                             row_index=row.name,
                             error=str(e))
        
        logger.info(f"Converted {len(input_objects)} rows to input objects")
        return input_objects
    
    async def _process_batch_data(self, 
                                input_data: List[BatteryDataInput], 
                                config: BatchJobConfig, 
                                job_status: BatchJobStatus) -> List[InferenceResult]:
        """Process batch data with parallel processing."""
        all_results = []
        
        # Split data into chunks
        data_chunks = [input_data[i:i + config.batch_size] 
                      for i in range(0, len(input_data), config.batch_size)]
        
        logger.info(f"Processing {len(data_chunks)} batches", 
                   total_items=len(input_data),
                   batch_size=config.batch_size)
        
        # Process chunks
        if config.enable_parallel_processing:
            # Process chunks concurrently
            semaphore = asyncio.Semaphore(config.max_concurrent_batches)
            
            async def process_chunk_with_semaphore(chunk):
                async with semaphore:
                    return await self._process_batch_chunk(chunk, config, job_status)
            
            # Create tasks for all chunks
            tasks = [process_chunk_with_semaphore(chunk) for chunk in data_chunks]
            
            # Process with progress tracking
            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    chunk_results = await task
                    all_results.extend(chunk_results)
                    
                    # Update progress
                    job_status.processed_items += len(chunk_results)
                    job_status.successful_items += len([r for r in chunk_results if not r.alerts])
                    job_status.failed_items += len([r for r in chunk_results if r.alerts])
                    
                    # Update processing rate
                    if job_status.start_time:
                        elapsed = (datetime.now() - job_status.start_time).total_seconds()
                        job_status.processing_rate = job_status.processed_items / elapsed
                        
                        # Estimate completion time
                        if job_status.processing_rate > 0:
                            remaining_items = job_status.total_items - job_status.processed_items
                            remaining_time = remaining_items / job_status.processing_rate
                            job_status.estimated_completion = datetime.now() + timedelta(seconds=remaining_time)
                    
                    # Check error rate
                    if job_status.processed_items > 0:
                        job_status.error_rate = job_status.failed_items / job_status.processed_items
                        if job_status.error_rate > config.error_threshold:
                            logger.error(f"Error rate {job_status.error_rate:.2%} exceeds threshold")
                            if not config.continue_on_error:
                                raise Exception(f"Error rate {job_status.error_rate:.2%} exceeds threshold")
                    
                    # Progress callback
                    for callback in self.progress_callbacks:
                        callback(job_status)
                    
                    # Checkpoint
                    if (config.save_checkpoints and 
                        job_status.processed_items % config.checkpoint_interval == 0):
                        await self._save_checkpoint(job_status, all_results, config)
                    
                    logger.info(f"Processed chunk {i+1}/{len(data_chunks)}", 
                               progress=f"{job_status.processed_items}/{job_status.total_items}")
                
                except Exception as e:
                    logger.error(f"Chunk processing failed", 
                                chunk_index=i,
                                error=str(e))
                    
                    if not config.continue_on_error:
                        raise
        else:
            # Sequential processing
            for i, chunk in enumerate(data_chunks):
                try:
                    chunk_results = await self._process_batch_chunk(chunk, config, job_status)
                    all_results.extend(chunk_results)
                    
                    # Update progress
                    job_status.processed_items += len(chunk_results)
                    job_status.successful_items += len([r for r in chunk_results if not r.alerts])
                    job_status.failed_items += len([r for r in chunk_results if r.alerts])
                    
                    logger.info(f"Processed chunk {i+1}/{len(data_chunks)}")
                    
                except Exception as e:
                    logger.error(f"Chunk processing failed", 
                                chunk_index=i,
                                error=str(e))
                    
                    if not config.continue_on_error:
                        raise
        
        return all_results
    
    async def _process_batch_chunk(self, 
                                 chunk: List[BatteryDataInput], 
                                 config: BatchJobConfig, 
                                 job_status: BatchJobStatus) -> List[InferenceResult]:
        """Process a single batch chunk."""
        try:
            # Use the inference pipeline's batch prediction
            results = await self.inference_pipeline.predict_batch(chunk)
            
            # Update job status
            job_status.successful_items += len([r for r in results if not r.alerts])
            job_status.failed_items += len([r for r in results if r.alerts])
            
            return results
            
        except Exception as e:
            logger.error("Batch chunk processing failed", 
                        chunk_size=len(chunk),
                        error=str(e))
            
            # Create error results for the chunk
            error_results = []
            for input_data in chunk:
                error_result = InferenceResult(
                    battery_id=input_data.battery_id,
                    timestamp=input_data.timestamp,
                    prediction_timestamp=datetime.now(),
                    state_of_health=0.0,
                    state_of_health_confidence=0.0,
                    degradation_rate=0.0,
                    remaining_useful_life_days=0,
                    capacity_forecast_30d=0.0,
                    capacity_forecast_90d=0.0,
                    capacity_forecast_365d=0.0,
                    optimal_charging_current=0.0,
                    optimal_charging_voltage=0.0,
                    recommended_actions=[],
                    prediction_uncertainty=1.0,
                    model_confidence=0.0,
                    model_version="1.0.0",
                    inference_time_ms=0.0,
                    models_used=[],
                    data_quality_score=0.0,
                    prediction_quality_score=0.0,
                    alerts=[f"Batch processing failed: {str(e)}"],
                    warnings=[]
                )
                error_results.append(error_result)
            
            job_status.failed_items += len(error_results)
            return error_results
    
    def _create_job_result(self, 
                          config: BatchJobConfig, 
                          job_status: BatchJobStatus, 
                          results: List[InferenceResult]) -> BatchJobResult:
        """Create comprehensive job result."""
        if not results:
            return BatchJobResult(
                job_id=config.job_id,
                status="failed",
                total_processed=0,
                successful_predictions=0,
                failed_predictions=0,
                total_time_seconds=0.0,
                average_processing_time_ms=0.0,
                throughput_items_per_second=0.0,
                average_confidence=0.0,
                average_data_quality=0.0,
                output_files=[],
                result_summary={},
                error_summary={"error": "No results generated"},
                config=config,
                created_at=job_status.start_time or datetime.now()
            )
        
        # Calculate metrics
        total_time = ((job_status.end_time or datetime.now()) - 
                     (job_status.start_time or datetime.now())).total_seconds()
        
        successful_results = [r for r in results if not r.alerts]
        failed_results = [r for r in results if r.alerts]
        
        # Calculate averages
        avg_inference_time = np.mean([r.inference_time_ms for r in results])
        avg_confidence = np.mean([r.model_confidence for r in results])
        avg_data_quality = np.mean([r.data_quality_score for r in results])
        
        # Calculate throughput
        throughput = len(results) / total_time if total_time > 0 else 0
        
        # Create result summary
        result_summary = {
            "health_distribution": {
                "excellent": len([r for r in results if r.state_of_health > 0.9]),
                "good": len([r for r in results if 0.8 <= r.state_of_health <= 0.9]),
                "fair": len([r for r in results if 0.7 <= r.state_of_health < 0.8]),
                "poor": len([r for r in results if r.state_of_health < 0.7])
            },
            "average_health": np.mean([r.state_of_health for r in results]),
            "average_degradation_rate": np.mean([r.degradation_rate for r in results]),
            "average_remaining_life": np.mean([r.remaining_useful_life_days for r in results])
        }
        
        # Create error summary
        error_types = defaultdict(int)
        for result in failed_results:
            for alert in result.alerts:
                error_types[alert] += 1
        
        error_summary = {
            "total_errors": len(failed_results),
            "error_rate": len(failed_results) / len(results),
            "error_types": dict(error_types),
            "common_errors": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return BatchJobResult(
            job_id=config.job_id,
            status=job_status.status,
            total_processed=len(results),
            successful_predictions=len(successful_results),
            failed_predictions=len(failed_results),
            total_time_seconds=total_time,
            average_processing_time_ms=avg_inference_time,
            throughput_items_per_second=throughput,
            average_confidence=avg_confidence,
            average_data_quality=avg_data_quality,
            output_files=[],
            result_summary=result_summary,
            error_summary=error_summary,
            config=config,
            created_at=job_status.start_time or datetime.now(),
            completed_at=job_status.end_time
        )
    
    async def _save_batch_results(self, results: List[InferenceResult], config: BatchJobConfig):
        """Save batch results to specified output format."""
        if not config.output_path:
            return
        
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config.output_format == "json":
            # Save as JSON
            json_data = []
            for result in results:
                json_data.append({
                    "input_id": result.input_id,
                    "battery_id": result.battery_id,
                    "predictions": result.predictions,
                    "confidence_scores": result.confidence_scores,
                    "processing_time_ms": result.processing_time_ms,
                    "data_quality_score": result.data_quality_score,
                    "anomaly_flags": result.anomaly_flags,
                    "feature_importance": result.feature_importance,
                    "timestamp": result.timestamp.isoformat(),
                    "model_version": result.model_version,
                    "success": result.success,
                    "error_message": result.error_message
                })
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
        
        elif config.output_format == "csv":
            # Save as CSV
            csv_data = []
            for result in results:
                row = {
                    "input_id": result.input_id,
                    "battery_id": result.battery_id,
                    "timestamp": result.timestamp.isoformat(),
                    "processing_time_ms": result.processing_time_ms,
                    "data_quality_score": result.data_quality_score,
                    "model_version": result.model_version,
                    "success": result.success,
                    "error_message": result.error_message
                }
                
                # Add prediction columns
                if result.predictions:
                    for key, value in result.predictions.items():
                        row[f"prediction_{key}"] = value
                
                # Add confidence columns
                if result.confidence_scores:
                    for key, value in result.confidence_scores.items():
                        row[f"confidence_{key}"] = value
                
                # Add anomaly flags
                if result.anomaly_flags:
                    for key, value in result.anomaly_flags.items():
                        row[f"anomaly_{key}"] = value
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
        
        elif config.output_format == "parquet":
            # Save as Parquet
            csv_data = []
            for result in results:
                row = {
                    "input_id": result.input_id,
                    "battery_id": result.battery_id,
                    "timestamp": result.timestamp,
                    "processing_time_ms": result.processing_time_ms,
                    "data_quality_score": result.data_quality_score,
                    "model_version": result.model_version,
                    "success": result.success,
                    "error_message": result.error_message
                }
                
                # Add prediction columns
                if result.predictions:
                    for key, value in result.predictions.items():
                        row[f"prediction_{key}"] = value
                
                # Add confidence columns
                if result.confidence_scores:
                    for key, value in result.confidence_scores.items():
                        row[f"confidence_{key}"] = value
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            df.to_parquet(output_path, index=False)
        
        logger.info(f"Batch results saved to {output_path}")

    async def _cleanup_job(self, job_id: str):
        """Clean up job resources and temporary files."""
        try:
            # Remove job status
            if job_id in self.job_statuses:
                del self.job_statuses[job_id]
            
            # Clean up temporary files
            temp_dir = Path(f"/tmp/batch_inference/{job_id}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            # Clean up model cache if needed
            if hasattr(self, 'model_cache'):
                # Remove old cached models to free memory
                cache_size = len(self.model_cache)
                if cache_size > 5:  # Keep only 5 most recent models
                    oldest_keys = list(self.model_cache.keys())[:-5]
                    for key in oldest_keys:
                        del self.model_cache[key]
            
            logger.info(f"Cleaned up resources for job {job_id}")
        
        except Exception as e:
            logger.error(f"Error cleaning up job {job_id}: {e}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running batch job."""
        if job_id not in self.job_statuses:
            return False
        
        job_status = self.job_statuses[job_id]
        
        if job_status.status in ["completed", "failed", "cancelled"]:
            return False
        
        # Update job status
        job_status.status = "cancelled"
        job_status.end_time = datetime.now()
        
        # Clean up resources
        await self._cleanup_job(job_id)
        
        logger.info(f"Cancelled batch job {job_id}")
        return True

    async def retry_failed_items(self, job_id: str, max_retries: int = 3) -> BatchJobResult:
        """Retry failed items from a previous batch job."""
        if job_id not in self.job_statuses:
            raise ValueError(f"Job {job_id} not found")
        
        job_status = self.job_statuses[job_id]
        
        if job_status.status != "completed":
            raise ValueError(f"Job {job_id} must be completed to retry failed items")
        
        # Get failed items
        failed_items = [item for item in job_status.processed_items if not item.get("success", False)]
        
        if not failed_items:
            logger.info(f"No failed items to retry for job {job_id}")
            return None
        
        # Create retry configuration
        retry_config = BatchJobConfig(
            job_id=f"{job_id}_retry",
            model_name=job_status.config.model_name,
            model_version=job_status.config.model_version,
            batch_size=job_status.config.batch_size,
            max_workers=job_status.config.max_workers,
            enable_monitoring=job_status.config.enable_monitoring,
            output_path=job_status.config.output_path,
            output_format=job_status.config.output_format,
            max_retries=max_retries,
            retry_delay_seconds=job_status.config.retry_delay_seconds * 2,  # Exponential backoff
            enable_caching=job_status.config.enable_caching,
            cache_ttl_seconds=job_status.config.cache_ttl_seconds
        )
        
        # Extract input data for failed items
        retry_data = []
        for item in failed_items:
            if "input_data" in item:
                retry_data.append(item["input_data"])
        
        logger.info(f"Retrying {len(retry_data)} failed items for job {job_id}")
        
        # Run retry batch job
        return await self.process_batch(retry_data, retry_config)

    def get_job_statistics(self) -> Dict[str, Any]:
        """Get overall batch processing statistics."""
        stats = {
            "total_jobs": len(self.job_statuses),
            "jobs_by_status": defaultdict(int),
            "total_items_processed": 0,
            "total_successful_predictions": 0,
            "total_failed_predictions": 0,
            "average_processing_time_ms": 0,
            "average_throughput_items_per_second": 0,
            "most_common_errors": [],
            "model_usage_stats": defaultdict(int)
        }
        
        processing_times = []
        throughputs = []
        all_errors = []
        
        for job_status in self.job_statuses.values():
            stats["jobs_by_status"][job_status.status] += 1
            
            if hasattr(job_status, 'result') and job_status.result:
                result = job_status.result
                stats["total_items_processed"] += result.total_processed
                stats["total_successful_predictions"] += result.successful_predictions
                stats["total_failed_predictions"] += result.failed_predictions
                
                if result.average_processing_time_ms:
                    processing_times.append(result.average_processing_time_ms)
                
                if result.throughput_items_per_second:
                    throughputs.append(result.throughput_items_per_second)
                
                if result.error_summary and "common_errors" in result.error_summary:
                    all_errors.extend([error[0] for error in result.error_summary["common_errors"]])
                
                stats["model_usage_stats"][f"{job_status.config.model_name}:{job_status.config.model_version}"] += 1
        
        # Calculate averages
        if processing_times:
            stats["average_processing_time_ms"] = sum(processing_times) / len(processing_times)
        
        if throughputs:
            stats["average_throughput_items_per_second"] = sum(throughputs) / len(throughputs)
        
        # Most common errors
        if all_errors:
            error_counts = defaultdict(int)
            for error in all_errors:
                error_counts[error] += 1
            stats["most_common_errors"] = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the batch inference pipeline."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_jobs": len([job for job in self.job_statuses.values() if job.status == "running"]),
            "total_jobs": len(self.job_statuses),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_usage_percent": psutil.Process().cpu_percent(),
            "disk_usage_gb": shutil.disk_usage("/").used / (1024**3),
            "available_models": [],
            "system_resources": {
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "disk_total_gb": shutil.disk_usage("/").total / (1024**3),
                "disk_free_gb": shutil.disk_usage("/").free / (1024**3)
            }
        }
        
        # Check available models
        try:
            # This would typically query the model registry
            model_registry_path = Path("../../model-artifacts/version_control/model_registry.json")
            if model_registry_path.exists():
                with open(model_registry_path, 'r') as f:
                    model_registry = json.load(f)
                
                health_status["available_models"] = list(model_registry.get("models", {}).keys())
        except Exception as e:
            logger.warning(f"Could not load model registry: {e}")
            health_status["available_models"] = ["model_registry_unavailable"]
        
        # Check system health
        memory_usage_percent = (psutil.virtual_memory().total - psutil.virtual_memory().available) / psutil.virtual_memory().total * 100
        disk_usage_percent = shutil.disk_usage("/").used / shutil.disk_usage("/").total * 100
        
        if memory_usage_percent > 90:
            health_status["status"] = "warning"
            health_status["warnings"] = health_status.get("warnings", [])
            health_status["warnings"].append("High memory usage")
        
        if disk_usage_percent > 85:
            health_status["status"] = "warning"
            health_status["warnings"] = health_status.get("warnings", [])
            health_status["warnings"].append("High disk usage")
        
        # Check for stuck jobs
        stuck_jobs = []
        current_time = datetime.now()
        for job_id, job_status in self.job_statuses.items():
            if job_status.status == "running":
                if job_status.start_time and (current_time - job_status.start_time).total_seconds() > 3600:  # 1 hour
                    stuck_jobs.append(job_id)
        
        if stuck_jobs:
            health_status["status"] = "warning"
            health_status["warnings"] = health_status.get("warnings", [])
            health_status["warnings"].append(f"Potential stuck jobs: {stuck_jobs}")
        
        return health_status

    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize batch inference performance based on system resources."""
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {},
            "recommendations": []
        }
        
        # Get system resources
        memory_info = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        # Memory optimization
        if memory_info.percent > 80:
            # Clear model cache
            if hasattr(self, 'model_cache'):
                old_cache_size = len(self.model_cache)
                self.model_cache.clear()
                optimization_results["optimizations_applied"].append(f"Cleared model cache ({old_cache_size} models)")
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            optimization_results["optimizations_applied"].append(f"Garbage collection freed {collected} objects")
        
        # CPU optimization
        if cpu_count > 4:
            recommended_workers = min(cpu_count - 2, 8)  # Leave 2 cores free, cap at 8
            optimization_results["recommendations"].append(f"Consider using {recommended_workers} workers for optimal performance")
        
        # Batch size optimization
        if memory_info.available > 8 * (1024**3):  # 8GB available
            optimization_results["recommendations"].append("Consider increasing batch size to 64 for better throughput")
        elif memory_info.available < 2 * (1024**3):  # Less than 2GB available
            optimization_results["recommendations"].append("Consider decreasing batch size to 16 to reduce memory usage")
        
        # Disk optimization
        disk_usage = shutil.disk_usage("/")
        if disk_usage.free < 1 * (1024**3):  # Less than 1GB free
            optimization_results["recommendations"].append("Low disk space - consider cleaning up old job artifacts")
        
        # Model loading optimization
        optimization_results["recommendations"].append("Enable model caching for frequently used models")
        optimization_results["recommendations"].append("Use model quantization for edge deployment")
        
        return optimization_results

    def __del__(self):
        """Cleanup when the batch processor is destroyed."""
        try:
            # Clean up all active jobs
            for job_id in list(self.job_statuses.keys()):
                asyncio.create_task(self._cleanup_job(job_id))
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    # Factory function for creating batch processors
    def create_batch_processor(config: Optional[Dict[str, Any]] = None) -> BatteryBatchInferenceProcessor:
        """
        Factory function to create a batch inference processor.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            BatteryBatchInferenceProcessor: Configured batch processor
        """
        if config is None:
            config = {}
        
        processor = BatteryBatchInferenceProcessor()
        
        # Apply configuration
        if "max_concurrent_jobs" in config:
            processor.max_concurrent_jobs = config["max_concurrent_jobs"]
        
        if "default_batch_size" in config:
            processor.default_batch_size = config["default_batch_size"]
        
        if "enable_monitoring" in config:
            processor.enable_monitoring = config["enable_monitoring"]
        
        return processor

    # Utility functions for batch processing
    def validate_batch_input(data: List[Dict[str, Any]]) -> List[str]:
        """
        Validate batch input data format.
        
        Args:
            data: List of input data dictionaries
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if not isinstance(data, list):
            errors.append("Input data must be a list")
            return errors
        
        if len(data) == 0:
            errors.append("Input data cannot be empty")
            return errors
        
        required_fields = ["battery_id", "sensor_data"]
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"Item {i}: Must be a dictionary")
                continue
            
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {i}: Missing required field '{field}'")
            
            # Validate sensor data structure
            if "sensor_data" in item:
                sensor_data = item["sensor_data"]
                if not isinstance(sensor_data, dict):
                    errors.append(f"Item {i}: sensor_data must be a dictionary")
                else:
                    # Check for required sensor fields
                    required_sensors = ["voltage", "current", "temperature"]
                    for sensor in required_sensors:
                        if sensor not in sensor_data:
                            errors.append(f"Item {i}: Missing sensor data for '{sensor}'")
        
        return errors

    def estimate_processing_time(data_size: int, batch_size: int = 32, 
                            avg_processing_time_ms: float = 100.0) -> Dict[str, float]:
        """
        Estimate processing time for batch inference.
        
        Args:
            data_size: Number of items to process
            batch_size: Batch size for processing
            avg_processing_time_ms: Average processing time per item
            
        Returns:
            Dictionary with time estimates
        """
        num_batches = math.ceil(data_size / batch_size)
        
        # Conservative estimates including overhead
        total_time_seconds = (num_batches * avg_processing_time_ms * batch_size) / 1000
        overhead_factor = 1.2  # 20% overhead for I/O, queueing, etc.
        
        return {
            "estimated_total_time_seconds": total_time_seconds * overhead_factor,
            "estimated_total_time_minutes": (total_time_seconds * overhead_factor) / 60,
            "estimated_throughput_items_per_second": data_size / (total_time_seconds * overhead_factor),
            "number_of_batches": num_batches,
            "average_batch_time_seconds": (total_time_seconds / num_batches) * overhead_factor
        }

    # Example usage and testing
    if __name__ == "__main__":
        async def main():
            # Create batch processor
            processor = create_batch_processor({
                "max_concurrent_jobs": 3,
                "default_batch_size": 32,
                "enable_monitoring": True
            })
            
            # Create sample data
            sample_data = [
                {
                    "battery_id": "battery_001",
                    "sensor_data": {
                        "voltage": 3.7,
                        "current": 2.5,
                        "temperature": 25.0,
                        "soc": 0.8
                    }
                },
                {
                    "battery_id": "battery_002",
                    "sensor_data": {
                        "voltage": 3.6,
                        "current": 1.8,
                        "temperature": 28.0,
                        "soc": 0.7
                    }
                }
            ]
            
            # Validate input
            errors = validate_batch_input(sample_data)
            if errors:
                print(f"Validation errors: {errors}")
                return
            
            # Estimate processing time
            time_estimate = estimate_processing_time(len(sample_data))
            print(f"Estimated processing time: {time_estimate}")
            
            # Create batch job configuration
            config = BatchJobConfig(
                job_id="test_batch_001",
                model_name="transformer_battery_health",
                model_version="1.0.0",
                batch_size=32,
                output_path="./test_batch_results.json",
                output_format="json"
            )
            
            # Process batch
            print("Starting batch processing...")
            result = await processor.process_batch(sample_data, config)
            
            if result:
                print(f"Batch processing completed:")
                print(f"  Total processed: {result.total_processed}")
                print(f"  Successful: {result.successful_predictions}")
                print(f"  Failed: {result.failed_predictions}")
                print(f"  Average time: {result.average_processing_time_ms:.2f}ms")
                print(f"  Throughput: {result.throughput_items_per_second:.2f} items/sec")
            
            # Check health
            health = await processor.health_check()
            print(f"System health: {health['status']}")
            
            # Get statistics
            stats = processor.get_job_statistics()
            print(f"Total jobs processed: {stats['total_jobs']}")
            
        # Run example
        asyncio.run(main())
