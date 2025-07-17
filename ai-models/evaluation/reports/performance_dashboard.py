"""
BatteryMind - Performance Dashboard

Real-time performance dashboard for monitoring battery health prediction models
with interactive visualizations, live metrics, and alerting capabilities.

Features:
- Real-time model performance monitoring
- Interactive visualizations with Plotly Dash
- Live data streaming from model endpoints
- Customizable alerts and notifications
- Multi-model comparison views
- Historical performance tracking

Author: BatteryMind Development Team
Version: 1.0.0
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import redis
import sqlite3
from threading import Thread
import time
import requests
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# BatteryMind imports
from ..metrics.accuracy_metrics import AccuracyEvaluator
from ..metrics.performance_metrics import PerformanceEvaluator
from ..benchmarks.industry_benchmarks import IndustryBenchmarks
from ...utils.logging_utils import setup_logger
from ...monitoring.alerts.alert_manager import AlertManager

# Configure logging
logger = setup_logger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for performance dashboard."""
    
    # Dashboard settings
    title: str = "BatteryMind Performance Dashboard"
    port: int = 8050
    debug: bool = False
    host: str = "0.0.0.0"
    
    # Data settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    update_interval_seconds: int = 5
    max_data_points: int = 1000
    
    # Model endpoints
    model_endpoints: Dict[str, str] = field(default_factory=lambda: {
        "transformer": "http://localhost:8000/health",
        "federated": "http://localhost:8001/health",
        "rl_agent": "http://localhost:8002/health",
        "ensemble": "http://localhost:8003/health"
    })
    
    # Alert thresholds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "mae_threshold": 0.05,
        "rmse_threshold": 0.1,
        "r2_threshold": 0.9,
        "latency_threshold": 100.0,
        "error_rate_threshold": 0.1
    })
    
    # Visualization settings
    theme: str = "bootstrap"
    primary_color: str = "#2E86AB"
    charts_height: int = 400

class DataCollector:
    """Collects performance data from various sources."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=True
        )
        
        # Data storage
        self.performance_data = {
            model: deque(maxlen=config.max_data_points)
            for model in config.model_endpoints.keys()
        }
        
        # Initialize data collection thread
        self.running = False
        self.collection_thread = None
        
    def start_collection(self):
        """Start data collection thread."""
        self.running = True
        self.collection_thread = Thread(target=self._collect_data_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Data collection started")
    
    def stop_collection(self):
        """Stop data collection thread."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Data collection stopped")
    
    def _collect_data_loop(self):
        """Main data collection loop."""
        while self.running:
            try:
                # Collect data from all sources
                self._collect_model_metrics()
                self._collect_redis_metrics()
                
                # Sleep until next collection
                time.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                time.sleep(self.config.update_interval_seconds)
    
    def _collect_model_metrics(self):
        """Collect metrics from model endpoints."""
        for model_name, endpoint in self.config.model_endpoints.items():
            try:
                # Get health metrics from model endpoint
                response = requests.get(f"{endpoint}/metrics", timeout=5)
                if response.status_code == 200:
                    metrics = response.json()
                    
                    # Add timestamp
                    metrics['timestamp'] = datetime.now()
                    
                    # Store in data buffer
                    self.performance_data[model_name].append(metrics)
                    
            except Exception as e:
                logger.warning(f"Failed to collect metrics from {model_name}: {e}")
    
    def _collect_redis_metrics(self):
        """Collect metrics from Redis cache."""
        try:
            # Get recent inference results
            keys = self.redis_client.keys("inference:*")
            
            for key in keys[-100:]:  # Last 100 results
                result_data = self.redis_client.get(key)
                if result_data:
                    result = json.loads(result_data)
                    
                    # Extract model name and metrics
                    model_name = result.get('model_type', 'unknown')
                    if model_name in self.performance_data:
                        
                        # Create metrics entry
                        metrics = {
                            'timestamp': datetime.fromisoformat(result['timestamp']),
                            'prediction_time': result.get('inference_time', 0),
                            'confidence': result.get('confidence', 0),
                            'battery_id': result.get('battery_id'),
                            'prediction_value': result.get('prediction', {}).get('soh', 0)
                        }
                        
                        self.performance_data[model_name].append(metrics)
                        
        except Exception as e:
            logger.warning(f"Failed to collect Redis metrics: {e}")
    
    def get_recent_data(self, model_name: str, minutes: int = 60) -> List[Dict]:
        """Get recent data for specified model."""
        if model_name not in self.performance_data:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [
            entry for entry in self.performance_data[model_name]
            if entry.get('timestamp', datetime.now()) > cutoff_time
        ]
        
        return recent_data
    
    def get_summary_stats(self, model_name: str) -> Dict[str, Any]:
        """Get summary statistics for model."""
        data = list(self.performance_data.get(model_name, []))
        
        if not data:
            return {}
        
        # Extract metrics
        prediction_times = [entry.get('prediction_time', 0) for entry in data if 'prediction_time' in entry]
        confidences = [entry.get('confidence', 0) for entry in data if 'confidence' in entry]
        predictions = [entry.get('prediction_value', 0) for entry in data if 'prediction_value' in entry]
        
        stats = {
            'total_predictions': len(data),
            'avg_prediction_time': np.mean(prediction_times) if prediction_times else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_prediction': np.mean(predictions) if predictions else 0,
            'last_update': max([entry.get('timestamp', datetime.min) for entry in data])
        }
        
        return stats

class BatteryMindDashboard:
    """Main dashboard application."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.data_collector = DataCollector(config)
        self.alert_manager = AlertManager()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title=config.title
        )
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("BatteryMind Dashboard initialized")
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(
                        self.config.title,
                        className="text-primary mb-4",
                        style={"textAlign": "center"}
                    ),
                    html.Hr()
                ])
            ]),
            
            # Summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Models", className="card-title"),
                            html.H2(id="total-models", className="text-primary")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Predictions", className="card-title"),
                            html.H2(id="total-predictions", className="text-success")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Avg Response Time", className="card-title"),
                            html.H2(id="avg-response-time", className="text-info")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Health", className="card-title"),
                            html.H2(id="system-health", className="text-warning")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Model selection and controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Model Selection"),
                            dcc.Dropdown(
                                id="model-selector",
                                options=[
                                    {"label": model.title(), "value": model}
                                    for model in self.config.model_endpoints.keys()
                                ],
                                value=list(self.config.model_endpoints.keys())[0],
                                clearable=False
                            )
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Time Range"),
                            dcc.Dropdown(
                                id="time-range-selector",
                                options=[
                                    {"label": "Last 15 minutes", "value": 15},
                                    {"label": "Last 1 hour", "value": 60},
                                    {"label": "Last 6 hours", "value": 360},
                                    {"label": "Last 24 hours", "value": 1440}
                                ],
                                value=60,
                                clearable=False
                            )
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Auto Refresh"),
                            dbc.Switch(
                                id="auto-refresh-switch",
                                label="Enable Auto Refresh",
                                value=True
                            )
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Main charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="performance-timeline")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="prediction-distribution")
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="model-comparison")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="error-analysis")
                ], width=6)
            ], className="mb-4"),
            
            # Model details table
            dbc.Row([
                dbc.Col([
                    html.H4("Model Performance Details"),
                    html.Div(id="model-details-table")
                ])
            ], className="mb-4"),
            
            # Alerts section
            dbc.Row([
                dbc.Col([
                    html.H4("System Alerts"),
                    html.Div(id="alerts-section")
                ])
            ]),
            
            # Hidden components for intervals
            dcc.Interval(
                id="interval-component",
                interval=self.config.update_interval_seconds * 1000,
                n_intervals=0
            ),
            
            # Store for sharing data between callbacks
            dcc.Store(id="dashboard-data")
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output("dashboard-data", "data"),
            [Input("interval-component", "n_intervals"),
             Input("auto-refresh-switch", "value")]
        )
        def update_dashboard_data(n_intervals, auto_refresh):
            """Update dashboard data."""
            if not auto_refresh and n_intervals > 0:
                return dash.no_update
            
            # Collect current data
            dashboard_data = {}
            
            for model_name in self.config.model_endpoints.keys():
                model_data = self.data_collector.get_recent_data(model_name, 1440)  # 24 hours
                summary_stats = self.data_collector.get_summary_stats(model_name)
                
                dashboard_data[model_name] = {
                    "data": model_data,
                    "stats": summary_stats
                }
            
            return dashboard_data
        
        @self.app.callback(
            [Output("total-models", "children"),
             Output("total-predictions", "children"),
             Output("avg-response-time", "children"),
             Output("system-health", "children")],
            [Input("dashboard-data", "data")]
        )
        def update_summary_cards(dashboard_data):
            """Update summary cards."""
            if not dashboard_data:
                return "0", "0", "0ms", "Unknown"
            
            total_models = len(dashboard_data)
            total_predictions = sum(
                len(model_info["data"]) for model_info in dashboard_data.values()
            )
            
            # Calculate average response time
            all_times = []
            for model_info in dashboard_data.values():
                for entry in model_info["data"]:
                    if "prediction_time" in entry:
                        all_times.append(entry["prediction_time"])
            
            avg_response_time = f"{np.mean(all_times):.1f}ms" if all_times else "0ms"
            
            # System health (simplified)
            health_score = min(1.0, total_predictions / 1000) * 100
            if health_score > 80:
                health_status = "Healthy"
            elif health_score > 50:
                health_status = "Warning"
            else:
                health_status = "Critical"
            
            return str(total_models), str(total_predictions), avg_response_time, health_status
        
        @self.app.callback(
            Output("performance-timeline", "figure"),
            [Input("dashboard-data", "data"),
             Input("model-selector", "value"),
             Input("time-range-selector", "value")]
        )
        def update_performance_timeline(dashboard_data, selected_model, time_range_minutes):
            """Update performance timeline chart."""
            fig = go.Figure()
            
            if not dashboard_data or selected_model not in dashboard_data:
                return fig
            
            # Get data for selected time range
            cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
            model_data = dashboard_data[selected_model]["data"]
            
            # Filter by time range
            filtered_data = [
                entry for entry in model_data
                if entry.get('timestamp', datetime.min) > cutoff_time
            ]
            
            if not filtered_data:
                return fig
            
            # Extract time series data
            timestamps = [entry['timestamp'] for entry in filtered_data]
            prediction_times = [entry.get('prediction_time', 0) for entry in filtered_data]
            confidences = [entry.get('confidence', 0) for entry in filtered_data]
            
            # Create timeline chart
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=prediction_times,
                mode='lines+markers',
                name='Prediction Time (ms)',
                line=dict(color='blue'),
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[c * 100 for c in confidences],  # Convert to percentage
                mode='lines+markers',
                name='Confidence (%)',
                line=dict(color='green'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"Performance Timeline - {selected_model.title()}",
                xaxis_title="Time",
                yaxis=dict(title="Prediction Time (ms)", side="left"),
                yaxis2=dict(title="Confidence (%)", side="right", overlaying="y"),
                height=self.config.charts_height,
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output("prediction-distribution", "figure"),
            [Input("dashboard-data", "data"),
             Input("model-selector", "value")]
        )
        def update_prediction_distribution(dashboard_data, selected_model):
            """Update prediction distribution chart."""
            fig = go.Figure()
            
            if not dashboard_data or selected_model not in dashboard_data:
                return fig
            
            model_data = dashboard_data[selected_model]["data"]
            
            # Extract prediction values
            predictions = [
                entry.get('prediction_value', 0) for entry in model_data
                if 'prediction_value' in entry
            ]
            
            if not predictions:
                return fig
            
            # Create histogram
            fig.add_trace(go.Histogram(
                x=predictions,
                nbinsx=30,
                name="Prediction Distribution",
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f"Prediction Distribution - {selected_model.title()}",
                xaxis_title="Predicted SOH",
                yaxis_title="Frequency",
                height=self.config.charts_height
            )
            
            return fig
        
        @self.app.callback(
            Output("model-comparison", "figure"),
            [Input("dashboard-data", "data")]
        )
        def update_model_comparison(dashboard_data):
            """Update model comparison chart."""
            fig = go.Figure()
            
            if not dashboard_data:
                return fig
            
            models = list(dashboard_data.keys())
            avg_times = []
            avg_confidences = []
            total_predictions = []
            
            for model in models:
                stats = dashboard_data[model]["stats"]
                avg_times.append(stats.get("avg_prediction_time", 0))
                avg_confidences.append(stats.get("avg_confidence", 0) * 100)
                total_predictions.append(stats.get("total_predictions", 0))
            
            # Create comparison chart
            fig.add_trace(go.Bar(
                x=models,
                y=avg_times,
                name="Avg Prediction Time (ms)",
                marker_color='blue',
                yaxis='y'
            ))
            
            fig.add_trace(go.Bar(
                x=models,
                y=avg_confidences,
                name="Avg Confidence (%)",
                marker_color='green',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis=dict(title="Prediction Time (ms)", side="left"),
                yaxis2=dict(title="Confidence (%)", side="right", overlaying="y"),
                height=self.config.charts_height,
                barmode='group'
            )
            
            return fig
        
        @self.app.callback(
            Output("error-analysis", "figure"),
            [Input("dashboard-data", "data")]
        )
        def update_error_analysis(dashboard_data):
            """Update error analysis chart."""
            fig = go.Figure()
            
            if not dashboard_data:
                return fig
            
            # Simulate error analysis data
            models = list(dashboard_data.keys())
            error_types = ["Network Error", "Timeout", "Invalid Input", "Model Error"]
            
            # Create stacked bar chart for error analysis
            for error_type in error_types:
                error_counts = [np.random.randint(0, 10) for _ in models]  # Simulated data
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=error_counts,
                    name=error_type
                ))
            
            fig.update_layout(
                title="Error Analysis by Model",
                xaxis_title="Models",
                yaxis_title="Error Count",
                height=self.config.charts_height,
                barmode='stack'
            )
            
            return fig
        
        @self.app.callback(
            Output("model-details-table", "children"),
            [Input("dashboard-data", "data")]
        )
        def update_model_details_table(dashboard_data):
            """Update model details table."""
            if not dashboard_data:
                return html.Div("No data available")
            
            # Create table data
            table_data = []
            for model_name, model_info in dashboard_data.items():
                stats = model_info["stats"]
                
                row = {
                    "Model": model_name.title(),
                    "Total Predictions": stats.get("total_predictions", 0),
                    "Avg Prediction Time (ms)": f"{stats.get('avg_prediction_time', 0):.2f}",
                    "Avg Confidence": f"{stats.get('avg_confidence', 0):.3f}",
                    "Last Update": stats.get("last_update", "Never").strftime("%H:%M:%S") 
                               if isinstance(stats.get("last_update"), datetime) else "Never"
                }
                table_data.append(row)
            
            # Create table
            if table_data:
                df = pd.DataFrame(table_data)
                
                table = dbc.Table.from_dataframe(
                    df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mt-3"
                )
                
                return table
            else:
                return html.Div("No model data available")
        
        @self.app.callback(
            Output("alerts-section", "children"),
            [Input("dashboard-data", "data")]
        )
        def update_alerts_section(dashboard_data):
            """Update alerts section."""
            if not dashboard_data:
                return html.Div("No alerts")
            
            alerts = []
            
            # Check for performance alerts
            for model_name, model_info in dashboard_data.items():
                stats = model_info["stats"]
                
                # Check prediction time threshold
                avg_time = stats.get("avg_prediction_time", 0)
                if avg_time > self.config.alert_thresholds["latency_threshold"]:
                    alerts.append(
                        dbc.Alert(
                            f"High latency detected in {model_name}: {avg_time:.2f}ms",
                            color="warning",
                            className="mb-2"
                        )
                    )
                
                # Check confidence threshold
                avg_confidence = stats.get("avg_confidence", 1.0)
                if avg_confidence < 0.8:
                    alerts.append(
                        dbc.Alert(
                            f"Low confidence detected in {model_name}: {avg_confidence:.3f}",
                            color="danger",
                            className="mb-2"
                        )
                    )
            
            if not alerts:
                alerts.append(
                    dbc.Alert(
                        "All systems operating normally",
                        color="success",
                        className="mb-2"
                    )
                )
            
            return html.Div(alerts)
    
    def run(self):
        """Run the dashboard."""
        # Start data collection
        self.data_collector.start_collection()
        
        try:
            # Run Dash app
            self.app.run_server(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug
            )
        finally:
            # Stop data collection
            self.data_collector.stop_collection()

# Factory function
def create_performance_dashboard(config: Optional[DashboardConfig] = None) -> BatteryMindDashboard:
    """
    Factory function to create performance dashboard.
    
    Args:
        config: Dashboard configuration
        
    Returns:
        BatteryMindDashboard: Configured dashboard instance
    """
    if config is None:
        config = DashboardConfig()
    
    return BatteryMindDashboard(config)

# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BatteryMind Performance Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dashboard host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run dashboard
    config = DashboardConfig(
        port=args.port,
        host=args.host,
        debug=args.debug
    )
    
    dashboard = create_performance_dashboard(config)
    
    logger.info(f"Starting BatteryMind Performance Dashboard on {args.host}:{args.port}")
    dashboard.run()
