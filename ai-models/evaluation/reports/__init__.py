"""
BatteryMind - Evaluation Reports Module

Comprehensive reporting system for battery AI/ML model evaluation, providing
automated report generation, visualization, and analysis capabilities for
model performance assessment and business impact analysis.

This module provides:
- Automated evaluation report generation
- Performance dashboard creation
- Model comparison analysis
- Business impact assessment
- Interactive visualizations
- Export capabilities for various formats

Features:
- Multi-format report generation (HTML, PDF, JSON)
- Interactive performance dashboards
- Statistical analysis and significance testing
- Business KPI tracking and ROI analysis
- Automated report scheduling and distribution
- Integration with monitoring and alerting systems

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Visualization and reporting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Report generation
from jinja2 import Template, Environment, FileSystemLoader
import weasyprint
import base64
from io import BytesIO

# Internal imports
from .evaluation_report import EvaluationReportGenerator
from .performance_dashboard import PerformanceDashboard
from .model_comparison import ModelComparisonAnalyzer
from .business_impact import BusinessImpactAnalyzer
from ...utils.logging_utils import setup_logging
from ...utils.visualization import create_styled_plot

# Configure logging
logger = setup_logging(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module exports
__all__ = [
    # Report generators
    "EvaluationReportGenerator",
    "PerformanceDashboard",
    "ModelComparisonAnalyzer",
    "BusinessImpactAnalyzer",
    
    # Configuration classes
    "ReportConfig",
    "DashboardConfig",
    "ComparisonConfig",
    "BusinessAnalysisConfig",
    
    # Report types
    "EvaluationReport",
    "PerformanceReport",
    "ComparisonReport",
    "BusinessReport",
    
    # Factory functions
    "create_evaluation_report",
    "create_performance_dashboard",
    "create_model_comparison",
    "create_business_analysis",
    
    # Utility functions
    "generate_report_template",
    "export_report",
    "schedule_report_generation",
    "validate_report_data"
]

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Report metadata
    report_title: str = "BatteryMind Model Evaluation Report"
    report_author: str = "BatteryMind AI System"
    report_version: str = "1.0.0"
    
    # Output settings
    output_format: str = "html"  # html, pdf, json, markdown
    output_path: str = "./reports"
    include_timestamp: bool = True
    
    # Content configuration
    include_executive_summary: bool = True
    include_detailed_metrics: bool = True
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_appendix: bool = True
    
    # Visualization settings
    plot_style: str = "seaborn"
    color_palette: str = "husl"
    figure_dpi: int = 300
    figure_format: str = "png"
    
    # Analysis settings
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    include_statistical_tests: bool = True
    
    # Business metrics
    include_business_impact: bool = True
    currency_symbol: str = "$"
    time_horizon_months: int = 12

@dataclass
class EvaluationReport:
    """Structure for evaluation reports."""
    
    # Report metadata
    report_id: str
    title: str
    created_at: datetime
    report_type: str = "evaluation"
    
    # Model information
    model_name: str = ""
    model_version: str = ""
    model_type: str = ""
    
    # Performance metrics
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    business_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Analysis results
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Visualizations
    plots: Dict[str, str] = field(default_factory=dict)  # plot_name -> base64_image
    
    # Raw data
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: str = "draft"  # draft, completed, published
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ReportManager:
    """Central manager for all report generation and management."""
    
    def __init__(self):
        self.report_generators = {}
        self.active_reports = {}
        self.report_history = []
        self.templates = {}
        self.config = ReportConfig()
        
        # Initialize report generators
        self._initialize_generators()
        
        # Load templates
        self._load_report_templates()
        
        logger.info("ReportManager initialized")
    
    def _initialize_generators(self):
        """Initialize all report generators."""
        self.report_generators = {
            'evaluation': EvaluationReportGenerator(),
            'performance': PerformanceDashboard(),
            'comparison': ModelComparisonAnalyzer(),
            'business': BusinessImpactAnalyzer()
        }
    
    def _load_report_templates(self):
        """Load report templates."""
        template_dir = Path(__file__).parent / "templates"
        
        if template_dir.exists():
            env = Environment(loader=FileSystemLoader(str(template_dir)))
            
            for template_file in template_dir.glob("*.html"):
                template_name = template_file.stem
                self.templates[template_name] = env.get_template(template_file.name)
        
        # Create default templates if none exist
        if not self.templates:
            self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default report templates."""
        # Basic HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #2E86AB; color: white; padding: 20px; text-align: center; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #2E86AB; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; }
                .plot { text-align: center; margin: 20px 0; }
                .recommendations { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.title }}</h1>
                <p>Generated: {{ report.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            {% if report.accuracy_metrics %}
            <div class="section">
                <h2>Performance Metrics</h2>
                {% for metric, value in report.accuracy_metrics.items() %}
                <div class="metric">{{ metric }}: {{ "%.4f"|format(value) }}</div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if report.plots %}
            <div class="section">
                <h2>Visualizations</h2>
                {% for plot_name, plot_data in report.plots.items() %}
                <div class="plot">
                    <h3>{{ plot_name }}</h3>
                    <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if report.recommendations %}
            <div class="section recommendations">
                <h2>Recommendations</h2>
                <ul>
                {% for recommendation in report.recommendations %}
                    <li>{{ recommendation }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        self.templates['default'] = Template(html_template)
    
    def generate_report(self, report_type: str, data: Dict[str, Any], 
                       config: ReportConfig = None) -> EvaluationReport:
        """
        Generate a report of the specified type.
        
        Args:
            report_type: Type of report to generate
            data: Data for report generation
            config: Report configuration
            
        Returns:
            Generated evaluation report
        """
        if report_type not in self.report_generators:
            raise ValueError(f"Unknown report type: {report_type}")
        
        report_config = config or self.config
        generator = self.report_generators[report_type]
        
        # Generate report
        report = generator.generate_report(data, report_config)
        
        # Store report
        self.active_reports[report.report_id] = report
        self.report_history.append({
            'report_id': report.report_id,
            'report_type': report_type,
            'created_at': report.created_at,
            'status': report.status
        })
        
        logger.info(f"Report generated: {report.report_id}")
        
        return report
    
    def export_report(self, report_id: str, format_type: str = None) -> str:
        """
        Export report to specified format.
        
        Args:
            report_id: ID of report to export
            format_type: Export format (html, pdf, json)
            
        Returns:
            Path to exported file
        """
        if report_id not in self.active_reports:
            raise ValueError(f"Report {report_id} not found")
        
        report = self.active_reports[report_id]
        export_format = format_type or self.config.output_format
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.config.include_timestamp else ""
        filename = f"report_{report_id}_{timestamp}.{export_format}"
        filepath = Path(self.config.output_path) / filename
        
        # Create output directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if export_format == "html":
            self._export_html(report, filepath)
        elif export_format == "pdf":
            self._export_pdf(report, filepath)
        elif export_format == "json":
            self._export_json(report, filepath)
        elif export_format == "markdown":
            self._export_markdown(report, filepath)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Report exported to {filepath}")
        return str(filepath)
    
    def _export_html(self, report: EvaluationReport, filepath: Path):
        """Export report as HTML."""
        template = self.templates.get('default', self.templates[list(self.templates.keys())[0]])
        html_content = template.render(report=report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_pdf(self, report: EvaluationReport, filepath: Path):
        """Export report as PDF."""
        # First generate HTML
        html_filepath = filepath.with_suffix('.html')
        self._export_html(report, html_filepath)
        
        # Convert HTML to PDF
        try:
            weasyprint.HTML(filename=str(html_filepath)).write_pdf(str(filepath))
            # Clean up temporary HTML file
            html_filepath.unlink()
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise
    
    def _export_json(self, report: EvaluationReport, filepath: Path):
        """Export report as JSON."""
        report_dict = {
            'report_id': report.report_id,
            'title': report.title,
            'created_at': report.created_at.isoformat(),
            'report_type': report.report_type,
            'model_name': report.model_name,
            'model_version': report.model_version,
            'model_type': report.model_type,
            'accuracy_metrics': report.accuracy_metrics,
            'performance_metrics': report.performance_metrics,
            'efficiency_metrics': report.efficiency_metrics,
            'business_metrics': report.business_metrics,
            'baseline_comparison': report.baseline_comparison,
            'statistical_analysis': report.statistical_analysis,
            'recommendations': report.recommendations,
            'status': report.status,
            'errors': report.errors,
            'warnings': report.warnings
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
    
    def _export_markdown(self, report: EvaluationReport, filepath: Path):
        """Export report as Markdown."""
        md_content = f"""# {report.title}

**Generated:** {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {report.model_name} v{report.model_version}
**Type:** {report.model_type}

## Performance Metrics

"""
        
        # Add accuracy metrics
        if report.accuracy_metrics:
            md_content += "### Accuracy Metrics\n\n"
            for metric, value in report.accuracy_metrics.items():
                md_content += f"- **{metric}:** {value:.4f}\n"
            md_content += "\n"
        
        # Add performance metrics
        if report.performance_metrics:
            md_content += "### Performance Metrics\n\n"
            for metric, value in report.performance_metrics.items():
                md_content += f"- **{metric}:** {value:.4f}\n"
            md_content += "\n"
        
        # Add recommendations
        if report.recommendations:
            md_content += "## Recommendations\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                md_content += f"{i}. {rec}\n"
            md_content += "\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def get_report(self, report_id: str) -> Optional[EvaluationReport]:
        """Get report by ID."""
        return self.active_reports.get(report_id)
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all available reports."""
        return self.report_history.copy()
    
    def delete_report(self, report_id: str) -> bool:
        """Delete a report."""
        if report_id in self.active_reports:
            del self.active_reports[report_id]
            # Remove from history
            self.report_history = [
                item for item in self.report_history 
                if item['report_id'] != report_id
            ]
            logger.info(f"Report {report_id} deleted")
            return True
        return False

# Global report manager instance
_report_manager = None

def get_report_manager() -> ReportManager:
    """Get the global report manager instance."""
    global _report_manager
    if _report_manager is None:
        _report_manager = ReportManager()
    return _report_manager

def create_evaluation_report(data: Dict[str, Any], config: ReportConfig = None) -> EvaluationReport:
    """Create an evaluation report."""
    manager = get_report_manager()
    return manager.generate_report('evaluation', data, config)

def create_performance_dashboard(data: Dict[str, Any], config: ReportConfig = None) -> EvaluationReport:
    """Create a performance dashboard report."""
    manager = get_report_manager()
    return manager.generate_report('performance', data, config)

def create_model_comparison(data: Dict[str, Any], config: ReportConfig = None) -> EvaluationReport:
    """Create a model comparison report."""
    manager = get_report_manager()
    return manager.generate_report('comparison', data, config)

def create_business_analysis(data: Dict[str, Any], config: ReportConfig = None) -> EvaluationReport:
    """Create a business impact analysis report."""
    manager = get_report_manager()
    return manager.generate_report('business', data, config)

def export_report(report_id: str, format_type: str = "html") -> str:
    """Export a report to specified format."""
    manager = get_report_manager()
    return manager.export_report(report_id, format_type)

def generate_report_template(template_name: str, template_content: str):
    """Register a custom report template."""
    manager = get_report_manager()
    manager.templates[template_name] = Template(template_content)

def validate_report_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate data for report generation."""
    errors = []
    
    # Check for required fields
    required_fields = ['model_results', 'evaluation_metrics']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate metrics structure
    if 'evaluation_metrics' in data:
        metrics = data['evaluation_metrics']
        if not isinstance(metrics, dict):
            errors.append("evaluation_metrics must be a dictionary")
        else:
            # Check for numeric values
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"Metric {key} must be numeric, got {type(value)}")
    
    return len(errors) == 0, errors

def schedule_report_generation(report_type: str, data_source: str, 
                             schedule_config: Dict[str, Any]) -> str:
    """Schedule automatic report generation."""
    # This would integrate with a job scheduler like Celery or APScheduler
    # For now, return a placeholder job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    logger.info(f"Scheduled {report_type} report generation with ID: {job_id}")
    return job_id

# Module initialization
logger.info(f"BatteryMind Evaluation Reports module v{__version__} initialized")
logger.info(f"Available report types: evaluation, performance, comparison, business")
