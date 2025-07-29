"""
BatteryMind - Visualization Utilities
Advanced visualization and plotting utilities for battery management AI systems
with comprehensive charting, dashboard generation, and interactive visualization capabilities.

Features:
- Real-time battery health visualization
- Fleet performance dashboards
- AI model interpretability charts
- Blockchain transaction visualization
- Circular economy flow diagrams
- Interactive data exploration tools
- Autonomous decision visualization

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import base64
from io import BytesIO
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.subplots as sp
    from plotly.offline import plot
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Additional visualization libraries
try:
    import bokeh.plotting as bp
    from bokeh.models import HoverTool, ColumnDataSource
    from bokeh.layouts import column, row
    from bokeh.io import output_file, show, save
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

class ChartType(Enum):
    """Types of charts available in BatteryMind."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    SUNBURST = "sunburst"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    RADAR = "radar"
    WATERFALL = "waterfall"
    CANDLESTICK = "candlestick"
    FUNNEL = "funnel"
    TIMELINE = "timeline"
    NETWORK = "network"
    GEOSPATIAL = "geospatial"
    THREE_D = "3d"

class ColorScheme(Enum):
    """Color schemes for BatteryMind visualizations."""
    BATTERYMIND = "batterymind"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    BLUES = "blues"
    GREENS = "greens"
    REDS = "reds"
    ORANGES = "oranges"
    PURPLES = "purples"
    SPECTRAL = "spectral"
    COOLWARM = "coolwarm"
    BATTERY_HEALTH = "battery_health"
    FLEET_STATUS = "fleet_status"
    AI_INSIGHTS = "ai_insights"
    BLOCKCHAIN = "blockchain"
    CIRCULAR_ECONOMY = "circular_economy"

@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    
    # Basic properties
    width: int = 800
    height: int = 600
    dpi: int = 300
    title: str = ""
    subtitle: str = ""
    
    # Color and styling
    color_scheme: ColorScheme = ColorScheme.BATTERYMIND
    background_color: str = "#ffffff"
    grid_color: str = "#e0e0e0"
    text_color: str = "#333333"
    
    # Font configuration
    title_font_size: int = 16
    subtitle_font_size: int = 12
    axis_font_size: int = 10
    legend_font_size: int = 9
    font_family: str = "Arial"
    
    # Interactive features
    interactive: bool = True
    hover_info: bool = True
    zoom_enabled: bool = True
    pan_enabled: bool = True
    
    # Export options
    export_formats: List[str] = field(default_factory=lambda: ['png', 'html', 'svg'])
    export_quality: str = "high"
    
    # Animation
    animation_enabled: bool = False
    animation_duration: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                data[key] = value.value
            else:
                data[key] = value
        return data

class BatteryVisualizer:
    """
    Specialized visualizer for battery data and health metrics.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.color_schemes = self._define_color_schemes()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self._get_color_palette())
        
        logger.info("Battery Visualizer initialized")
    
    def _define_color_schemes(self) -> Dict[str, List[str]]:
        """Define custom color schemes for BatteryMind."""
        return {
            'batterymind': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'battery_health': ['#00ff00', '#7fff00', '#ffff00', '#ff7f00', '#ff0000'],
            'fleet_status': ['#4CAF50', '#FFC107', '#FF5722', '#9C27B0', '#2196F3'],
            'ai_insights': ['#3F51B5', '#E91E63', '#00BCD4', '#4CAF50', '#FF9800'],
            'blockchain': ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA'],
            'circular_economy': ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800']
        }
    
    def _get_color_palette(self) -> List[str]:
        """Get color palette based on current scheme."""
        scheme_name = self.config.color_scheme.value
        if scheme_name in self.color_schemes:
            return self.color_schemes[scheme_name]
        else:
            return sns.color_palette(scheme_name).as_hex()
    
    def plot_battery_health_timeline(self, 
                                   df: pd.DataFrame,
                                   battery_id: str = None,
                                   save_path: str = None) -> Union[plt.Figure, str]:
        """
        Plot battery health timeline showing SoH, SoC, and other metrics.
        
        Args:
            df: Battery data DataFrame
            battery_id: Optional battery ID to filter
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure or HTML string
        """
        try:
            # Filter data if battery_id specified
            if battery_id and 'battery_id' in df.columns:
                df = df[df['battery_id'] == battery_id]
            
            if df.empty:
                logger.warning("No data available for visualization")
                return None
            
            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            if self.config.interactive and PLOTLY_AVAILABLE:
                return self._plot_interactive_battery_timeline(df, battery_id, save_path)
            else:
                return self._plot_static_battery_timeline(df, battery_id, save_path)
                
        except Exception as e:
            logger.error(f"Error plotting battery health timeline: {e}")
            return None
    
    def _plot_interactive_battery_timeline(self, 
                                         df: pd.DataFrame,
                                         battery_id: str = None,
                                         save_path: str = None) -> str:
        """Create interactive battery health timeline using Plotly."""
        try:
            # Create subplots
            fig = sp.make_subplots(
                rows=3, cols=1,
                subplot_titles=['State of Health (SoH)', 'State of Charge (SoC)', 'Temperature & Voltage'],
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            # SoH plot
            if 'soh' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['soh'],
                        mode='lines+markers',
                        name='SoH (%)',
                        line=dict(color=self.color_schemes['battery_health'][1], width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>SoH</b>: %{y:.1f}%<br><b>Time</b>: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # SoC plot
            if 'soc' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['soc'],
                        mode='lines+markers',
                        name='SoC (%)',
                        line=dict(color=self.color_schemes['battery_health'][2], width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>SoC</b>: %{y:.1f}%<br><b>Time</b>: %{x}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Temperature and Voltage
            if 'temperature' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['temperature'],
                        mode='lines',
                        name='Temperature (°C)',
                        line=dict(color=self.color_schemes['battery_health'][3], width=2),
                        yaxis='y3',
                        hovertemplate='<b>Temperature</b>: %{y:.1f}°C<br><b>Time</b>: %{x}<extra></extra>'
                    ),
                    row=3, col=1
                )
            
            if 'voltage' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['voltage'],
                        mode='lines',
                        name='Voltage (V)',
                        line=dict(color=self.color_schemes['battery_health'][4], width=2),
                        yaxis='y4',
                        hovertemplate='<b>Voltage</b>: %{y:.2f}V<br><b>Time</b>: %{x}<extra></extra>'
                    ),
                    row=3, col=1
                )
            
            # Update layout
            title = f"Battery Health Timeline - {battery_id}" if battery_id else "Battery Health Timeline"
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': self.config.title_font_size}
                },
                height=self.config.height,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor=self.config.background_color,
                paper_bgcolor=self.config.background_color,
                font=dict(family=self.config.font_family, color=self.config.text_color)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time", row=3, col=1)
            fig.update_yaxes(title_text="SoH (%)", row=1, col=1, range=[0, 100])
            fig.update_yaxes(title_text="SoC (%)", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
            
            # Save plot
            if save_path:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                elif save_path.endswith('.png'):
                    fig.write_image(save_path, width=self.config.width, height=self.config.height)
                elif save_path.endswith('.svg'):
                    fig.write_image(save_path, format='svg')
            
            return fig.to_html(include_plotlyjs=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive battery timeline: {e}")
            return ""
    
    def _plot_static_battery_timeline(self, 
                                    df: pd.DataFrame,
                                    battery_id: str = None,
                                    save_path: str = None) -> plt.Figure:
        """Create static battery health timeline using Matplotlib."""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(self.config.width/100, self.config.height/100))
            fig.suptitle(f"Battery Health Timeline - {battery_id}" if battery_id else "Battery Health Timeline",
                        fontsize=self.config.title_font_size)
            
            # SoH plot
            if 'soh' in df.columns:
                axes[0].plot(df['timestamp'], df['soh'], 'o-', 
                           color=self.color_schemes['battery_health'][1], linewidth=2, markersize=3)
                axes[0].set_ylabel('SoH (%)')
                axes[0].set_ylim(0, 100)
                axes[0].grid(True, alpha=0.3)
            
            # SoC plot
            if 'soc' in df.columns:
                axes[1].plot(df['timestamp'], df['soc'], 'o-', 
                           color=self.color_schemes['battery_health'][2], linewidth=2, markersize=3)
                axes[1].set_ylabel('SoC (%)')
                axes[1].set_ylim(0, 100)
                axes[1].grid(True, alpha=0.3)
            
            # Temperature and Voltage
            ax_temp = axes[2]
            ax_volt = ax_temp.twinx()
            
            if 'temperature' in df.columns:
                line1 = ax_temp.plot(df['timestamp'], df['temperature'], 'o-', 
                                   color=self.color_schemes['battery_health'][3], 
                                   linewidth=2, markersize=3, label='Temperature (°C)')
                ax_temp.set_ylabel('Temperature (°C)', color=self.color_schemes['battery_health'][3])
                ax_temp.tick_params(axis='y', labelcolor=self.color_schemes['battery_health'][3])
            
            if 'voltage' in df.columns:
                line2 = ax_volt.plot(df['timestamp'], df['voltage'], 's-', 
                                   color=self.color_schemes['battery_health'][4], 
                                   linewidth=2, markersize=3, label='Voltage (V)')
                ax_volt.set_ylabel('Voltage (V)', color=self.color_schemes['battery_health'][4])
                ax_volt.tick_params(axis='y', labelcolor=self.color_schemes['battery_health'][4])
            
            axes[2].set_xlabel('Time')
            axes[2].grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in axes:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating static battery timeline: {e}")
            return None
    
    def plot_battery_health_distribution(self, 
                                       df: pd.DataFrame,
                                       metric: str = 'soh',
                                       save_path: str = None) -> Union[plt.Figure, str]:
        """
        Plot distribution of battery health metrics across fleet.
        
        Args:
            df: Battery data DataFrame
            metric: Metric to plot distribution for
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure or HTML string
        """
        try:
            if metric not in df.columns:
                logger.error(f"Metric '{metric}' not found in data")
                return None
            
            if self.config.interactive and PLOTLY_AVAILABLE:
                return self._plot_interactive_distribution(df, metric, save_path)
            else:
                return self._plot_static_distribution(df, metric, save_path)
                
        except Exception as e:
            logger.error(f"Error plotting battery health distribution: {e}")
            return None
    
    def _plot_interactive_distribution(self, 
                                     df: pd.DataFrame,
                                     metric: str,
                                     save_path: str = None) -> str:
        """Create interactive distribution plot using Plotly."""
        try:
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    f'{metric.upper()} Distribution',
                    f'{metric.upper()} Box Plot',
                    f'{metric.upper()} by Battery Type',
                    f'{metric.upper()} Statistics'
                ],
                specs=[[{"type": "histogram"}, {"type": "box"}],
                       [{"type": "violin"}, {"type": "table"}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=df[metric],
                    nbinsx=30,
                    name=f'{metric.upper()} Distribution',
                    marker_color=self.color_schemes['battery_health'][1],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=df[metric],
                    name=f'{metric.upper()}',
                    marker_color=self.color_schemes['battery_health'][2],
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
            
            # Violin plot by battery type (if available)
            if 'battery_type' in df.columns:
                for i, battery_type in enumerate(df['battery_type'].unique()):
                    subset = df[df['battery_type'] == battery_type]
                    fig.add_trace(
                        go.Violin(
                            y=subset[metric],
                            name=battery_type,
                            marker_color=self.color_schemes['battery_health'][i % len(self.color_schemes['battery_health'])],
                            box_visible=True,
                            meanline_visible=True
                        ),
                        row=2, col=1
                    )
            
            # Statistics table
            stats = df[metric].describe()
            fig.add_trace(
                go.Table(
                    header=dict(values=['Statistic', 'Value']),
                    cells=dict(values=[
                        ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                        [f'{stats["count"]:.0f}', f'{stats["mean"]:.2f}', f'{stats["std"]:.2f}',
                         f'{stats["min"]:.2f}', f'{stats["25%"]:.2f}', f'{stats["50%"]:.2f}',
                         f'{stats["75%"]:.2f}', f'{stats["max"]:.2f}']
                    ])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{metric.upper()} Distribution Analysis',
                height=self.config.height,
                showlegend=True
            )
            
            if save_path:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                elif save_path.endswith('.png'):
                    fig.write_image(save_path)
            
            return fig.to_html(include_plotlyjs=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive distribution plot: {e}")
            return ""
    
    def _plot_static_distribution(self, 
                                df: pd.DataFrame,
                                metric: str,
                                save_path: str = None) -> plt.Figure:
        """Create static distribution plot using Matplotlib."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(self.config.width/100, self.config.height/100))
            fig.suptitle(f'{metric.upper()} Distribution Analysis', fontsize=self.config.title_font_size)
            
            # Histogram
            axes[0, 0].hist(df[metric], bins=30, alpha=0.7, 
                          color=self.color_schemes['battery_health'][1])
            axes[0, 0].set_title(f'{metric.upper()} Distribution')
            axes[0, 0].set_xlabel(metric.upper())
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[0, 1].boxplot(df[metric], patch_artist=True,
                             boxprops=dict(facecolor=self.color_schemes['battery_health'][2]))
            axes[0, 1].set_title(f'{metric.upper()} Box Plot')
            axes[0, 1].set_ylabel(metric.upper())
            axes[0, 1].grid(True, alpha=0.3)
            
            # Violin plot by battery type (if available)
            if 'battery_type' in df.columns:
                battery_types = df['battery_type'].unique()
                data_by_type = [df[df['battery_type'] == bt][metric].values for bt in battery_types]
                axes[1, 0].violinplot(data_by_type, positions=range(len(battery_types)))
                axes[1, 0].set_title(f'{metric.upper()} by Battery Type')
                axes[1, 0].set_xlabel('Battery Type')
                axes[1, 0].set_ylabel(metric.upper())
                axes[1, 0].set_xticks(range(len(battery_types)))
                axes[1, 0].set_xticklabels(battery_types, rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # Statistics text
            stats = df[metric].describe()
            stats_text = f"""
            Count: {stats['count']:.0f}
            Mean: {stats['mean']:.2f}
            Std: {stats['std']:.2f}
            Min: {stats['min']:.2f}
            25%: {stats['25%']:.2f}
            50%: {stats['50%']:.2f}
            75%: {stats['75%']:.2f}
            Max: {stats['max']:.2f}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='center')
            axes[1, 1].set_title(f'{metric.upper()} Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating static distribution plot: {e}")
            return None

class FleetVisualizer:
    """
    Specialized visualizer for fleet data and analytics.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.color_schemes = {
            'fleet_status': ['#4CAF50', '#FFC107', '#FF5722', '#9C27B0', '#2196F3'],
            'utilization': ['#E8F5E8', '#A5D6A7', '#66BB6A', '#4CAF50', '#388E3C'],
            'efficiency': ['#FFF3E0', '#FFCC02', '#FF9800', '#F57C00', '#E65100']
        }
        
        logger.info("Fleet Visualizer initialized")
    
    def plot_fleet_overview(self, 
                          fleet_data: Dict[str, pd.DataFrame],
                          save_path: str = None) -> Union[plt.Figure, str]:
        """
        Plot comprehensive fleet overview dashboard.
        
        Args:
            fleet_data: Dictionary containing fleet DataFrames
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure or HTML string
        """
        try:
            if self.config.interactive and PLOTLY_AVAILABLE:
                return self._plot_interactive_fleet_overview(fleet_data, save_path)
            else:
                return self._plot_static_fleet_overview(fleet_data, save_path)
                
        except Exception as e:
            logger.error(f"Error plotting fleet overview: {e}")
            return None
    
    def _plot_interactive_fleet_overview(self, 
                                       fleet_data: Dict[str, pd.DataFrame],
                                       save_path: str = None) -> str:
        """Create interactive fleet overview using Plotly."""
        try:
            fig = sp.make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Fleet Utilization', 'Battery Health Distribution',
                    'Vehicle Status', 'Energy Consumption Trends',
                    'Route Efficiency', 'Cost Analysis'
                ],
                specs=[[{"type": "indicator"}, {"type": "histogram"}],
                       [{"type": "pie"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            vehicles_df = fleet_data.get('vehicles', pd.DataFrame())
            routes_df = fleet_data.get('routes', pd.DataFrame())
            
            if not vehicles_df.empty:
                # Fleet utilization gauge
                utilization = vehicles_df[vehicles_df['status'] == 'active'].shape[0] / len(vehicles_df) * 100
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=utilization,
                        title={'text': "Fleet Utilization (%)"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': self.color_schemes['fleet_status'][0]},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 90}}
                    ),
                    row=1, col=1
                )
                
                # Battery health distribution
                if 'battery_health' in vehicles_df.columns:
                    fig.add_trace(
                        go.Histogram(
                            x=vehicles_df['battery_health'],
                            nbinsx=20,
                            name='Battery Health',
                            marker_color=self.color_schemes['fleet_status'][1]
                        ),
                        row=1, col=2
                    )
                
                # Vehicle status pie chart
                status_counts = vehicles_df['status'].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=status_counts.index,
                        values=status_counts.values,
                        name="Vehicle Status",
                        marker_colors=self.color_schemes['fleet_status']
                    ),
                    row=2, col=1
                )
            
            if not routes_df.empty:
                # Energy consumption trends
                if 'date' in routes_df.columns and 'energy_consumed_kwh' in routes_df.columns:
                    daily_consumption = routes_df.groupby('date')['energy_consumed_kwh'].sum()
                    fig.add_trace(
                        go.Scatter(
                            x=daily_consumption.index,
                            y=daily_consumption.values,
                            mode='lines+markers',
                            name='Daily Energy Consumption',
                            line=dict(color=self.color_schemes['efficiency'][2])
                        ),
                        row=2, col=2
                    )
                
                # Route efficiency
                if 'vehicle_id' in routes_df.columns and 'total_distance_km' in routes_df.columns:
                    efficiency = routes_df.groupby('vehicle_id').agg({
                        'total_distance_km': 'mean',
                        'energy_consumed_kwh': 'mean'
                    })
                    efficiency['efficiency'] = efficiency['total_distance_km'] / efficiency['energy_consumed_kwh']
                    
                    fig.add_trace(
                        go.Bar(
                            x=efficiency.index,
                            y=efficiency['efficiency'],
                            name='Route Efficiency (km/kWh)',
                            marker_color=self.color_schemes['efficiency'][3]
                        ),
                        row=3, col=1
                    )
            
            fig.update_layout(
                title='Fleet Overview Dashboard',
                height=self.config.height * 1.5,
                showlegend=True
            )
            
            if save_path:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                elif save_path.endswith('.png'):
                    fig.write_image(save_path)
            
            return fig.to_html(include_plotlyjs=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive fleet overview: {e}")
            return ""
    
    def _plot_static_fleet_overview(self, 
                                  fleet_data: Dict[str, pd.DataFrame],
                                  save_path: str = None) -> plt.Figure:
        """Create static fleet overview using Matplotlib."""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(self.config.width/100, self.config.height*1.5/100))
            fig.suptitle('Fleet Overview Dashboard', fontsize=self.config.title_font_size)
            
            vehicles_df = fleet_data.get('vehicles', pd.DataFrame())
            routes_df = fleet_data.get('routes', pd.DataFrame())
            
            if not vehicles_df.empty:
                # Fleet utilization
                status_counts = vehicles_df['status'].value_counts()
                utilization = status_counts.get('active', 0) / len(vehicles_df) * 100
                
                # Create utilization gauge (simplified as bar)
                axes[0, 0].bar(['Fleet Utilization'], [utilization], 
                              color=self.color_schemes['fleet_status'][0])
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].set_ylabel('Utilization (%)')
                axes[0, 0].set_title('Fleet Utilization')
                
                # Battery health distribution
                if 'battery_health' in vehicles_df.columns:
                    axes[0, 1].hist(vehicles_df['battery_health'], bins=20, 
                                  color=self.color_schemes['fleet_status'][1], alpha=0.7)
                    axes[0, 1].set_title('Battery Health Distribution')
                    axes[0, 1].set_xlabel('Battery Health (%)')
                    axes[0, 1].set_ylabel('Count')
                
                # Vehicle status pie chart
                axes[1, 0].pie(status_counts.values, labels=status_counts.index, 
                              colors=self.color_schemes['fleet_status'], autopct='%1.1f%%')
                axes[1, 0].set_title('Vehicle Status Distribution')
            
            if not routes_df.empty:
                # Energy consumption trends
                if 'date' in routes_df.columns and 'energy_consumed_kwh' in routes_df.columns:
                    daily_consumption = routes_df.groupby('date')['energy_consumed_kwh'].sum()
                    axes[1, 1].plot(daily_consumption.index, daily_consumption.values, 
                                  'o-', color=self.color_schemes['efficiency'][2])
                    axes[1, 1].set_title('Daily Energy Consumption')
                    axes[1, 1].set_xlabel('Date')
                    axes[1, 1].set_ylabel('Energy (kWh)')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                
                # Route efficiency
                if 'vehicle_id' in routes_df.columns:
                    efficiency = routes_df.groupby('vehicle_id').agg({
                        'total_distance_km': 'mean',
                        'energy_consumed_kwh': 'mean'
                    })
                    efficiency['efficiency'] = efficiency['total_distance_km'] / efficiency['energy_consumed_kwh']
                    
                    axes[2, 0].bar(range(len(efficiency)), efficiency['efficiency'], 
                                 color=self.color_schemes['efficiency'][3])
                    axes[2, 0].set_title('Route Efficiency by Vehicle')
                    axes[2, 0].set_xlabel('Vehicle Index')
                    axes[2, 0].set_ylabel('Efficiency (km/kWh)')
            
            # Remove empty subplot
            if len(axes) > 2:
                fig.delaxes(axes[2, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating static fleet overview: {e}")
            return None

class AIInsightsVisualizer:
    """
    Specialized visualizer for AI model insights and explainability.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        logger.info("AI Insights Visualizer initialized")
    
    def plot_model_performance_comparison(self, 
                                        performance_data: Dict[str, Dict[str, float]],
                                        save_path: str = None) -> Union[plt.Figure, str]:
        """
        Plot model performance comparison across different metrics.
        
        Args:
            performance_data: Dictionary of model performance metrics
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure or HTML string
        """
        try:
            if self.config.interactive and PLOTLY_AVAILABLE:
                return self._plot_interactive_model_comparison(performance_data, save_path)
            else:
                return self._plot_static_model_comparison(performance_data, save_path)
                
        except Exception as e:
            logger.error(f"Error plotting model performance comparison: {e}")
            return None
    
    def _plot_interactive_model_comparison(self, 
                                         performance_data: Dict[str, Dict[str, float]],
                                         save_path: str = None) -> str:
        """Create interactive model comparison using Plotly."""
        try:
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(performance_data).T
            
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Model Accuracy Comparison',
                    'Model Latency Comparison',
                    'Model Performance Radar',
                    'Performance Matrix Heatmap'
                ],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatterpolar"}, {"type": "heatmap"}]]
            )
            
            models = list(performance_data.keys())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Accuracy comparison
            if 'accuracy' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=df['accuracy'],
                        name='Accuracy',
                        marker_color=colors[0]
                    ),
                    row=1, col=1
                )
            
            # Latency comparison
            if 'latency_ms' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=df['latency_ms'],
                        name='Latency (ms)',
                        marker_color=colors[1]
                    ),
                    row=1, col=2
                )
            
            # Radar chart for multi-metric comparison
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            available_metrics = [m for m in metrics if m in df.columns]
            
            for i, model in enumerate(models):
                fig.add_trace(
                    go.Scatterpolar(
                        r=[df.loc[model, metric] for metric in available_metrics],
                        theta=available_metrics,
                        fill='toself',
                        name=model,
                        line_color=colors[i % len(colors)]
                    ),
                    row=2, col=1
                )
            
            # Performance heatmap
            fig.add_trace(
                go.Heatmap(
                    z=df.values,
                    x=df.columns,
                    y=df.index,
                    colorscale='Viridis',
                    name='Performance Matrix'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='AI Model Performance Analysis',
                height=self.config.height * 1.2,
                showlegend=True
            )
            
            if save_path:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                elif save_path.endswith('.png'):
                    fig.write_image(save_path)
            
            return fig.to_html(include_plotlyjs=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive model comparison: {e}")
            return ""
    
    def _plot_static_model_comparison(self, 
                                    performance_data: Dict[str, Dict[str, float]],
                                    save_path: str = None) -> plt.Figure:
        """Create static model comparison using Matplotlib."""
        try:
            df = pd.DataFrame(performance_data).T
            
            fig, axes = plt.subplots(2, 2, figsize=(self.config.width/100, self.config.height/100))
            fig.suptitle('AI Model Performance Analysis', fontsize=self.config.title_font_size)
            
            # Accuracy comparison
            if 'accuracy' in df.columns:
                axes[0, 0].bar(df.index, df['accuracy'], color='skyblue')
                axes[0, 0].set_title('Model Accuracy Comparison')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Latency comparison
            if 'latency_ms' in df.columns:
                axes[0, 1].bar(df.index, df['latency_ms'], color='lightcoral')
                axes[0, 1].set_title('Model Latency Comparison')
                axes[0, 1].set_ylabel('Latency (ms)')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Performance heatmap
            sns.heatmap(df, annot=True, cmap='viridis', ax=axes[1, 0])
            axes[1, 0].set_title('Performance Matrix')
            
            # Multi-metric comparison
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics:
                df_subset = df[available_metrics]
                df_subset.plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Multi-Metric Comparison')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating static model comparison: {e}")
            return None

# Factory functions and utilities
def create_battery_visualizer(config: Optional[VisualizationConfig] = None) -> BatteryVisualizer:
    """Create a battery visualizer instance."""
    return BatteryVisualizer(config)

def create_fleet_visualizer(config: Optional[VisualizationConfig] = None) -> FleetVisualizer:
    """Create a fleet visualizer instance."""
    return FleetVisualizer(config)

def create_ai_insights_visualizer(config: Optional[VisualizationConfig] = None) -> AIInsightsVisualizer:
    """Create an AI insights visualizer instance."""
    return AIInsightsVisualizer(config)

def save_plot_as_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graphic = base64.b64encode(image_png)
        return graphic.decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error converting plot to base64: {e}")
        return ""

def export_plot_multiple_formats(fig: Union[plt.Figure, str], 
                                base_path: str, 
                                formats: List[str] = None):
    """Export plot in multiple formats."""
    try:
        if formats is None:
            formats = ['png', 'svg', 'pdf']
        
        if isinstance(fig, plt.Figure):
            for fmt in formats:
                if fmt in ['png', 'svg', 'pdf', 'eps']:
                    fig.savefig(f"{base_path}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)
        elif isinstance(fig, str) and PLOTLY_AVAILABLE:
            # Handle Plotly HTML string
            for fmt in formats:
                if fmt == 'html':
                    with open(f"{base_path}.html", 'w') as f:
                        f.write(fig)
        
        logger.info(f"Plot exported in formats: {formats}")
        
    except Exception as e:
        logger.error(f"Error exporting plot in multiple formats: {e}")

# Log module initialization
logger.info("BatteryMind Visualization Utils Module v1.0.0 loaded successfully")
