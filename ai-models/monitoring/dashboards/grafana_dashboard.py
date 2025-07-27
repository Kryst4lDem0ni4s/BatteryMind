"""
BatteryMind - Grafana Dashboard Manager
Advanced Grafana dashboard integration for battery management AI models with
automated dashboard generation, real-time metrics, and intelligent alerting.

Features:
- Automated dashboard creation and management
- Real-time battery health and performance visualization
- Custom panel generation with business logic
- Alert rule management and notification integration
- Template-based dashboard provisioning
- Multi-tenancy and access control
- Performance optimization and caching

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import requests
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from urllib.parse import urljoin
import threading
from collections import defaultdict

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class PanelType(Enum):
    """Grafana panel types."""
    GRAPH = "graph"
    TIMESERIES = "timeseries"
    STAT = "stat"
    GAUGE = "gauge"
    BAR_GAUGE = "bargauge"
    TABLE = "table"
    PIE_CHART = "piechart"
    HEATMAP = "heatmap"
    ALERT_LIST = "alertlist"
    DASHBOARD_LIST = "dashlist"
    TEXT = "text"
    NEWS = "news"
    PLUGIN_LIST = "pluginlist"
    GEOMAP = "geomap"
    STATE_TIMELINE = "state-timeline"
    STATUS_HISTORY = "status-history"

class DataSourceType(Enum):
    """Grafana data source types."""
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"
    CLOUDWATCH = "cloudwatch"
    ELASTICSEARCH = "elasticsearch"
    MYSQL = "mysql"
    POSTGRES = "postgres"
    GRAPHITE = "graphite"
    LOKI = "loki"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"

class VisualizationType(Enum):
    """Visualization types for panels."""
    LINE = "line"
    AREA = "area"
    BARS = "bars"
    POINTS = "points"
    LINES_AND_POINTS = "lines_and_points"

class AlertConditionType(Enum):
    """Alert condition types."""
    QUERY = "query"
    CLASSIC_CONDITION = "classic_condition"

class AlertExecutionErrorState(Enum):
    """Alert execution error states."""
    ALERTING = "alerting"
    KEEP_STATE = "keep_state"

@dataclass
class GrafanaQuery:
    """Grafana query configuration."""
    
    expr: str
    refId: str = "A"
    legendFormat: str = ""
    interval: str = ""
    step: int = 15
    datasource: Optional[str] = None
    hide: bool = False
    instant: bool = False
    
    # Advanced query settings
    exemplar: bool = True
    format: str = "time_series"  # time_series, table, heatmap
    intervalFactor: int = 1
    maxDataPoints: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary format."""
        return {
            "datasource": self.datasource,
            "expr": self.expr,
            "format": self.format,
            "hide": self.hide,
            "instant": self.instant,
            "interval": self.interval,
            "intervalFactor": self.intervalFactor,
            "legendFormat": self.legendFormat,
            "maxDataPoints": self.maxDataPoints,
            "refId": self.refId,
            "step": self.step,
            "exemplar": self.exemplar
        }

@dataclass
class GrafanaThreshold:
    """Grafana threshold configuration."""
    
    value: float
    color: str
    fill: bool = True
    fillColor: Optional[str] = None
    op: str = "gt"  # gt, lt, within_range, outside_range
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert threshold to dictionary format."""
        threshold = {
            "value": self.value,
            "color": self.color,
            "fill": self.fill,
            "op": self.op
        }
        if self.fillColor:
            threshold["fillColor"] = self.fillColor
        return threshold

@dataclass
class GrafanaFieldOverride:
    """Grafana field override configuration."""
    
    matcher: Dict[str, Any]
    properties: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert field override to dictionary format."""
        return {
            "matcher": self.matcher,
            "properties": self.properties
        }

@dataclass
class GrafanaPanel:
    """Grafana panel configuration."""
    
    # Basic panel settings
    id: int
    title: str
    type: PanelType
    targets: List[GrafanaQuery] = field(default_factory=list)
    
    # Grid position
    gridPos: Dict[str, int] = field(default_factory=lambda: {"h": 8, "w": 12, "x": 0, "y": 0})
    
    # Panel options
    description: str = ""
    transparent: bool = False
    datasource: Optional[str] = None
    
    # Field configuration
    fieldConfig: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Alert configuration
    alert: Optional[Dict[str, Any]] = None
    
    # Visualization settings
    thresholds: List[GrafanaThreshold] = field(default_factory=list)
    overrides: List[GrafanaFieldOverride] = field(default_factory=list)
    
    # Time settings
    timeFrom: Optional[str] = None
    timeShift: Optional[str] = None
    
    # Display settings
    legend: Dict[str, Any] = field(default_factory=lambda: {
        "displayMode": "list",
        "placement": "bottom",
        "values": []
    })
    
    def add_target(self, query: GrafanaQuery) -> None:
        """Add a query target to the panel."""
        self.targets.append(query)
    
    def add_threshold(self, threshold: GrafanaThreshold) -> None:
        """Add a threshold to the panel."""
        self.thresholds.append(threshold)
    
    def add_override(self, override: GrafanaFieldOverride) -> None:
        """Add a field override to the panel."""
        self.overrides.append(override)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert panel to dictionary format."""
        panel_dict = {
            "id": self.id,
            "title": self.title,
            "type": self.type.value,
            "targets": [target.to_dict() for target in self.targets],
            "gridPos": self.gridPos,
            "description": self.description,
            "transparent": self.transparent,
            "fieldConfig": self.fieldConfig,
            "options": self.options,
            "legend": self.legend
        }
        
        # Add optional fields
        if self.datasource:
            panel_dict["datasource"] = self.datasource
        
        if self.alert:
            panel_dict["alert"] = self.alert
        
        if self.timeFrom:
            panel_dict["timeFrom"] = self.timeFrom
        
        if self.timeShift:
            panel_dict["timeShift"] = self.timeShift
        
        # Add thresholds to field config
        if self.thresholds:
            if "defaults" not in panel_dict["fieldConfig"]:
                panel_dict["fieldConfig"]["defaults"] = {}
            panel_dict["fieldConfig"]["defaults"]["thresholds"] = {
                "steps": [threshold.to_dict() for threshold in self.thresholds]
            }
        
        # Add overrides to field config
        if self.overrides:
            panel_dict["fieldConfig"]["overrides"] = [override.to_dict() for override in self.overrides]
        
        return panel_dict

@dataclass
class GrafanaAlert:
    """Grafana alert configuration."""
    
    name: str
    message: str
    frequency: str = "10s"
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    executionErrorState: AlertExecutionErrorState = AlertExecutionErrorState.ALERTING
    noDataState: str = "no_data"
    for_: str = "5m"
    
    # Notification settings
    notifications: List[str] = field(default_factory=list)
    alertRuleTags: Dict[str, str] = field(default_factory=dict)
    
    def add_condition(self, query_ref: str, reducer_type: str = "last", 
                     operator_type: str = "gt", threshold: float = 0.0) -> None:
        """Add an alert condition."""
        condition = {
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "model": {
                "conditions": [
                    {
                        "evaluator": {
                            "params": [threshold],
                            "type": operator_type
                        },
                        "operator": {"type": "and"},
                        "query": {"params": [query_ref]},
                        "reducer": {
                            "params": [],
                            "type": reducer_type
                        },
                        "type": "query"
                    }
                ],
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "expression": "",
                "hide": False,
                "intervalMs": 1000,
                "maxDataPoints": 43200,
                "reducer": reducer_type,
                "refId": query_ref
            },
            "queryType": "",
            "refId": query_ref
        }
        self.conditions.append(condition)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            "name": self.name,
            "message": self.message,
            "frequency": self.frequency,
            "conditions": self.conditions,
            "executionErrorState": self.executionErrorState.value,
            "noDataState": self.noDataState,
            "for": self.for_,
            "notifications": self.notifications,
            "alertRuleTags": self.alertRuleTags
        }

@dataclass
class GrafanaDashboard:
    """Grafana dashboard configuration."""
    
    # Basic dashboard settings
    id: Optional[int] = None
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "BatteryMind Dashboard"
    description: str = ""
    tags: List[str] = field(default_factory=lambda: ["batterymind"])
    
    # Time settings
    time: Dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    timepicker: Dict[str, Any] = field(default_factory=dict)
    timezone: str = "browser"
    
    # Refresh settings
    refresh: str = "30s"
    schemaVersion: int = 30
    version: int = 1
    
    # Dashboard structure
    panels: List[GrafanaPanel] = field(default_factory=list)
    templating: Dict[str, Any] = field(default_factory=lambda: {"list": []})
    annotations: Dict[str, Any] = field(default_factory=lambda: {"list": []})
    
    # Display settings
    editable: bool = True
    fiscalYearStartMonth: int = 0
    graphTooltip: int = 0
    hideControls: bool = False
    
    # Access control
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def add_panel(self, panel: GrafanaPanel) -> None:
        """Add a panel to the dashboard."""
        self.panels.append(panel)
    
    def add_variable(self, name: str, query: str, datasource: str = "prometheus") -> None:
        """Add a template variable to the dashboard."""
        variable = {
            "name": name,
            "type": "query",
            "query": query,
            "datasource": datasource,
            "refresh": 1,
            "includeAll": True,
            "multi": True,
            "allValue": None,
            "current": {"value": "$__all", "text": "All"},
            "options": [],
            "hide": 0,
            "label": name.title(),
            "sort": 1
        }
        self.templating["list"].append(variable)
    
    def add_annotation(self, name: str, datasource: str, query: str) -> None:
        """Add an annotation to the dashboard."""
        annotation = {
            "builtIn": 0,
            "datasource": datasource,
            "enable": True,
            "hide": False,
            "iconColor": "rgba(0, 211, 255, 1)",
            "name": name,
            "query": query,
            "type": "dashboard"
        }
        self.annotations["list"].append(annotation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to dictionary format."""
        dashboard_dict = {
            "id": self.id,
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "timezone": self.timezone,
            "editable": self.editable,
            "fiscalYearStartMonth": self.fiscalYearStartMonth,
            "graphTooltip": self.graphTooltip,
            "hideControls": self.hideControls,
            "refresh": self.refresh,
            "schemaVersion": self.schemaVersion,
            "version": self.version,
            "time": self.time,
            "timepicker": self.timepicker,
            "templating": self.templating,
            "annotations": self.annotations,
            "panels": [panel.to_dict() for panel in self.panels]
        }
        
        if self.meta:
            dashboard_dict["meta"] = self.meta
        
        return dashboard_dict

class GrafanaDashboardManager:
    """
    Manager for Grafana dashboard operations.
    """
    
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Dashboard cache
        self.dashboard_cache: Dict[str, GrafanaDashboard] = {}
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'dashboards_created': 0,
            'dashboards_updated': 0,
            'dashboards_deleted': 0,
            'api_calls': 0,
            'api_errors': 0
        }
        
        logger.info(f"Grafana Dashboard Manager initialized for {base_url}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated request to Grafana API."""
        url = urljoin(f"{self.base_url}/", f"api/{endpoint}")
        
        try:
            self.stats['api_calls'] += 1
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            self.stats['api_errors'] += 1
            logger.error(f"Grafana API request failed: {e}")
            raise
    
    def create_dashboard(self, dashboard: GrafanaDashboard, 
                        folder_id: Optional[int] = None,
                        overwrite: bool = False) -> Dict[str, Any]:
        """
        Create a new dashboard in Grafana.
        
        Args:
            dashboard: Dashboard configuration
            folder_id: ID of folder to create dashboard in
            overwrite: Whether to overwrite existing dashboard
            
        Returns:
            API response with dashboard information
        """
        try:
            payload = {
                "dashboard": dashboard.to_dict(),
                "overwrite": overwrite,
                "message": f"Created dashboard: {dashboard.title}"
            }
            
            if folder_id is not None:
                payload["folderId"] = folder_id
            
            response = self._make_request('POST', 'dashboards/db', json=payload)
            result = response.json()
            
            # Update dashboard with returned ID and UID
            dashboard.id = result.get('id')
            dashboard.uid = result.get('uid', dashboard.uid)
            
            # Cache the dashboard
            with self.cache_lock:
                self.dashboard_cache[dashboard.uid] = dashboard
            
            self.stats['dashboards_created'] += 1
            logger.info(f"Created dashboard: {dashboard.title} (UID: {dashboard.uid})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            raise
    
    def update_dashboard(self, dashboard: GrafanaDashboard) -> Dict[str, Any]:
        """
        Update an existing dashboard.
        
        Args:
            dashboard: Dashboard configuration
            
        Returns:
            API response with dashboard information
        """
        try:
            # Increment version
            dashboard.version += 1
            
            payload = {
                "dashboard": dashboard.to_dict(),
                "overwrite": True,
                "message": f"Updated dashboard: {dashboard.title}"
            }
            
            response = self._make_request('POST', 'dashboards/db', json=payload)
            result = response.json()
            
            # Update cache
            with self.cache_lock:
                self.dashboard_cache[dashboard.uid] = dashboard
            
            self.stats['dashboards_updated'] += 1
            logger.info(f"Updated dashboard: {dashboard.title} (UID: {dashboard.uid})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")
            raise
    
    def get_dashboard(self, uid: str) -> Optional[GrafanaDashboard]:
        """
        Get dashboard by UID.
        
        Args:
            uid: Dashboard UID
            
        Returns:
            Dashboard configuration or None if not found
        """
        try:
            # Check cache first
            with self.cache_lock:
                if uid in self.dashboard_cache:
                    return self.dashboard_cache[uid]
            
            response = self._make_request('GET', f'dashboards/uid/{uid}')
            result = response.json()
            
            if 'dashboard' in result:
                dashboard_data = result['dashboard']
                dashboard = self._parse_dashboard_data(dashboard_data)
                
                # Update cache
                with self.cache_lock:
                    self.dashboard_cache[uid] = dashboard
                
                return dashboard
            
            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Dashboard not found: {uid}")
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get dashboard: {e}")
            raise
    
    def delete_dashboard(self, uid: str) -> bool:
        """
        Delete dashboard by UID.
        
        Args:
            uid: Dashboard UID
            
        Returns:
            Success status
        """
        try:
            self._make_request('DELETE', f'dashboards/uid/{uid}')
            
            # Remove from cache
            with self.cache_lock:
                self.dashboard_cache.pop(uid, None)
            
            self.stats['dashboards_deleted'] += 1
            logger.info(f"Deleted dashboard: {uid}")
            
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Dashboard not found for deletion: {uid}")
                return False
            raise
        except Exception as e:
            logger.error(f"Failed to delete dashboard: {e}")
            return False
    
    def list_dashboards(self, query: str = "", tag: str = "", 
                       starred: bool = False, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        List dashboards with optional filtering.
        
        Args:
            query: Search query
            tag: Filter by tag
            starred: Filter starred dashboards
            limit: Maximum number of results
            
        Returns:
            List of dashboard metadata
        """
        try:
            params = {
                'query': query,
                'limit': limit
            }
            
            if tag:
                params['tag'] = tag
            
            if starred:
                params['starred'] = 'true'
            
            response = self._make_request('GET', 'search', params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            return []
    
    def create_folder(self, title: str, uid: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a folder for organizing dashboards.
        
        Args:
            title: Folder title
            uid: Optional folder UID
            
        Returns:
            Folder information
        """
        try:
            payload = {
                "title": title
            }
            
            if uid:
                payload["uid"] = uid
            
            response = self._make_request('POST', 'folders', json=payload)
            result = response.json()
            
            logger.info(f"Created folder: {title} (UID: {result.get('uid')})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create folder: {e}")
            raise
    
    def _parse_dashboard_data(self, dashboard_data: Dict[str, Any]) -> GrafanaDashboard:
        """Parse dashboard data from Grafana API response."""
        try:
            dashboard = GrafanaDashboard(
                id=dashboard_data.get('id'),
                uid=dashboard_data.get('uid'),
                title=dashboard_data.get('title', ''),
                description=dashboard_data.get('description', ''),
                tags=dashboard_data.get('tags', []),
                time=dashboard_data.get('time', {}),
                timezone=dashboard_data.get('timezone', 'browser'),
                refresh=dashboard_data.get('refresh', '30s'),
                schemaVersion=dashboard_data.get('schemaVersion', 30),
                version=dashboard_data.get('version', 1),
                templating=dashboard_data.get('templating', {'list': []}),
                annotations=dashboard_data.get('annotations', {'list': []}),
                editable=dashboard_data.get('editable', True),
                fiscalYearStartMonth=dashboard_data.get('fiscalYearStartMonth', 0),
                graphTooltip=dashboard_data.get('graphTooltip', 0),
                hideControls=dashboard_data.get('hideControls', False)
            )
            
            # Parse panels
            for panel_data in dashboard_data.get('panels', []):
                panel = self._parse_panel_data(panel_data)
                dashboard.add_panel(panel)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to parse dashboard data: {e}")
            raise
    
    def _parse_panel_data(self, panel_data: Dict[str, Any]) -> GrafanaPanel:
        """Parse panel data from dashboard configuration."""
        try:
            panel_type = PanelType(panel_data.get('type', 'timeseries'))
            
            panel = GrafanaPanel(
                id=panel_data.get('id', 0),
                title=panel_data.get('title', ''),
                type=panel_type,
                gridPos=panel_data.get('gridPos', {}),
                description=panel_data.get('description', ''),
                transparent=panel_data.get('transparent', False),
                datasource=panel_data.get('datasource'),
                fieldConfig=panel_data.get('fieldConfig', {}),
                options=panel_data.get('options', {}),
                legend=panel_data.get('legend', {}),
                timeFrom=panel_data.get('timeFrom'),
                timeShift=panel_data.get('timeShift'),
                alert=panel_data.get('alert')
            )
            
            # Parse targets
            for target_data in panel_data.get('targets', []):
                query = GrafanaQuery(
                    expr=target_data.get('expr', ''),
                    refId=target_data.get('refId', 'A'),
                    legendFormat=target_data.get('legendFormat', ''),
                    interval=target_data.get('interval', ''),
                    step=target_data.get('step', 15),
                    datasource=target_data.get('datasource'),
                    hide=target_data.get('hide', False),
                    instant=target_data.get('instant', False),
                    exemplar=target_data.get('exemplar', True),
                    format=target_data.get('format', 'time_series')
                )
                panel.add_target(query)
            
            return panel
            
        except Exception as e:
            logger.error(f"Failed to parse panel data: {e}")
            raise
    
    def create_battery_health_dashboard(self) -> GrafanaDashboard:
        """Create a comprehensive battery health monitoring dashboard."""
        dashboard = GrafanaDashboard(
            title="BatteryMind - Battery Health Monitoring",
            description="Real-time battery health and performance monitoring",
            tags=["batterymind", "battery", "health", "monitoring"]
        )
        
        # Add template variables
        dashboard.add_variable("battery_id", 'label_values(batterymind_battery_soh, battery_id)')
        dashboard.add_variable("fleet_id", 'label_values(batterymind_battery_soh, fleet_id)')
        
        # Battery SoH Overview Panel
        soh_panel = GrafanaPanel(
            id=1,
            title="Battery State of Health (SoH)",
            type=PanelType.STAT,
            gridPos={"h": 8, "w": 6, "x": 0, "y": 0}
        )
        
        soh_query = GrafanaQuery(
            expr='batterymind_battery_soh{battery_id=~"$battery_id"}',
            legendFormat="{{battery_id}}"
        )
        soh_panel.add_target(soh_query)
        
        # Add thresholds
        soh_panel.add_threshold(GrafanaThreshold(value=0, color="red"))
        soh_panel.add_threshold(GrafanaThreshold(value=70, color="yellow"))
        soh_panel.add_threshold(GrafanaThreshold(value=85, color="green"))
        
        # Configure field options
        soh_panel.fieldConfig = {
            "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 100,
                "custom": {
                    "displayMode": "gradient",
                    "orientation": "horizontal"
                }
            }
        }
        
        dashboard.add_panel(soh_panel)
        
        # Battery Temperature Panel
        temp_panel = GrafanaPanel(
            id=2,
            title="Battery Temperature",
            type=PanelType.TIMESERIES,
            gridPos={"h": 8, "w": 6, "x": 6, "y": 0}
        )
        
        temp_query = GrafanaQuery(
            expr='batterymind_battery_temperature{battery_id=~"$battery_id"}',
            legendFormat="{{battery_id}}"
        )
        temp_panel.add_target(temp_query)
        
        temp_panel.fieldConfig = {
            "defaults": {
                "unit": "celsius",
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "smooth",
                    "fillOpacity": 0.1
                }
            }
        }
        
        dashboard.add_panel(temp_panel)
        
        # Charging Current Panel
        current_panel = GrafanaPanel(
            id=3,
            title="Charging/Discharging Current",
            type=PanelType.TIMESERIES,
            gridPos={"h": 8, "w": 12, "x": 0, "y": 8}
        )
        
        current_query = GrafanaQuery(
            expr='batterymind_battery_current{battery_id=~"$battery_id"}',
            legendFormat="{{battery_id}}"
        )
        current_panel.add_target(current_query)
        
        current_panel.fieldConfig = {
            "defaults": {
                "unit": "amp",
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "linear"
                }
            }
        }
        
        dashboard.add_panel(current_panel)
        
        # Voltage Panel
        voltage_panel = GrafanaPanel(
            id=4,
            title="Battery Voltage",
            type=PanelType.TIMESERIES,
            gridPos={"h": 8, "w": 12, "x": 0, "y": 16}
        )
        
        voltage_query = GrafanaQuery(
            expr='batterymind_battery_voltage{battery_id=~"$battery_id"}',
            legendFormat="{{battery_id}}"
        )
        voltage_panel.add_target(voltage_query)
        
        voltage_panel.fieldConfig = {
            "defaults": {
                "unit": "volt",
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "smooth"
                }
            }
        }
        
        dashboard.add_panel(voltage_panel)
        
        return dashboard
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard manager statistics."""
        with self.cache_lock:
            cache_size = len(self.dashboard_cache)
        
        return {
            **self.stats,
            'cached_dashboards': cache_size,
            'base_url': self.base_url
        }
    
    def clear_cache(self) -> None:
        """Clear dashboard cache."""
        with self.cache_lock:
            self.dashboard_cache.clear()
        
        logger.info("Dashboard cache cleared")

# Factory function for creating battery-specific dashboards
def create_battery_monitoring_dashboard(dashboard_manager: GrafanaDashboardManager) -> GrafanaDashboard:
    """Create and deploy a battery monitoring dashboard."""
    dashboard = dashboard_manager.create_battery_health_dashboard()
    result = dashboard_manager.create_dashboard(dashboard, overwrite=True)
    
    logger.info(f"Battery monitoring dashboard created: {result.get('url')}")
    return dashboard
