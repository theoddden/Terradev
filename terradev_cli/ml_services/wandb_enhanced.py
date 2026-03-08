#!/usr/bin/env python3
"""
Enhanced Weights & Biases Service with Deep Dashboard Integration
Integrates W&B dashboards, reports, and Terradev monitoring
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import base64


@dataclass
class WAndBEnhancedConfig:
    """Enhanced W&B configuration"""
    api_key: str
    entity: Optional[str] = None
    project: Optional[str] = None
    base_url: Optional[str] = None
    team: Optional[str] = None
    dashboard_enabled: bool = False
    reports_enabled: bool = False
    alerts_enabled: bool = False
    integration_enabled: bool = False


class EnhancedWAndBService:
    """Enhanced W&B service with deep dashboard integration"""
    
    def __init__(self, config: WAndBEnhancedConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_base = config.base_url or "https://api.wandb.ai"
        self.app_base = config.base_url or "https://wandb.ai"
        
    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive W&B dashboard"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.app_base}/api/graphql"
            
            # GraphQL mutation for creating dashboard
            mutation = """
            mutation CreateDashboard($input: CreateDashboardInput!) {
                createDashboard(input: $input) {
                    dashboard {
                        id
                        name
                        description
                        createdAt
                        updatedAt
                    }
                }
            }
            """
            
            variables = {
                "input": {
                    "name": dashboard_config.get("name", "Terradev Dashboard"),
                    "description": dashboard_config.get("description", "Dashboard created by Terradev"),
                    "entityName": self.config.entity or "default",
                    "projectName": self.config.project or "terradev",
                    "panels": dashboard_config.get("panels", [])
                }
            }
            
            payload = {
                "query": mutation,
                "variables": variables
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "created",
                        "dashboard": data.get("data", {}).get("createDashboard", {}).get("dashboard")
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create dashboard: {response.status} - {error_text}")
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_terradev_dashboard(self) -> Dict[str, Any]:
        """Create a Terradev-specific dashboard with ML infrastructure metrics"""
        try:
            dashboard_config = {
                "name": "Terradev Infrastructure Dashboard",
                "description": "Comprehensive dashboard for Terradev ML infrastructure monitoring",
                "panels": [
                    {
                        "name": "GPU Utilization",
                        "type": "line",
                        "query": "avg(gpu_utilization)",
                        "yAxisLabel": "GPU %",
                        "xAxisLabel": "Time"
                    },
                    {
                        "name": "Cost per Hour",
                        "type": "line",
                        "query": "sum(cost_per_hour)",
                        "yAxisLabel": "Cost ($/hr)",
                        "xAxisLabel": "Time"
                    },
                    {
                        "name": "Training Loss",
                        "type": "line",
                        "query": "avg(loss)",
                        "yAxisLabel": "Loss",
                        "xAxisLabel": "Step"
                    },
                    {
                        "name": "Model Accuracy",
                        "type": "line",
                        "query": "avg(accuracy)",
                        "yAxisLabel": "Accuracy",
                        "xAxisLabel": "Step"
                    },
                    {
                        "name": "Instance Count",
                        "type": "bar",
                        "query": "count(instance_id)",
                        "yAxisLabel": "Count",
                        "xAxisLabel": "Provider"
                    },
                    {
                        "name": "Region Distribution",
                        "type": "pie",
                        "query": "count(region)",
                        "yAxisLabel": "Count",
                        "xAxisLabel": "Region"
                    }
                ]
            }
            
            return await self.create_dashboard(dashboard_config)
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_report(self, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive W&B report"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.app_base}/api/graphql"
            
            # GraphQL mutation for creating report
            mutation = """
            mutation CreateReport($input: CreateReportInput!) {
                createReport(input: $input) {
                    report {
                        id
                        name
                        description
                        createdAt
                        updatedAt
                    }
                }
            }
            """
            
            variables = {
                "input": {
                    "name": report_config.get("name", "Terradev Report"),
                    "description": report_config.get("description", "Report created by Terradev"),
                    "entityName": self.config.entity or "default",
                    "projectName": self.config.project or "terradev",
                    "content": report_config.get("content", ""),
                    "visibility": report_config.get("visibility", "TEAM")
                }
            }
            
            payload = {
                "query": mutation,
                "variables": variables
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "created",
                        "report": data.get("data", {}).get("createReport", {}).get("report")
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create report: {response.status} - {error_text}")
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_terradev_report(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Terradev-specific report with infrastructure metrics"""
        try:
            # Generate HTML content for the report
            html_content = self._generate_terradev_report_html(metrics_data)
            
            report_config = {
                "name": f"Terradev Infrastructure Report - {datetime.now().strftime('%Y-%m-%d')}",
                "description": "Comprehensive report on Terradev ML infrastructure performance and costs",
                "content": html_content,
                "visibility": "TEAM"
            }
            
            return await self.create_report(report_config)
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_terradev_report_html(self, metrics_data: Dict[str, Any]) -> str:
        """Generate HTML content for Terradev report"""
        html = f"""
        <html>
        <head>
            <title>Terradev Infrastructure Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Terradev Infrastructure Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Infrastructure Overview</h2>
            <div class="metric">
                <strong>Total Instances:</strong> {metrics_data.get('total_instances', 0)}
            </div>
            <div class="metric">
                <strong>Total Cost:</strong> ${metrics_data.get('total_cost', 0):.2f}
            </div>
            <div class="metric">
                <strong>Average GPU Utilization:</strong> {metrics_data.get('avg_gpu_utilization', 0):.1f}%
            </div>
            
            <h2>Provider Breakdown</h2>
            <table>
                <tr>
                    <th>Provider</th>
                    <th>Instances</th>
                    <th>Cost</th>
                    <th>Avg GPU Util</th>
                </tr>
        """
        
        # Add provider breakdown
        for provider, data in metrics_data.get('providers', {}).items():
            html += f"""
                <tr>
                    <td>{provider}</td>
                    <td>{data.get('instances', 0)}</td>
                    <td>${data.get('cost', 0):.2f}</td>
                    <td>{data.get('avg_gpu_util', 0):.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Performance Metrics</h2>
            <div class="chart">
                <p>Performance charts and detailed metrics would be embedded here</p>
            </div>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Consider optimizing GPU utilization for better cost efficiency</li>
                <li>Review instance types for optimal performance</li>
                <li>Monitor regional cost variations</li>
            </ul>
        </body>
        </html>
        """
        
        return html
    
    async def setup_alerts(self, alert_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up W&B alerts for Terradev metrics"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.app_base}/api/graphql"
            
            # GraphQL mutation for creating alert
            mutation = """
            mutation CreateAlert($input: CreateAlertInput!) {
                createAlert(input: $input) {
                    alert {
                        id
                        name
                        description
                        createdAt
                        updatedAt
                    }
                }
            }
            """
            
            variables = {
                "input": {
                    "name": alert_config.get("name", "Terradev Alert"),
                    "description": alert_config.get("description", "Alert created by Terradev"),
                    "entityName": self.config.entity or "default",
                    "projectName": self.config.project or "terradev",
                    "query": alert_config.get("query", ""),
                    "trigger": alert_config.get("trigger", {}),
                    "actions": alert_config.get("actions", [])
                }
            }
            
            payload = {
                "query": mutation,
                "variables": variables
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "created",
                        "alert": data.get("data", {}).get("createAlert", {}).get("alert")
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create alert: {response.status} - {error_text}")
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_terradev_alerts(self) -> Dict[str, Any]:
        """Create standard Terradev alerts"""
        try:
            alerts = [
                {
                    "name": "High GPU Cost Alert",
                    "description": "Alert when GPU cost exceeds threshold",
                    "query": "avg(cost_per_hour) > 5.0",
                    "trigger": {
                        "type": "threshold",
                        "value": 5.0,
                        "operator": "greater_than"
                    },
                    "actions": [
                        {
                            "type": "email",
                            "config": {
                                "to": ["admin@example.com"],
                                "subject": "High GPU Cost Alert"
                            }
                        }
                    ]
                },
                {
                    "name": "Low GPU Utilization Alert",
                    "description": "Alert when GPU utilization is low",
                    "query": "avg(gpu_utilization) < 30",
                    "trigger": {
                        "type": "threshold",
                        "value": 30,
                        "operator": "less_than"
                    },
                    "actions": [
                        {
                            "type": "email",
                            "config": {
                                "to": ["admin@example.com"],
                                "subject": "Low GPU Utilization Alert"
                            }
                        }
                    ]
                }
            ]
            
            results = []
            for alert_config in alerts:
                result = await self.setup_alerts(alert_config)
                results.append(result)
            
            return {
                "status": "completed",
                "alerts": results
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_dashboard_status(self) -> Dict[str, Any]:
        """Get comprehensive dashboard and monitoring status"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Get projects
            projects = await self.list_projects()
            
            # Get recent runs
            runs = await self.list_runs(limit=50)
            
            # Get dashboards (if available)
            dashboards = await self._get_dashboards()
            
            # Get reports (if available)
            reports = await self._get_reports()
            
            return {
                "status": "connected",
                "entity": self.config.entity or "default",
                "project": self.config.project or "terradev",
                "projects": projects,
                "recent_runs": runs[:10],
                "dashboards": dashboards,
                "reports": reports,
                "monitoring": {
                    "dashboard_enabled": self.config.dashboard_enabled,
                    "reports_enabled": self.config.reports_enabled,
                    "alerts_enabled": self.config.alerts_enabled,
                    "integration_enabled": self.config.integration_enabled
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _get_dashboards(self) -> List[Dict[str, Any]]:
        """Get dashboards (placeholder - W&B API may not expose this directly)"""
        # This would need to be implemented based on actual W&B API capabilities
        return []
    
    async def _get_reports(self) -> List[Dict[str, Any]]:
        """Get reports (placeholder - W&B API may not expose this directly)"""
        # This would need to be implemented based on actual W&B API capabilities
        return []
    
    async def list_projects(self) -> List[Dict[str, Any]]:
        """List all W&B projects"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            entity = self.config.entity or "me"
            url = f"{self.api_base}/v1/entities/{entity}/projects"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("projects", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list projects: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to list projects: {e}")
    
    async def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List runs in a project"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            entity = self.config.entity or "me"
            project = self.config.project or "terradev"
            url = f"{self.api_base}/v1/entities/{entity}/projects/{project}/runs"
            params = {"limit": limit}
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("runs", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list runs: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to list runs: {e}")
    
    def get_enhanced_config(self) -> Dict[str, str]:
        """Get enhanced W&B configuration for environment variables"""
        config = self.get_wandb_config()
        
        # Add enhanced configuration
        if self.config.dashboard_enabled:
            config["WANDB_DASHBOARD_ENABLED"] = "true"
        
        if self.config.reports_enabled:
            config["WANDB_REPORTS_ENABLED"] = "true"
        
        if self.config.alerts_enabled:
            config["WANDB_ALERTS_ENABLED"] = "true"
        
        if self.config.integration_enabled:
            config["WANDB_INTEGRATION_ENABLED"] = "true"
        
        return config
    
    def get_wandb_config(self) -> Dict[str, str]:
        """Get W&B configuration for environment variables"""
        config = {
            "WANDB_API_KEY": self.config.api_key
        }
        
        if self.config.entity:
            config["WANDB_ENTITY"] = self.config.entity
            
        if self.config.project:
            config["WANDB_PROJECT"] = self.config.project
        else:
            config["WANDB_PROJECT"] = "terradev"
            
        if self.config.base_url:
            config["WANDB_BASE_URL"] = self.config.base_url
            
        if self.config.team:
            config["WANDB_TEAM"] = self.config.team
            
        return config
    
    def generate_integration_script(self) -> str:
        """Generate W&B integration script for Terradev"""
        script_lines = [
            "# W&B Integration Script (generated by Terradev)",
            "",
            "# Set up W&B environment variables",
            f"export WANDB_API_KEY='{self.config.api_key}'",
            f"export WANDB_ENTITY='{self.config.entity or 'default'}'",
            f"export WANDB_PROJECT='{self.config.project or 'terradev'}'",
            "",
            "# Enhanced W&B features",
            "export WANDB_DASHBOARD_ENABLED=true",
            "export WANDB_REPORTS_ENABLED=true",
            "export WANDB_ALERTS_ENABLED=true",
            "export WANDB_INTEGRATION_ENABLED=true",
            "",
            "# Test W&B connection",
            "python -c \"import wandb; wandb.login(); print('W&B configured successfully')\"",
            "",
            "# Example usage in training script with Terradev integration:",
            "import wandb",
            "import terradev_cli.ml_services.wandb_enhanced as wandb_enhanced",
            "",
            "# Initialize W&B with Terradev metadata",
            "wandb.init(project='terradev')",
            "wandb.config.update({",
            "    'gpu_type': 'A100',",
            "    'provider': 'aws',",
            "    'cost_per_hour': 4.50,",
            "    'region': 'us-east-1'",
            "    'instance_id': 'i-1234567890'",
            "})",
            "",
            "# Log metrics",
            "wandb.log({",
            "    'accuracy': 0.95,",
            "    'loss': 0.05,",
            "    'gpu_utilization': 85.2,",
            "    'cost_per_hour': 4.50",
            "})",
            "",
            "# Create Terradev dashboard",
            "service = wandb_enhanced.create_enhanced_wandb_service_from_credentials({",
            "    'api_key': os.environ.get('WANDB_API_KEY'),",
            "    'entity': os.environ.get('WANDB_ENTITY'),",
            "    'project': os.environ.get('WANDB_PROJECT'),",
            "    'dashboard_enabled': True",
            "    'reports_enabled': True",
            "    'alerts_enabled': True",
            "})",
            "",
            "# Create dashboard and reports",
            "await service.create_terradev_dashboard()",
            "await service.create_terradev_report(metrics_data)",
            "await service.create_terradev_alerts()",
            "",
            "print('W&B integration complete! Check your dashboard at: https://wandb.ai/' + os.environ.get('WANDB_ENTITY', 'default') + '/' + os.environ.get('WANDB_PROJECT', 'terradev'))"
        ]
        
        return "\n".join(script_lines)


def create_enhanced_wandb_service_from_credentials(credentials: Dict[str, str]) -> EnhancedWAndBService:
    """Create EnhancedWAndBService from credential dictionary"""
    config = WAndBEnhancedConfig(
        api_key=credentials["api_key"],
        entity=credentials.get("entity"),
        project=credentials.get("project"),
        base_url=credentials.get("base_url"),
        team=credentials.get("team"),
        dashboard_enabled=credentials.get("dashboard_enabled", "false").lower() == "true",
        reports_enabled=credentials.get("reports_enabled", "false").lower() == "true",
        alerts_enabled=credentials.get("alerts_enabled", "false").lower() == "true",
        integration_enabled=credentials.get("integration_enabled", "false").lower() == "true"
    )
    
    return EnhancedWAndBService(config)


def get_enhanced_wandb_setup_instructions() -> str:
    """Get enhanced setup instructions for W&B with dashboard integration"""
    return """
🚀 Enhanced W&B Setup Instructions:

1. Create a W&B account:
   - Go to https://wandb.ai
   - Sign up for a free account

2. Create an API key:
   - Navigate to https://wandb.ai/settings
   - Click "Create API Key"
   - Copy the API key

3. Configure Terradev with enhanced W&B:
   terradev configure --provider wandb \\
     --api-key YOUR_KEY \\
     --entity your-entity \\
     --project terradev \\
     --dashboard-enabled true \\
     --reports-enabled true \\
     --alerts-enabled true \\
     --integration-enabled true

4. Create Terradev dashboard:
   terradev ml wandb --create-dashboard

5. Generate infrastructure report:
   terradev ml wandb --create-report

6. Set up alerts:
   terradev ml wandb --setup-alerts

7. Get comprehensive status:
   terradev ml wandb --dashboard-status

📋 Enhanced Credentials:
- api_key: W&B API key (required)
- entity: W&B entity (team/username, optional)
- project: Default project name (optional, default: "terradev")
- base_url: W&B server URL (optional, for self-hosted)
- team: W&B team name (optional)
- dashboard_enabled: Enable dashboard features (default: "false")
- reports_enabled: Enable report generation (default: "false")
- alerts_enabled: Enable alerting (default: "false")
- integration_enabled: Enable deep integration (default: "false")

💡 Enhanced Usage Examples:
# Test connection
terradev ml wandb --test

# Create Terradev dashboard
terradev ml wandb --create-dashboard

# Generate infrastructure report
terradev ml wandb --create-report

# Set up alerts
terradev ml wandb --setup-alerts

# Get dashboard status
terradev ml wandb --dashboard-status

# List projects and runs
terradev ml wandb --list-projects
terradev ml wandb --list-runs

# Export runs data
terradev ml wandb --export json > runs.json

🔗 Dashboard Integration:
- **Terradev Dashboard**: Pre-configured infrastructure dashboard
- **Custom Panels**: GPU utilization, cost tracking, performance metrics
- **Real-time Updates**: Live metrics from Terradev provisioned instances
- **Historical Analysis**: Track trends over time
- **Comparative Analysis**: Compare providers and regions

🎯 Dashboard Features:
- **GPU Utilization**: Real-time GPU usage across all instances
- **Cost Tracking**: Per-hour and total cost monitoring
- **Performance Metrics**: Training loss, accuracy, and other ML metrics
- **Provider Comparison**: Compare performance across cloud providers
- **Regional Analysis**: Cost and performance by region
- **Alert Integration**: Automatic alerts for cost and performance issues

📊 Report Generation:
- **Infrastructure Reports**: Comprehensive HTML reports
- **Performance Analysis**: Detailed performance breakdowns
- **Cost Analysis**: Cost optimization recommendations
- **Trend Analysis**: Historical performance trends
- **Executive Summaries**: High-level overview for stakeholders

🚨 Alert System:
- **Cost Alerts**: Notify when costs exceed thresholds
- **Performance Alerts**: Alert on performance degradation
- **Utilization Alerts**: Low GPU utilization warnings
- **Integration Alerts**: System integration health checks
- **Custom Alerts**: User-defined alert conditions

🔧 Integration with Terradev:
- **Automatic Metrics**: Terradev automatically logs infrastructure metrics
- **Cost Tracking**: Real-time cost monitoring per instance
- **Performance Monitoring**: GPU and system performance metrics
- **Provider Comparison**: Compare performance across providers
- **Regional Optimization**: Optimize based on regional performance

📊 Dashboard URLs:
- Main Dashboard: https://wandb.ai/your-entity/terradev
- Reports: https://wandb.ai/your-entity/terradev/reports
- Alerts: https://wandb.ai/your-entity/terradev/alerts

🎯 Advanced Features:
- **Custom Dashboards**: Create custom dashboards for specific needs
- **Team Collaboration**: Share dashboards with team members
- **API Integration**: Programmatic dashboard creation and management
- **Export Options**: Export data and reports in multiple formats
- **Integration Hooks**: Webhooks and API integrations

📝 Example Integration:
```python
import wandb
import terradev_cli.ml_services.wandb_enhanced as wandb_enhanced

# Initialize with Terradev metadata
wandb.init(project='terradev')

# Log infrastructure metrics
wandb.log({
    'gpu_utilization': 85.2,
    'cost_per_hour': 4.50,
    'instance_id': 'i-1234567890',
    'provider': 'aws',
    'region': 'us-east-1'
})

# Create enhanced dashboard
service = wandb_enhanced.create_enhanced_wandb_service_from_credentials(creds)
await service.create_terradev_dashboard()
```
"""
