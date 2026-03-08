"""
Terradev CLI Optimization Commands

Command-line interface for optimization features including CUCo integration.
"""

import click
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time

from .optimization.cuco_optimizer import CUCoOptimizer, OptimizationDecision
from .optimization.auto_optimizer import AutoOptimizer
from .core.config import TerradevConfig
from .core.monitoring import MetricsCollector
from .core.optimization_config import get_optimization_config, update_optimization_config

logger = logging.getLogger(__name__)

@click.group()
def optimize():
    """Terradev optimization commands"""
    pass

@optimize.command()
@click.argument('deployment_id')
@click.option('--workload-spec', type=click.Path(exists=True), help='Workload specification file')
@click.option('--auto-apply', is_flag=True, help='Automatically apply optimizations')
@click.option('--force', is_flag=True, help='Force optimization even if confidence is low')
def analyze(deployment_id: str, workload_spec: Optional[str], auto_apply: bool, force: bool):
    """Analyze deployment for optimization opportunities"""
    
    try:
        # Load workload specification
        if workload_spec:
            with open(workload_spec, 'r') as f:
                workload_data = json.load(f)
        else:
            # Generate default workload spec
            workload_data = {
                "deployment_id": deployment_id,
                "type": "llm_training",
                "gpu_count": 4,
                "framework": "pytorch",
                "model_size": 70000000000,
                "batch_size": 32,
                "sequence_length": 2048
            }
        
        # Initialize optimizer
        config = TerradevConfig()
        metrics_collector = MetricsCollector()
        auto_optimizer = AutoOptimizer(config, metrics_collector)
        
        # Analyze deployment
        click.echo(f"🔍 Analyzing deployment {deployment_id}...")
        
        async def run_analysis():
            plan = await auto_optimizer.analyze_deployment(deployment_id, workload_data)
            return plan
        
        plan = asyncio.run(run_analysis())
        
        # Display results
        click.echo(f"\n📊 Optimization Analysis Results:")
        click.echo(f"==============================")
        click.echo(f"Deployment ID: {deployment_id}")
        click.echo(f"Recommended Optimizations: {', '.join(plan.optimizations)}")
        click.echo(f"Expected Performance Gain: {plan.expected_performance_gain:.2f}x")
        click.echo(f"Expected Cost Increase: {plan.expected_cost_increase:.1%}")
        click.echo(f"Confidence Score: {plan.confidence_score:.1%}")
        click.echo(f"Estimated Duration: {plan.estimated_duration:.0f}s")
        click.echo(f"Reasoning: {plan.reasoning}")
        
        # Apply optimizations if requested
        if auto_apply and plan.optimizations:
            if plan.confidence_score >= 0.7 or force:
                click.echo(f"\n🚀 Applying optimizations...")
                
                async def run_optimization():
                    results = await auto_optimizer.apply_optimizations(deployment_id, plan)
                    return results
                
                results = asyncio.run(run_optimization())
                
                click.echo(f"✅ Optimization completed!")
                click.echo(f"Applied: {', '.join(results['applied_optimizations'])}")
                click.echo(f"Failed: {', '.join(results['failed_optimizations'])}")
                click.echo(f"Actual Performance Gain: {results['actual_performance_gain']:.2f}x")
                click.echo(f"Actual Cost Increase: {results['actual_cost_increase']:.1%}")
                click.echo(f"Total Time: {results['total_time']:.0f}s")
                
                if results['errors']:
                    click.echo(f"Errors: {', '.join(results['errors'])}")
            else:
                click.echo(f"⚠️  Confidence too low ({plan.confidence_score:.1%}) for auto-application. Use --force to apply anyway.")
        
    except Exception as e:
        click.echo(f"❌ Error during analysis: {str(e)}")
        raise click.ClickException(str(e))

@optimize.command()
@click.argument('deployment_id')
@click.option('--duration', default=10, help='Benchmark duration in minutes')
@click.option('--output', type=click.Path(), help='Output file for benchmark results')
def benchmark(deployment_id: str, duration: int, output: Optional[str]):
    """Benchmark optimization performance impact"""
    
    try:
        # Initialize optimizer
        config = TerradevConfig()
        metrics_collector = MetricsCollector()
        auto_optimizer = AutoOptimizer(config, metrics_collector)
        
        click.echo(f"🏃 Running benchmark for deployment {deployment_id}...")
        click.echo(f"Duration: {duration} minutes")
        
        # Run benchmark
        results = auto_optimizer.benchmark_optimization_impact(deployment_id, duration)
        
        # Display results
        click.echo(f"\n📈 Benchmark Results:")
        click.echo(f"====================")
        
        if "error" in results:
            click.echo(f"❌ {results['error']}")
            return
        
        performance = results.get("performance_comparison", {})
        for metric, data in performance.items():
            improvement = "✅" if data.get("improvement") else "❌"
            click.echo(f"{improvement} {metric}: {data['baseline']:.2f} → {data['current']:.2f} ({data['change_ratio']:.2f}x)")
        
        p95_compliance = results.get("p95_compliance", {})
        click.echo(f"\n🎯 P95 Compliance:")
        click.echo(f"Compliance Rate: {p95_compliance.get('compliance_rate', 0):.1%}")
        click.echo(f"Violations: {len(p95_compliance.get('violations', []))}")
        
        cost_efficiency = results.get("cost_efficiency", {})
        click.echo(f"\n💰 Cost Efficiency:")
        click.echo(f"Cost per Request: ${cost_efficiency.get('cost_per_request', 0):.6f}")
        click.echo(f"Performance per Dollar: {cost_efficiency.get('performance_per_dollar', 0):.2f}")
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"\n💾 Results saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Error during benchmark: {str(e)}")
        raise click.ClickException(str(e))

@optimize.command()
@click.argument('deployment_id')
@click.option('--type', 'rollback_type', default='all', type=click.Choice(['cuco', 'warm_pool', 'all']))
def rollback(deployment_id: str, rollback_type: str):
    """Rollback optimizations for deployment"""
    
    try:
        # Initialize optimizer
        config = TerradevConfig()
        metrics_collector = MetricsCollector()
        auto_optimizer = AutoOptimizer(config, metrics_collector)
        
        click.echo(f"🔄 Rolling back {rollback_type} optimizations for deployment {deployment_id}...")
        
        # Perform rollback
        results = auto_optimizer.rollback_optimization(deployment_id, rollback_type)
        
        # Display results
        click.echo(f"\n📋 Rollback Results:")
        click.echo(f"===================")
        click.echo(f"Rolled Back: {', '.join(results['rolled_back_optimizations'])}")
        click.echo(f"Failed: {', '.join(results['failed_rollbacks'])}")
        
        if results.get("performance_impact"):
            impact = results["performance_impact"]
            click.echo(f"\n📊 Performance Impact:")
            click.echo(f"Current Latency: {impact.get('current_latency', 0):.2f}ms")
            click.echo(f"Current Throughput: {impact.get('current_throughput', 0):.2f} RPS")
            click.echo(f"Current Cost: ${impact.get('current_cost', 0):.2f}/hour")
        
    except Exception as e:
        click.echo(f"❌ Error during rollback: {str(e)}")
        raise click.ClickException(str(e))

@optimize.command()
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def dashboard(output_json: bool):
    """Show optimization dashboard"""
    
    try:
        # Initialize optimizer
        config = TerradevConfig()
        metrics_collector = MetricsCollector()
        auto_optimizer = AutoOptimizer(config, metrics_collector)
        
        # Get dashboard data
        dashboard_data = auto_optimizer.get_optimization_dashboard()
        
        if output_json:
            click.echo(json.dumps(dashboard_data, indent=2))
        else:
            # Display formatted dashboard
            click.echo(f"📊 Terradev Optimization Dashboard")
            click.echo(f"=================================")
            click.echo(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(dashboard_data['dashboard_timestamp']))}")
            
            # Overall metrics
            overall = dashboard_data.get("overall_metrics", {})
            click.echo(f"\n📈 Overall Metrics:")
            click.echo(f"Total Optimizations: {overall.get('total_optimizations', 0)}")
            click.echo(f"Average Speedup: {overall.get('average_speedup', 1.0):.2f}x")
            click.echo(f"Active Deployments: {overall.get('active_deployments', 0)}")
            click.echo(f"Success Rate: {overall.get('optimization_success_rate', 0):.1%}")
            
            # CUCo summary
            cuco = dashboard_data.get("cuco_summary", {})
            click.echo(f"\n🚀 CUCo Summary:")
            click.echo(f"CUCo Optimizations: {cuco.get('total_optimizations', 0)}")
            click.echo(f"Average Speedup: {cuco.get('average_speedup', 1.0):.2f}x")
            click.echo(f"Max Speedup: {cuco.get('max_speedup', 1.0):.2f}x")
            
            # Active optimizations
            active = dashboard_data.get("active_optimizations", [])
            click.echo(f"\n🔄 Active Optimizations:")
            for opt in active:
                click.echo(f"  • {opt['deployment_id']}: {opt['optimization_type']} ({opt['performance_gain']:.2f}x)")
            
            # Performance trends
            trends = dashboard_data.get("performance_trends", {})
            click.echo(f"\n📊 Performance Trends:")
            click.echo(f"Trend: {trends.get('trend_direction', 'unknown')}")
            click.echo(f"Average Gain: {trends.get('average_gain', 1.0):.2f}x")
            click.echo(f"Confidence: {trends.get('trend_confidence', 0):.1%}")
            
            # Cost savings
            savings = dashboard_data.get("cost_savings", {})
            click.echo(f"\n💰 Cost Savings:")
            click.echo(f"Hourly Savings: ${savings.get('total_savings_per_hour', 0):.2f}")
            click.echo(f"Monthly Savings: ${savings.get('monthly_savings', 0):.2f}")
            click.echo(f"Savings Rate: {savings.get('savings_rate', 0):.1%}")
            
            # Alerts
            alerts = dashboard_data.get("alerts", [])
            if alerts:
                click.echo(f"\n⚠️  Alerts:")
                for alert in alerts:
                    click.echo(f"  • {alert['level'].upper()}: {alert['message']}")
        
    except Exception as e:
        click.echo(f"❌ Error generating dashboard: {str(e)}")
        raise click.ClickException(str(e))

@optimize.command()
@click.argument('deployment_id')
def recommendations(deployment_id: str):
    """Get optimization recommendations for deployment"""
    
    try:
        # Initialize optimizer
        config = TerradevConfig()
        metrics_collector = MetricsCollector()
        auto_optimizer = AutoOptimizer(config, metrics_collector)
        
        click.echo(f"💡 Getting optimization recommendations for deployment {deployment_id}...")
        
        # Get recommendations
        recommendations = auto_optimizer.get_optimization_recommendations(deployment_id)
        
        if "error" in recommendations:
            click.echo(f"❌ {recommendations['error']}")
            return
        
        # Display current performance
        current = recommendations.get("current_performance", {})
        click.echo(f"\n📊 Current Performance:")
        click.echo(f"Latency: {current.get('latency_ms', 0):.2f}ms")
        click.echo(f"Throughput: {current.get('throughput_rps', 0):.2f} RPS")
        click.echo(f"GPU Utilization: {current.get('gpu_utilization', 0):.1%}")
        click.echo(f"Error Rate: {current.get('error_rate', 0):.1%}")
        click.echo(f"Cost/Hour: ${current.get('cost_per_hour', 0):.2f}")
        
        # Display recommendations
        recs = recommendations.get("recommendations", [])
        click.echo(f"\n💡 Recommendations:")
        for i, rec in enumerate(recs, 1):
            priority_emoji = "🔴" if rec.get("priority") == "high" else "🟡" if rec.get("priority") == "medium" else "🟢"
            click.echo(f"{i}. {priority_emoji} {rec.get('type', 'unknown').replace('_', ' ').title()}")
            click.echo(f"   Expected Gain: {rec.get('expected_gain', 1.0):.2f}x")
            click.echo(f"   Complexity: {rec.get('complexity', 'unknown')}")
            click.echo(f"   Reasoning: {rec.get('reasoning', 'No reasoning provided')}")
        
        # Display potential gains
        gains = recommendations.get("potential_gains", {})
        click.echo(f"\n📈 Potential Gains:")
        click.echo(f"Performance Gain: {gains.get('performance_gain', 1.0):.2f}x")
        click.echo(f"Cost Increase: {gains.get('cost_increase', 0):.1%}")
        click.echo(f"ROI: {gains.get('roi', 0):.2f}")
        
        # Display optimization priority
        priority = recommendations.get("optimization_priority", "unknown")
        complexity = recommendations.get("implementation_complexity", "unknown")
        click.echo(f"\n🎯 Overall Assessment:")
        click.echo(f"Priority: {priority}")
        click.echo(f"Implementation Complexity: {complexity}")
        
        # Display risks
        risks = recommendations.get("risk_assessment", [])
        if risks:
            click.echo(f"\n⚠️  Risks:")
            for risk in risks:
                click.echo(f"  • {risk}")
        
    except Exception as e:
        click.echo(f"❌ Error getting recommendations: {str(e)}")
        raise click.ClickException(str(e))

@optimize.group()
def config():
    """Optimization configuration management"""
    pass

@config.command()
def show():
    """Show current optimization configuration"""
    
    try:
        config = get_optimization_config()
        
        click.echo(f"⚙️  Optimization Configuration:")
        click.echo(f"=============================")
        click.echo(f"Auto Optimize: {config.auto_optimize}")
        click.echo(f"Optimization Interval: {config.optimization_interval}s")
        click.echo(f"Performance Threshold: {config.performance_threshold}")
        click.echo(f"Cost Threshold: {config.cost_threshold}")
        click.echo(f"Enable CUCo: {config.enable_cuco}")
        click.echo(f"Enable Warm Pool: {config.enable_warm_pool}")
        click.echo(f"Enable Semantic Routing: {config.enable_semantic_routing}")
        click.echo(f"Enable Auto Scaling: {config.enable_auto_scaling}")
        
        click.echo(f"\n🚀 CUCo Configuration:")
        click.echo(f"Enabled: {config.cuco_config.enabled}")
        click.echo(f"Min GPU Count: {config.cuco_config.min_gpu_count}")
        click.echo(f"Min Communication Intensity: {config.cuco_config.min_communication_intensity}")
        click.echo(f"Min Performance Gain: {config.cuco_config.min_performance_gain}x")
        click.echo(f"Max Cost Increase: {config.cuco_config.max_cost_increase:.1%}")
        click.echo(f"Auto Apply: {config.cuco_config.auto_apply}")
        click.echo(f"Monitoring Enabled: {config.cuco_config.monitoring_enabled}")
        click.echo(f"P95 Strict Mode: {config.cuco_config.p95_strict_mode}")
        
    except Exception as e:
        click.echo(f"❌ Error showing configuration: {str(e)}")
        raise click.ClickException(str(e))

@config.command()
@click.option('--auto-optimize', type=bool, help='Enable/disable auto optimization')
@click.option('--enable-cuco', type=bool, help='Enable/disable CUCo')
@click.option('--min-performance-gain', type=float, help='Minimum performance gain for CUCo')
@click.option('--max-cost-increase', type=float, help='Maximum cost increase for CUCo')
@click.option('--p95-strict-mode', type=bool, help='Enable/disable P95 strict mode')
def set(**kwargs):
    """Update optimization configuration"""
    
    try:
        # Filter out None values
        updates = {k: v for k, v in kwargs.items() if v is not None}
        
        if not updates:
            click.echo("No configuration updates provided")
            return
        
        # Update configuration
        update_optimization_config(updates)
        
        click.echo("✅ Configuration updated successfully:")
        for key, value in updates.items():
            click.echo(f"  {key}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Error updating configuration: {str(e)}")
        raise click.ClickException(str(e))

@config.command()
@click.option('--workload-type', type=click.Choice(['moe', 'attention', 'llm_training', 'distributed_inference']))
def p95_boundaries(workload_type: Optional[str]):
    """Show P95 boundaries for workload types"""
    
    try:
        from .core.optimization_config import OptimizationConfigManager
        
        config_manager = OptimizationConfigManager()
        boundaries = config_manager.get_p95_boundaries()
        
        if workload_type:
            if workload_type in boundaries:
                click.echo(f"🎯 P95 Boundaries for {workload_type.upper()}:")
                click.echo(f"=================================")
                for metric, value in boundaries[workload_type].items():
                    click.echo(f"{metric}: {value}")
            else:
                click.echo(f"❌ Unknown workload type: {workload_type}")
        else:
            click.echo(f"🎯 P95 Boundaries for All Workload Types:")
            click.echo(f"=======================================")
            for wtype, metrics in boundaries.items():
                click.echo(f"\n{wtype.upper()}:")
                for metric, value in metrics.items():
                    click.echo(f"  {metric}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Error showing P95 boundaries: {str(e)}")
        raise click.ClickException(str(e))

@optimize.command()
@click.argument('workload_type', type=click.Choice(['moe', 'attention', 'llm_training', 'distributed_inference']))
@click.argument('metrics_file', type=click.Path(exists=True))
def validate_p95(workload_type: str, metrics_file: str):
    """Validate metrics against P95 boundaries"""
    
    try:
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Initialize CUCo optimizer for validation
        config = TerradevConfig()
        metrics_collector = MetricsCollector()
        cuco_optimizer = CUCoOptimizer(config, metrics_collector)
        
        click.echo(f"🎯 Validating P95 boundaries for {workload_type}...")
        
        # Validate
        results = cuco_optimizer.validate_p95_boundaries(workload_type, metrics)
        
        if "error" in results:
            click.echo(f"❌ {results['error']}")
            return
        
        # Display results
        click.echo(f"\n📊 P95 Validation Results:")
        click.echo(f"=========================")
        click.echo(f"Overall Compliance: {'✅ PASS' if results['overall_compliance'] else '❌ FAIL'}")
        click.echo(f"Compliance Rate: {results['compliance_rate']:.1%}")
        
        # Display metric compliance
        compliance = results.get("compliance_results", {})
        click.echo(f"\n📈 Metric Compliance:")
        for metric, result in compliance.items():
            status = "✅" if result["meets_threshold"] else "❌"
            performance_level = result["performance_level"].title()
            click.echo(f"{status} {metric}: {result['provided_value']:.3f} / {result['p95_boundary']:.3f} ({performance_level})")
        
        # Display violations
        violations = results.get("violations", [])
        if violations:
            click.echo(f"\n❌ Violations:")
            for violation in violations:
                click.echo(f"  • {violation['metric']}: {violation['provided']:.3f} < {violation['expected']:.3f} (gap: {violation['gap_percentage']:.1f}%)")
        
        # Display recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            click.echo(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                click.echo(f"{i}. {rec}")
        
    except Exception as e:
        click.echo(f"❌ Error validating P95 boundaries: {str(e)}")
        raise click.ClickException(str(e))

# Register CLI commands
def register_optimization_commands(cli):
    """Register optimization commands with CLI"""
    cli.add_command(optimize)
