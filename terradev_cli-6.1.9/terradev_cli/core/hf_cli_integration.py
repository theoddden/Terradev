#!/usr/bin/env python3
"""
HuggingFace Spaces CLI Integration with Smart Templates
Implements the CLI commands for HF Spaces deployment with hardware optimization
"""

import click
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .hf_spaces import HFSpaceConfig, HFSpacesDeployer
from .hf_smart_templates import SmartTemplateGenerator, HardwareOptimizer


@click.group()
def hf():
    """HuggingFace Spaces deployment commands"""
    pass


@hf.command()
@click.argument("space_name")
@click.option("--model-id", required=True, help="HuggingFace model ID")
@click.option(
    "--template", default="auto", help="Template type (auto, chat, embedding, image)"
)
@click.option("--hardware", help="Override hardware tier")
@click.option("--sdk", default="gradio", help="SDK (gradio, streamlit, docker)")
@click.option("--private", is_flag=True, help="Create private space")
@click.option("--env", multiple=True, help="Environment variables (key=value)")
@click.option("--budget", type=float, help="Budget constraint in $/hour")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be deployed without creating"
)
def space(
    space_name: str,
    model_id: str,
    template: str,
    hardware: Optional[str],
    sdk: str,
    private: bool,
    env: tuple,
    budget: Optional[float],
    dry_run: bool,
):
    """Deploy a model to HuggingFace Spaces with smart optimization"""

    click.echo(f"🚀 Deploying {model_id} to HuggingFace Spaces...")
    click.echo(f"📝 Space name: {space_name}")
    click.echo(f"🎯 Template: {template}")

    # Initialize smart template generator
    template_gen = SmartTemplateGenerator()
    hardware_optimizer = HardwareOptimizer()

    # Analyze model and get hardware recommendation
    click.echo("🔍 Analyzing model requirements...")
    model_spec = template_gen.analyze_model(model_id)

    if not model_spec:
        click.echo(f"❌ Unable to analyze model: {model_id}")
        click.echo("💡 Try with a known model like 'meta-llama/Llama-3-8B-Instruct'")
        return

    click.echo("✅ Model analyzed:")
    click.echo(f"   📊 Parameters: {model_spec.parameters}")
    click.echo(f"   💾 Model size: {model_spec.model_size_gb}GB")
    click.echo(f"   🎯 Model type: {model_spec.model_type}")
    click.echo(f"   🧠 Min memory: {model_spec.min_memory_gb}GB")

    # Get hardware recommendation
    click.echo("🔧 Finding optimal hardware...")
    hardware_rec = hardware_optimizer.get_hardware_recommendation(model_id, budget)

    if "error" in hardware_rec:
        click.echo(f"❌ {hardware_rec['error']}")
        return

    # Use provided hardware or recommendation
    hardware or hardware_rec["recommended_hardware"]

    click.echo("✅ Hardware recommendation:")
    click.echo(f"   🎯 Recommended: {hardware_rec['recommended_hardware']}")
    click.echo(
        f"   💰 Hourly cost: ${hardware_rec['cost_breakdown']['hourly_cost']:.2f}"
    )
    click.echo(f"   📊 Memory utilization: {hardware_rec['memory_utilization']:.1f}%")
    click.echo(f"   🚀 Performance score: {hardware_rec['performance_score']}")

    if budget:
        click.echo(f"   💸 Budget constraint: ${budget}/hour")

    # Show cost breakdown
    cost_breakdown = hardware_rec["cost_breakdown"]
    click.echo("💰 Cost breakdown:")
    click.echo(f"   📈 Hourly: ${cost_breakdown['hourly_cost']:.2f}")
    click.echo(
        f"   📅 Daily (24/7): ${cost_breakdown['cost_breakdown']['daily_24_7']:.2f}"
    )
    click.echo(f"   📅 Daily (8h): ${cost_breakdown['cost_breakdown']['daily_8h']:.2f}")
    click.echo(
        f"   📊 Monthly (24/7): ${cost_breakdown['cost_breakdown']['monthly_24_7']:.2f}"
    )
    click.echo(
        f"   📊 Monthly (8h): ${cost_breakdown['cost_breakdown']['monthly_8h']:.2f}"
    )

    # Show alternatives if available
    if "alternative_options" in hardware_rec and hardware_rec["alternative_options"]:
        click.echo("🔄 Alternative hardware options:")
        for alt in hardware_rec["alternative_options"][:3]:  # Show top 3
            status = "✅" if alt["suitable"] else "❌"
            click.echo(
                f"   {status} {alt['hardware']}: ${alt['hourly_cost']:.2f}/hr, {alt['memory_gb']}GB RAM"
            )

    if dry_run:
        click.echo("🔍 Dry run mode - not creating space")
        return

    # Generate smart template
    click.echo("🎨 Generating smart template...")
    template_config = template_gen.generate_smart_template(
        model_id, template, space_name
    )

    if "error" in template_config:
        click.echo(f"❌ {template_config['error']}")
        return

    # Override hardware if specified
    if hardware:
        template_config["hardware"] = hardware

    # Add environment variables
    env_vars = template_config.get("env_vars", {})
    for env_var in env:
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            env_vars[key] = value

    template_config["env_vars"] = env_vars
    template_config["private"] = private
    template_config["sdk"] = sdk

    click.echo("✅ Template generated:")
    click.echo(f"   🎯 Template type: {template_config['template_type']}")
    click.echo(f"   🔧 Hardware: {template_config['hardware']}")
    click.echo(f"   🛠️ SDK: {template_config['sdk']}")

    # Get HF token
    hf_token = _get_hf_token()
    if not hf_token:
        click.echo("❌ HuggingFace token not found")
        click.echo(
            "💡 Set HF_TOKEN environment variable or run 'terradev hf configure'"
        )
        return

    # Deploy to HF Spaces
    click.echo("🚀 Deploying to HuggingFace Spaces...")

    async def deploy_space():
        deployer = HFSpacesDeployer(hf_token)

        # Create config object
        config = HFSpaceConfig(
            name=template_config["name"],
            model_id=template_config["model_id"],
            hardware=template_config["hardware"],
            sdk=template_config["sdk"],
            python_version=template_config["python_version"],
            private=template_config["private"],
            env_vars=template_config["env_vars"],
        )

        # Deploy
        result = await deployer.create_space(config)

        if result["status"] == "created":
            click.echo("✅ Space created successfully!")
            click.echo(f"🔗 Space URL: {result['space_url']}")
            click.echo(f"🎯 Hardware: {result['hardware']}")
            click.echo(f"📝 Model: {result['model_id']}")

            # Show next steps
            click.echo("\n🎉 Next steps:")
            click.echo(f"1. Visit your space: {result['space_url']}")
            click.echo("2. Wait for the model to load (may take a few minutes)")
            click.echo("3. Test your deployed model!")

            # Save to local registry
            _save_deployment(result, template_config)

        else:
            click.echo(f"❌ Failed to create space: {result['error']}")

    # Run deployment
    asyncio.run(deploy_space())


@hf.command()
@click.argument("model_id")
@click.option("--budget", type=float, help="Budget constraint in $/hour")
def optimize(model_id: str, budget: Optional[float]):
    """Show hardware optimization recommendations for a model"""

    click.echo(f"🔍 Analyzing hardware options for {model_id}...")

    hardware_optimizer = HardwareOptimizer()

    # Get recommendation
    recommendation = hardware_optimizer.get_hardware_recommendation(model_id, budget)

    if "error" in recommendation:
        click.echo(f"❌ {recommendation['error']}")
        return

    # Show model analysis
    model_analysis = recommendation["model_analysis"]
    click.echo("\n📊 Model Analysis:")
    click.echo(f"   🎯 Parameters: {model_analysis['parameters']}")
    click.echo(f"   💾 Model size: {model_analysis['model_size_gb']}GB")
    click.echo(f"   🎯 Type: {model_analysis['model_type']}")
    click.echo(f"   🧠 Min memory: {model_analysis['min_memory_gb']}GB")

    # Show recommendation
    click.echo("\n🎯 Recommended Hardware:")
    click.echo(f"   🔧 Hardware: {recommendation['recommended_hardware']}")
    click.echo(
        f"   💰 Hourly cost: ${recommendation['cost_breakdown']['hourly_cost']:.2f}"
    )
    click.echo(f"   📊 Memory utilization: {recommendation['memory_utilization']:.1f}%")
    click.echo(f"   🚀 Performance score: {recommendation['performance_score']}")
    click.echo(f"   💬 Reason: {recommendation['recommendation_reason']}")

    # Show cost breakdown
    cost_breakdown = recommendation["cost_breakdown"]
    click.echo("\n💰 Cost Breakdown:")
    click.echo(f"   📈 Hourly: ${cost_breakdown['hourly_cost']:.2f}")
    click.echo(
        f"   📅 Daily (24/7): ${cost_breakdown['cost_breakdown']['daily_24_7']:.2f}"
    )
    click.echo(f"   📅 Daily (8h): ${cost_breakdown['cost_breakdown']['daily_8h']:.2f}")
    click.echo(
        f"   📊 Monthly (24/7): ${cost_breakdown['cost_breakdown']['monthly_24_7']:.2f}"
    )
    click.echo(
        f"   📊 Monthly (8h): ${cost_breakdown['cost_breakdown']['monthly_8h']:.2f}"
    )

    # Show alternatives
    if "alternative_options" in recommendation:
        click.echo("\n🔄 Alternative Options:")
        for alt in recommendation["alternative_options"]:
            status = "✅" if alt["suitable"] else "❌"
            click.echo(
                f"   {status} {alt['hardware']}: ${alt['hourly_cost']:.2f}/hr, {alt['memory_gb']}GB RAM"
            )
            click.echo(f"      Monthly (8h): ${alt['monthly_cost_8h']:.2f}")


@hf.command()
@click.argument("model_id")
def compare(model_id: str):
    """Compare all hardware options for a model"""

    click.echo(f"🔍 Comparing all hardware options for {model_id}...")

    hardware_optimizer = HardwareOptimizer()

    # Get comparison
    comparison = hardware_optimizer.compare_hardware_options(model_id)

    if "error" in comparison:
        click.echo(f"❌ {comparison['error']}")
        return

    # Show model requirements
    requirements = comparison["model_requirements"]
    click.echo("\n📊 Model Requirements:")
    click.echo(f"   🎯 Parameters: {requirements['parameters']}")
    click.echo(f"   💾 Model size: {requirements['model_size_gb']}GB")
    click.echo(f"   🧠 Min memory: {requirements['min_memory_gb']}GB")

    # Show comparison table
    click.echo("\n🔧 Hardware Comparison:")
    click.echo(
        f"{'Hardware':<15} {'Suitable':<10} {'Memory':<10} {'GPU':<10} {'Cost/hr':<10} {'Perf':<8} {'Util%':<8} {'Rec':<12}"
    )
    click.echo("-" * 85)

    for hw in comparison["hardware_comparison"]:
        suitable = "✅ Yes" if hw["suitable"] else "❌ No"
        utilization = (
            f"{hw['memory_utilization']:.1f}%" if hw["memory_utilization"] else "N/A"
        )
        click.echo(
            f"{hw['hardware']:<15} {suitable:<10} {hw['memory_gb']:<10} {hw['gpu_type']:<10} ${hw['hourly_cost']:<9.2f} {hw['performance_score']:<8} {utilization:<8} {hw['recommendation']:<12}"
        )


@hf.command()
@click.argument("model_id")
@click.option("--template", default="auto", help="Template type")
@click.option("--space-name", help="Space name (auto-generated if not provided)")
def preview(model_id: str, template: str, space_name: Optional[str]):
    """Preview the generated template without deploying"""

    click.echo(f"🔍 Previewing template for {model_id}...")

    template_gen = SmartTemplateGenerator()

    # Generate template
    template_config = template_gen.generate_smart_template(
        model_id, template, space_name
    )

    if "error" in template_config:
        click.echo(f"❌ {template_config['error']}")
        return

    # Show template configuration
    click.echo("\n📋 Template Configuration:")
    click.echo(f"   📝 Space name: {template_config['name']}")
    click.echo(f"   🎯 Model ID: {template_config['model_id']}")
    click.echo(f"   🔧 Hardware: {template_config['hardware']}")
    click.echo(f"   🛠️ SDK: {template_config['sdk']}")
    click.echo(f"   🎨 Template type: {template_config['template_type']}")

    # Show model spec
    model_spec = template_config["model_spec"]
    click.echo("\n📊 Model Specifications:")
    click.echo(f"   🎯 Parameters: {model_spec['parameters']}")
    click.echo(f"   💾 Model size: {model_spec['model_size_gb']}GB")
    click.echo(f"   🎯 Type: {model_spec['model_type']}")
    click.echo(f"   🧠 Min memory: {model_spec['min_memory_gb']}GB")

    # Show environment variables
    if template_config.get("env_vars"):
        click.echo("\n🔧 Environment Variables:")
        for key, value in template_config["env_vars"].items():
            click.echo(f"   {key}: {value}")

    # Show cost breakdown
    cost_breakdown = template_config["cost_breakdown"]
    click.echo("\n💰 Cost Breakdown:")
    click.echo(f"   📈 Hourly: ${cost_breakdown['hourly_cost']:.2f}")
    click.echo(
        f"   📅 Daily (24/7): ${cost_breakdown['cost_breakdown']['daily_24_7']:.2f}"
    )
    click.echo(f"   📅 Daily (8h): ${cost_breakdown['cost_breakdown']['daily_8h']:.2f}")
    click.echo(
        f"   📊 Monthly (24/7): ${cost_breakdown['cost_breakdown']['monthly_24_7']:.2f}"
    )
    click.echo(
        f"   📊 Monthly (8h): ${cost_breakdown['cost_breakdown']['monthly_8h']:.2f}"
    )

    # Show alternatives
    if "alternative_hardware" in template_config:
        click.echo("\n🔄 Alternative Hardware:")
        for alt in template_config["alternative_hardware"]:
            click.echo(
                f"   {alt['name']}: ${alt['hourly_cost']:.2f}/hr, {alt['memory_gb']}GB RAM"
            )


@hf.command()
def configure():
    """Configure HuggingFace token"""

    click.echo("🔧 Configuring HuggingFace token...")

    # Get token from user
    token = click.prompt("Enter your HuggingFace token", hide_input=True)

    # Save token
    config_dir = Path.home() / ".terradev"
    config_dir.mkdir(exist_ok=True)

    credentials_file = config_dir / "credentials.json"

    # Load existing credentials

    credentials = {}
    if credentials_file.exists():
        with open(credentials_file, "r") as f:
            credentials = json.load(f)

    # Add HF token
    credentials["huggingface_token"] = token

    # Save credentials
    with open(credentials_file, "w") as f:
        json.dump(credentials, f, indent=2)

    # Set file permissions
    import os

    os.chmod(credentials_file, 0o600)

    click.echo("✅ HuggingFace token configured successfully!")
    click.echo("💡 You can also set HF_TOKEN environment variable")


@hf.command()
def list():
    """List deployed spaces"""

    click.echo("📋 Listing deployed spaces...")

    # Load deployment registry
    registry_file = Path.home() / ".terradev" / "hf_deployments.json"

    if not registry_file.exists():
        click.echo("📝 No deployments found")
        return


    with open(registry_file, "r") as f:
        deployments = json.load(f)

    if not deployments:
        click.echo("📝 No deployments found")
        return

    click.echo(f"\n🚀 Deployed Spaces ({len(deployments)}):")
    click.echo(
        f"{'Name':<20} {'Model':<30} {'Hardware':<15} {'Created':<20} {'URL':<40}"
    )
    click.echo("-" * 125)

    for deployment in deployments:
        created = deployment.get("created_at", "Unknown")
        url = deployment.get("space_url", "Unknown")
        click.echo(
            f"{deployment['name']:<20} {deployment['model_id']:<30} {deployment['hardware']:<15} {created:<20} {url:<40}"
        )


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or credentials"""
    import os

    # Check environment variable first
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    # Check credentials file
    config_dir = Path.home() / ".terradev"
    credentials_file = config_dir / "credentials.json"

    if credentials_file.exists():
        import json

        with open(credentials_file, "r") as f:
            credentials = json.load(f)

        return credentials.get("huggingface_token")

    return None


def _save_deployment(result: Dict[str, Any], template_config: Dict[str, Any]):
    """Save deployment to local registry"""

    registry_file = Path.home() / ".terradev" / "hf_deployments.json"
    registry_file.parent.mkdir(exist_ok=True)


    # Load existing deployments
    deployments = []
    if registry_file.exists():
        with open(registry_file, "r") as f:
            deployments = json.load(f)

    # Add new deployment
    deployment = {
        "name": result["space_name"],
        "model_id": result["model_id"],
        "hardware": result["hardware"],
        "space_url": result["space_url"],
        "template_type": template_config.get("template_type", "auto"),
        "created_at": datetime.now().isoformat(),
        "cost_breakdown": template_config.get("cost_breakdown", {}),
    }

    deployments.append(deployment)

    # Save deployments
    with open(registry_file, "w") as f:
        json.dump(deployments, f, indent=2)

    # Set file permissions
    import os

    os.chmod(registry_file, 0o600)


# Add the HF commands to the main CLI
def register_hf_commands(cli):
    """Register the unified HuggingFace top-level command with the main CLI."""
    cli.add_command(hf, name="huggingface")
