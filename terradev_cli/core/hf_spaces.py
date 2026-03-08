#!/usr/bin/env python3
"""
HuggingFace Spaces One-Click Deployment
Q1 2026: +$5M revenue opportunity with 100k researcher reach
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HFSpaceConfig:
    """HuggingFace Space configuration"""
    name: str
    model_id: str
    hardware: str = "cpu-basic"  # cpu-basic, cpu-upgrade, t4-medium, a10g-large, a10g-large
    sdk: str = "gradio"  # gradio, streamlit, docker
    python_version: str = "3.10"
    secrets: Dict[str, str] = None
    env_vars: Dict[str, str] = None
    private: bool = False

class HFSpacesDeployer:
    """One-click HuggingFace Spaces deployment"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.api_base = "https://huggingface.co/api"
        
        try:
            import aiohttp
            self.aiohttp = aiohttp
        except ImportError:
            print("❌ aiohttp required for HF Spaces deployment")
            print("   Install with: pip install aiohttp")
            self.aiohttp = None
        
    async def create_space(self, config: HFSpaceConfig) -> Dict[str, Any]:
        """Create HuggingFace Space with one click"""
        if not self.aiohttp:
            return {"status": "error", "error": "aiohttp not available"}
            
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        # Space configuration
        space_data = {
            "name": config.name,
            "sdk": config.sdk,
            "python_version": config.python_version,
            "hardware": config.hardware,
            "private": config.private,
            "tags": ["terradev", "gpu", "ml"],
            "description": f"Terradev deployment of {config.model_id}"
        }
        
        try:
            async with self.aiohttp.ClientSession() as session:
                # Create space
                async with session.post(
                    f"{self.api_base}/spaces/create",
                    headers=headers,
                    json=space_data
                ) as response:
                    if response.status == 200:
                        space_info = await response.json()
                        space_name = space_info.get("name", config.name)
                        
                        # Add secrets if provided
                        if config.secrets:
                            await self._add_secrets(session, space_name, config.secrets, headers)
                        
                        # Add environment variables if provided
                        if config.env_vars:
                            await self._add_env_vars(session, space_name, config.env_vars, headers)
                        
                        # Generate and upload app.py
                        await self._upload_app_file(session, space_name, config, headers)
                        
                        return {
                            "status": "created",
                            "space_name": space_name,
                            "space_url": f"https://huggingface.co/spaces/{space_name}",
                            "hardware": config.hardware,
                            "model_id": config.model_id
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "error": f"Failed to create space: {response.status} - {error_text}"
                        }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _add_secrets(self, session, space_name: str, secrets: Dict[str, str], headers: Dict[str, str]):
        """Add secrets to space"""
        for key, value in secrets.items():
            async with session.post(
                f"{self.api_base}/spaces/{space_name}/secrets",
                headers=headers,
                json={"key": key, "value": value}
            ) as response:
                if response.status != 200:
                    print(f"Warning: Failed to add secret {key}")
    
    async def _add_env_vars(self, session, space_name: str, env_vars: Dict[str, str], headers: Dict[str, str]):
        """Add environment variables to space"""
        async with session.post(
            f"{self.api_base}/spaces/{space_name}/env",
            headers=headers,
            json=env_vars
        ) as response:
            if response.status != 200:
                print(f"Warning: Failed to add environment variables")
    
    async def _upload_app_file(self, session, space_name: str, config: HFSpaceConfig, headers: Dict[str, str]):
        """Generate and upload app.py file"""
        app_content = self._generate_app_content(config)
        
        # Upload app.py
        files_data = self.aiohttp.FormData()
        files_data.add_field('file', app_content, filename='app.py', content_type='text/plain')
        
        async with session.post(
            f"{self.api_base}/spaces/{space_name}/files/main/app.py",
            headers=headers,
            data=files_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to upload app.py: {response.status} - {error_text}")
    
    def _generate_app_content(self, config: HFSpaceConfig) -> str:
        """Generate app.py content based on SDK"""
        if config.sdk == "gradio":
            return self._generate_gradio_app(config)
        elif config.sdk == "streamlit":
            return self._generate_streamlit_app(config)
        elif config.sdk == "docker":
            return self._generate_docker_app(config)
        else:
            return self._generate_gradio_app(config)  # default
    
    def _generate_gradio_app(self, config: HFSpaceConfig) -> str:
        """Generate Gradio app.py"""
        return f'''import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Model configuration
MODEL_ID = "{config.model_id}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model: {{e}}")
    # Fallback for demo
    tokenizer = None
    model = None

def generate_text(prompt, max_length=100, temperature=0.7):
    """Generate text using the model"""
    if not model or not tokenizer:
        return "Model loading failed. Please check the configuration."
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    except Exception as e:
        return f"Generation error: {{e}}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=500, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="{config.name}",
    description="Terradev deployment of {config.model_id}",
    examples=[
        ["The future of AI is", 100, 0.7],
        ["Machine learning will", 150, 0.8],
        ["In a world where", 200, 0.6]
    ]
)

if __name__ == "__main__":
    iface.launch()
'''
    
    def _generate_streamlit_app(self, config: HFSpaceConfig) -> str:
        """Generate Streamlit app.py"""
        return f'''import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Model configuration
MODEL_ID = "{config.model_id}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {{e}}")
        return None, None

st.title("{config.name}")
st.write("Terradev deployment of {config.model_id}")

# Load model
tokenizer, model = load_model()

if not model or not tokenizer:
    st.error("Model loading failed. Please check the configuration.")
    st.stop()

# User inputs
prompt = st.text_area("Enter your prompt:", height=100, placeholder="Enter your prompt here...")
max_length = st.slider("Max Length", min_value=10, max_value=500, value=100)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7)

if st.button("Generate"):
    if prompt:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = generated_text[len(prompt):].strip()
            
            st.success("Generated Text:")
            st.write(result)
        except Exception as e:
            st.error(f"Generation error: {{e}}")
    else:
        st.warning("Please enter a prompt.")
'''
    
    def _generate_docker_app(self, config: HFSpaceConfig) -> str:
        """Generate Docker app.py"""
        return f'''# Docker-based deployment for {config.model_id}
# This is a placeholder for Docker-based Spaces

import subprocess
import sys
import os

def main():
    print("Terradev Docker Space for {config.model_id}")
    print("Model: {config.model_id}")
    print("Hardware: {config.hardware}")
    
    # Your Docker app logic here
    # This would typically involve running your model server
    
if __name__ == "__main__":
    main()
'''

class HFSpaceTemplates:
    """Pre-configured templates for common use cases"""
    
    @staticmethod
    def get_llm_template(model_id: str, space_name: str) -> HFSpaceConfig:
        """Template for LLM deployment"""
        return HFSpaceConfig(
            name=space_name,
            model_id=model_id,
            hardware="a10g-large",  # GPU for LLM
            sdk="gradio",
            python_version="3.10",
            private=False,
            env_vars={
                "MODEL_ID": model_id,
                "MAX_LENGTH": "500",
                "TEMPERATURE": "0.7"
            }
        )
    
    @staticmethod
    def get_embedding_template(model_id: str, space_name: str) -> HFSpaceConfig:
        """Template for embedding model deployment"""
        return HFSpaceConfig(
            name=space_name,
            model_id=model_id,
            hardware="cpu-upgrade",  # CPU for embeddings
            sdk="streamlit",
            python_version="3.10",
            private=False,
            env_vars={
                "MODEL_ID": model_id,
                "BATCH_SIZE": "32"
            }
        )
    
    @staticmethod
    def get_image_model_template(model_id: str, space_name: str) -> HFSpaceConfig:
        """Template for image model deployment"""
        return HFSpaceConfig(
            name=space_name,
            model_id=model_id,
            hardware="t4-medium",  # GPU for image models
            sdk="gradio",
            python_version="3.10",
            private=False,
            env_vars={
                "MODEL_ID": model_id,
                "IMAGE_SIZE": "512"
            }
        )
