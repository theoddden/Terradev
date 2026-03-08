#!/usr/bin/env python3
"""
Smart HuggingFace Spaces Templates with Hardware Optimization
Implements intelligent template generation and hardware tier optimization
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelSpec:
    """Model specifications for hardware optimization"""
    model_id: str
    model_size_gb: float
    parameters: str
    model_type: str  # llm, embedding, image, audio
    min_memory_gb: float
    recommended_hardware: str
    hourly_cost: float
    streaming_capable: bool = True
    quantization_support: bool = True

@dataclass
class HardwareTier:
    """Hardware tier specifications"""
    name: str
    memory_gb: float
    gpu_type: str
    hourly_cost: float
    performance_score: float
    available_regions: List[str]

class SmartTemplateGenerator:
    """Generates intelligent templates based on model analysis"""
    
    def __init__(self):
        self.model_specs = self._load_model_specs()
        self.hardware_tiers = self._load_hardware_tiers()
        
    def _load_model_specs(self) -> Dict[str, ModelSpec]:
        """Load model specifications database"""
        return {
            # Llama Models
            "meta-llama/Llama-3-8B-Instruct": ModelSpec(
                model_id="meta-llama/Llama-3-8B-Instruct",
                model_size_gb=16.0,
                parameters="8B",
                model_type="llm",
                min_memory_gb=12.0,
                recommended_hardware="a10g-large",
                hourly_cost=0.60,
                streaming_capable=True,
                quantization_support=True
            ),
            "meta-llama/Llama-3-70B-Instruct": ModelSpec(
                model_id="meta-llama/Llama-3-70B-Instruct",
                model_size_gb=140.0,
                parameters="70B",
                model_type="llm",
                min_memory_gb=80.0,
                recommended_hardware="a100-80gb",
                hourly_cost=4.06,
                streaming_capable=True,
                quantization_support=True
            ),
            "meta-llama/Llama-2-7b-hf": ModelSpec(
                model_id="meta-llama/Llama-2-7b-hf",
                model_size_gb=13.0,
                parameters="7B",
                model_type="llm",
                min_memory_gb=10.0,
                recommended_hardware="a10g-large",
                hourly_cost=0.60,
                streaming_capable=True,
                quantization_support=True
            ),
            "meta-llama/Llama-2-70b-hf": ModelSpec(
                model_id="meta-llama/Llama-2-70b-hf",
                model_size_gb=140.0,
                parameters="70B",
                model_type="llm",
                min_memory_gb=80.0,
                recommended_hardware="a100-80gb",
                hourly_cost=4.06,
                streaming_capable=True,
                quantization_support=True
            ),
            
            # Mistral Models
            "mistralai/Mistral-7B-Instruct-v0.2": ModelSpec(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
                model_size_gb=14.0,
                parameters="7B",
                model_type="llm",
                min_memory_gb=10.0,
                recommended_hardware="a10g-large",
                hourly_cost=0.60,
                streaming_capable=True,
                quantization_support=True
            ),
            "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelSpec(
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_size_gb=100.0,
                parameters="8x7B",
                model_type="llm",
                min_memory_gb=48.0,
                recommended_hardware="a100-40gb",
                hourly_cost=2.50,
                streaming_capable=True,
                quantization_support=True
            ),
            
            # Embedding Models
            "sentence-transformers/all-MiniLM-L6-v2": ModelSpec(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                model_size_gb=0.5,
                parameters="22M",
                model_type="embedding",
                min_memory_gb=2.0,
                recommended_hardware="cpu-upgrade",
                hourly_cost=0.15,
                streaming_capable=False,
                quantization_support=False
            ),
            "sentence-transformers/all-mpnet-base-v2": ModelSpec(
                model_id="sentence-transformers/all-mpnet-base-v2",
                model_size_gb=1.0,
                parameters="110M",
                model_type="embedding",
                min_memory_gb=4.0,
                recommended_hardware="cpu-upgrade",
                hourly_cost=0.15,
                streaming_capable=False,
                quantization_support=False
            ),
            
            # Image Models
            "runwayml/stable-diffusion-v1-5": ModelSpec(
                model_id="runwayml/stable-diffusion-v1-5",
                model_size_gb=4.0,
                parameters="2B",
                model_type="image",
                min_memory_gb=8.0,
                recommended_hardware="t4-medium",
                hourly_cost=0.35,
                streaming_capable=False,
                quantization_support=False
            ),
            "stabilityai/stable-diffusion-xl-base-1.0": ModelSpec(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                model_size_gb=12.0,
                parameters="2.6B",
                model_type="image",
                min_memory_gb=16.0,
                recommended_hardware="a10g-large",
                hourly_cost=0.60,
                streaming_capable=False,
                quantization_support=False
            ),
            
            # Code Models
            "codellama/CodeLlama-7b-hf": ModelSpec(
                model_id="codellama/CodeLlama-7b-hf",
                model_size_gb=13.0,
                parameters="7B",
                model_type="llm",
                min_memory_gb=10.0,
                recommended_hardware="a10g-large",
                hourly_cost=0.60,
                streaming_capable=True,
                quantization_support=True
            ),
            "microsoft/CodeGPT-small-py": ModelSpec(
                model_id="microsoft/CodeGPT-small-py",
                model_size_gb=2.0,
                parameters="124M",
                model_type="llm",
                min_memory_gb=4.0,
                recommended_hardware="cpu-upgrade",
                hourly_cost=0.15,
                streaming_capable=True,
                quantization_support=True
            )
        }
    
    def _load_hardware_tiers(self) -> Dict[str, HardwareTier]:
        """Load hardware tier specifications"""
        return {
            "cpu-basic": HardwareTier(
                name="cpu-basic",
                memory_gb=8.0,
                gpu_type="CPU",
                hourly_cost=0.05,
                performance_score=1.0,
                available_regions=["us-east-1", "eu-west-1"]
            ),
            "cpu-upgrade": HardwareTier(
                name="cpu-upgrade",
                memory_gb=16.0,
                gpu_type="CPU",
                hourly_cost=0.15,
                performance_score=2.0,
                available_regions=["us-east-1", "eu-west-1"]
            ),
            "t4-medium": HardwareTier(
                name="t4-medium",
                memory_gb=16.0,
                gpu_type="T4",
                hourly_cost=0.35,
                performance_score=5.0,
                available_regions=["us-east-1", "eu-west-1"]
            ),
            "a10g-large": HardwareTier(
                name="a10g-large",
                memory_gb=24.0,
                gpu_type="A10G",
                hourly_cost=0.60,
                performance_score=8.0,
                available_regions=["us-east-1", "eu-west-1"]
            ),
            "a10g-xlarge": HardwareTier(
                name="a10g-xlarge",
                memory_gb=48.0,
                gpu_type="A10G",
                hourly_cost=1.20,
                performance_score=16.0,
                available_regions=["us-east-1", "eu-west-1"]
            ),
            "a100-40gb": HardwareTier(
                name="a100-40gb",
                memory_gb=80.0,
                gpu_type="A100",
                hourly_cost=2.50,
                performance_score=20.0,
                available_regions=["us-east-1"]
            ),
            "a100-80gb": HardwareTier(
                name="a100-80gb",
                memory_gb=160.0,
                gpu_type="A100",
                hourly_cost=4.06,
                performance_score=40.0,
                available_regions=["us-east-1"]
            )
        }
    
    def analyze_model(self, model_id: str) -> Optional[ModelSpec]:
        """Analyze model and return specifications"""
        # Direct lookup
        if model_id in self.model_specs:
            return self.model_specs[model_id]
        
        # Pattern-based estimation
        return self._estimate_model_specs(model_id)
    
    def _estimate_model_specs(self, model_id: str) -> Optional[ModelSpec]:
        """Estimate model specs from model ID patterns"""
        model_id_lower = model_id.lower()
        
        # LLM patterns
        if any(keyword in model_id_lower for keyword in ["llama", "mistral", "mixtral", "codellama"]):
            if "70b" in model_id_lower or "8x7b" in model_id_lower:
                return ModelSpec(
                    model_id=model_id,
                    model_size_gb=140.0,
                    parameters="70B",
                    model_type="llm",
                    min_memory_gb=80.0,
                    recommended_hardware="a100-80gb",
                    hourly_cost=4.06,
                    streaming_capable=True,
                    quantization_support=True
                )
            elif "34b" in model_id_lower:
                return ModelSpec(
                    model_id=model_id,
                    model_size_gb=68.0,
                    parameters="34B",
                    model_type="llm",
                    min_memory_gb=40.0,
                    recommended_hardware="a100-40gb",
                    hourly_cost=2.50,
                    streaming_capable=True,
                    quantization_support=True
                )
            elif "13b" in model_id_lower or "8b" in model_id_lower or "7b" in model_id_lower:
                return ModelSpec(
                    model_id=model_id,
                    model_size_gb=16.0,
                    parameters="8B",
                    model_type="llm",
                    min_memory_gb=12.0,
                    recommended_hardware="a10g-large",
                    hourly_cost=0.60,
                    streaming_capable=True,
                    quantization_support=True
                )
        
        # Embedding patterns
        elif "embedding" in model_id_lower or "sentence-transformers" in model_id_lower:
            return ModelSpec(
                model_id=model_id,
                model_size_gb=1.0,
                parameters="100M",
                model_type="embedding",
                min_memory_gb=4.0,
                recommended_hardware="cpu-upgrade",
                hourly_cost=0.15,
                streaming_capable=False,
                quantization_support=False
            )
        
        # Image patterns
        elif any(keyword in model_id_lower for keyword in ["stable-diffusion", "sd", "midjourney"]):
            if "xl" in model_id_lower or "sdxl" in model_id_lower:
                return ModelSpec(
                    model_id=model_id,
                    model_size_gb=12.0,
                    parameters="2.6B",
                    model_type="image",
                    min_memory_gb=16.0,
                    recommended_hardware="a10g-large",
                    hourly_cost=0.60,
                    streaming_capable=False,
                    quantization_support=False
                )
            else:
                return ModelSpec(
                    model_id=model_id,
                    model_size_gb=4.0,
                    parameters="2B",
                    model_type="image",
                    min_memory_gb=8.0,
                    recommended_hardware="t4-medium",
                    hourly_cost=0.35,
                    streaming_capable=False,
                    quantization_support=False
                )
        
        return None
    
    def optimize_hardware(self, model_spec: ModelSpec, budget_constraint: Optional[float] = None) -> List[HardwareTier]:
        """Optimize hardware selection for model"""
        suitable_tiers = []
        
        for tier_name, tier in self.hardware_tiers.items():
            # Check memory requirements
            if tier.memory_gb >= model_spec.min_memory_gb:
                # Check budget constraint
                if budget_constraint is None or tier.hourly_cost <= budget_constraint:
                    suitable_tiers.append(tier)
        
        # Sort by performance (best first)
        suitable_tiers.sort(key=lambda x: x.performance_score, reverse=True)
        
        return suitable_tiers
    
    def generate_cost_breakdown(self, model_spec: ModelSpec, hardware_tier: HardwareTier) -> Dict[str, Any]:
        """Generate cost breakdown for deployment"""
        monthly_cost_24_7 = hardware_tier.hourly_cost * 24 * 30
        monthly_cost_8h = hardware_tier.hourly_cost * 8 * 30  # 8 hours/day
        
        return {
            "model_id": model_spec.model_id,
            "model_size_gb": model_spec.model_size_gb,
            "parameters": model_spec.parameters,
            "recommended_hardware": hardware_tier.name,
            "hardware_memory": hardware_tier.memory_gb,
            "hourly_cost": hardware_tier.hourly_cost,
            "cost_breakdown": {
                "hourly": hardware_tier.hourly_cost,
                "daily_24_7": hardware_tier.hourly_cost * 24,
                "daily_8h": hardware_tier.hourly_cost * 8,
                "monthly_24_7": monthly_cost_24_7,
                "monthly_8h": monthly_cost_8h
            },
            "memory_utilization": (model_spec.min_memory_gb / hardware_tier.memory_gb) * 100,
            "performance_score": hardware_tier.performance_score,
            "streaming_supported": model_spec.streaming_capable,
            "quantization_supported": model_spec.quantization_support
        }
    
    def generate_smart_template(self, model_id: str, template_type: str = "auto", space_name: str = None) -> Dict[str, Any]:
        """Generate smart template based on model analysis"""
        model_spec = self.analyze_model(model_id)
        if not model_spec:
            return {"error": f"Unable to analyze model: {model_id}"}
        
        # Optimize hardware
        suitable_hardware = self.optimize_hardware(model_spec)
        if not suitable_hardware:
            return {"error": f"No suitable hardware found for model: {model_id}"}
        
        best_hardware = suitable_hardware[0]
        
        # Generate space name if not provided
        if not space_name:
            space_name = f"terradev-{model_spec.model_type}-{model_spec.parameters.lower()}"
        
        # Generate template based on model type
        if template_type == "auto":
            template_type = model_spec.model_type
        
        template_config = {
            "name": space_name,
            "model_id": model_id,
            "model_spec": {
                "parameters": model_spec.parameters,
                "model_size_gb": model_spec.model_size_gb,
                "model_type": model_spec.model_type,
                "min_memory_gb": model_spec.min_memory_gb
            },
            "hardware": best_hardware.name,
            "sdk": "gradio" if model_spec.streaming_capable else "streamlit",
            "python_version": "3.10",
            "private": False,
            "template_type": template_type
        }
        
        # Add model-specific environment variables
        env_vars = {
            "MODEL_ID": model_id,
            "MODEL_TYPE": model_spec.model_type,
            "HARDWARE": best_hardware.name
        }
        
        if model_spec.model_type == "llm":
            env_vars.update({
                "MAX_LENGTH": "500",
                "TEMPERATURE": "0.7",
                "STREAMING": "true" if model_spec.streaming_capable else "false"
            })
        elif model_spec.model_type == "embedding":
            env_vars.update({
                "BATCH_SIZE": "32",
                "MAX_SEQ_LENGTH": "512"
            })
        elif model_spec.model_type == "image":
            env_vars.update({
                "IMAGE_SIZE": "512",
                "NUM_INFERENCE_STEPS": "20"
            })
        
        template_config["env_vars"] = env_vars
        
        # Generate cost breakdown
        cost_breakdown = self.generate_cost_breakdown(model_spec, best_hardware)
        template_config["cost_breakdown"] = cost_breakdown
        
        # Add alternative hardware options
        if len(suitable_hardware) > 1:
            template_config["alternative_hardware"] = [
                {
                    "name": tier.name,
                    "hourly_cost": tier.hourly_cost,
                    "memory_gb": tier.memory_gb,
                    "performance_score": tier.performance_score
                }
                for tier in suitable_hardware[1:4]  # Top 3 alternatives
            ]
        
        return template_config
    
    def generate_chat_template(self, model_id: str, space_name: str) -> str:
        """Generate optimized chat template with streaming"""
        model_spec = self.analyze_model(model_id)
        if not model_spec or model_spec.model_type != "llm":
            return "# Error: This model is not suitable for chat applications"
        
        return f'''import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import time

# Model configuration
MODEL_ID = "{model_id}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 500
TEMPERATURE = 0.7

# Load model and tokenizer
print(f"Loading model: {{MODEL_ID}}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use optimized loading for large models
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {{e}}")
    model = None
    tokenizer = None

def generate_response_stream(message, history, temperature=TEMPERATURE, max_length=MAX_LENGTH):
    """Generate streaming response"""
    if not model or not tokenizer:
        yield "❌ Model loading failed. Please check the configuration."
        return
    
    try:
        # Format conversation history
        conversation = []
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            if assistant:
                conversation.append({"role": "assistant", "content": assistant})
        conversation.append({"role": "user", "content": message})
        
        # Generate response with streaming
        inputs = tokenizer.apply_chat_template(
            conversation, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(DEVICE)
        
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            inputs=inputs,
            streamer=streamer,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the response
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            yield partial_text
            
    except Exception as e:
        yield f"❌ Generation error: {{e}}"

def generate_response_non_stream(message, history, temperature=TEMPERATURE, max_length=MAX_LENGTH):
    """Generate non-streaming response (fallback)"""
    if not model or not tokenizer:
        return "❌ Model loading failed. Please check the configuration."
    
    try:
        # Simple prompt for non-streaming mode
        prompt = message
        if history:
            context = "\\n".join([f"User: {{h}}\\nAssistant: {{a}}" for h, a in history[-3:]])
            prompt = f"{{context}}\\nUser: {{message}}\\nAssistant: "
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        return response
        
    except Exception as e:
        return f"❌ Generation error: {{e}}"

# Create enhanced chat interface
with gr.Blocks(title="{space_name}", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🤖 {space_name}
    
    **Model:** `{MODEL_ID}`  
    **Hardware:** {model_spec.recommended_hardware.upper()}  
    **Parameters:** {model_spec.parameters}  
    **Streaming:** {'✅ Enabled' if model_spec.streaming_capable else '❌ Disabled'}
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=TEMPERATURE,
                step=0.1,
                label="Temperature",
                info="Higher = more creative"
            )
            
            max_length_slider = gr.Slider(
                minimum=50,
                maximum=1000,
                value=MAX_LENGTH,
                step=50,
                label="Max Length",
                info="Maximum response length"
            )
            
            streaming_checkbox = gr.Checkbox(
                value=True,
                label="Enable Streaming",
                info="Show response as it's generated"
            )
    
    # Event handlers
    def user_input(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot_response(history, temperature, max_length, streaming):
        if not history:
            return history
        
        user_message = history[-1][0]
        
        if streaming:
            # Streaming response
            history[-1][1] = ""
            for chunk in generate_response_stream(user_message, history[:-1], temperature, max_length):
                history[-1][1] = chunk
                yield history
        else:
            # Non-streaming response
            response = generate_response_non_stream(user_message, history[:-1], temperature, max_length)
            history[-1][1] = response
            yield history
    
    def clear_chat():
        return []
    
    # Wire up the events
    msg.submit(
        user_input,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        bot_response,
        [chatbot, temperature_slider, max_length_slider, streaming_checkbox],
        chatbot
    )
    
    submit_btn.click(
        user_input,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        bot_response,
        [chatbot, temperature_slider, max_length_slider, streaming_checkbox],
        chatbot
    )
    
    clear_btn.click(clear_chat, outputs=chatbot)
    
    # Add example prompts
    gr.Markdown("### 💡 Example Prompts")
    gr.Examples(
        examples=[
            ["Explain quantum computing in simple terms"],
            ["Write a Python function to calculate fibonacci numbers"],
            ["What are the benefits of renewable energy?"],
            ["Create a short story about a robot learning to paint"],
            ["How does machine learning work?"]
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
'''
    
    def generate_embedding_template(self, model_id: str, space_name: str) -> str:
        """Generate optimized embedding template"""
        return f'''import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go

# Model configuration
MODEL_ID = "{model_id}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512

st.set_page_config(page_title="{space_name}", layout="wide")

st.title(f"🔤 {space_name}")
st.markdown(f"**Model:** `{MODEL_ID}`")
st.markdown("**Hardware:** CPU-Optimized for Embeddings")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        # Use sentence-transformers for better performance
        model = SentenceTransformer(MODEL_ID)
        return model
    except Exception as e:
        st.error(f"Error loading model: {{e}}")
        return None

st.header("Text Embedding Service")

# Load model
model = load_model()

if not model:
    st.error("❌ Model loading failed. Please check the configuration.")
    st.stop()

# Input section
st.subheader("📝 Input Text")
input_text = st.text_area(
    "Enter text to embed:",
    height=150,
    placeholder="Enter your text here... You can enter multiple sentences, one per line."
)

# Options
col1, col2 = st.columns(2)
with col1:
    normalize_embeddings = st.checkbox("Normalize embeddings", value=True)
    show_dimensions = st.checkbox("Show dimension values", value=False)

with col2:
    batch_mode = st.checkbox("Batch processing", value=False)
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7)

if st.button("🚀 Generate Embeddings", type="primary"):
    if input_text:
        with st.spinner("Generating embeddings..."):
            try:
                # Split text into lines for batch processing
                texts = [line.strip() for line in input_text.split('\\n') if line.strip()]
                
                if batch_mode and len(texts) > 1:
                    # Batch processing
                    embeddings = model.encode(texts, normalize_embeddings=normalize_embeddings)
                    
                    # Display results
                    st.subheader("📊 Results")
                    
                    # Create dataframe with embeddings
                    results_df = pd.DataFrame({
                        'Text': texts,
                        'Embedding': [emb.tolist() for emb in embeddings],
                        'Dimensions': [len(emb) for emb in embeddings]
                    })
                    
                    # Show summary
                    st.write(f"✅ Generated embeddings for **{{len(texts)}}** texts")
                    st.write(f"📏 Embedding dimensions: **{{len(embeddings[0])}}**")
                    
                    # Similarity matrix
                    if len(texts) > 1:
                        st.subheader("🔗 Similarity Matrix")
                        
                        # Calculate cosine similarity
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarity_matrix = cosine_similarity(embeddings)
                        
                        # Create heatmap
                        fig = px.imshow(
                            similarity_matrix,
                            labels=dict(x="Text", y="Text", color="Similarity"),
                            x=[f"Text {{i+1}}" for i in range(len(texts))],
                            y=[f"Text {{i+1}}" for i in range(len(texts))],
                            color_continuous_scale="viridis"
                        )
                        fig.update_layout(title="Text Similarity Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show high similarities
                        high_similarities = []
                        for i in range(len(texts)):
                            for j in range(i+1, len(texts)):
                                if similarity_matrix[i][j] > similarity_threshold:
                                    high_similarities.append({
                                        'Text 1': texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i],
                                        'Text 2': texts[j][:50] + "..." if len(texts[j]) > 50 else texts[j],
                                        'Similarity': f"{similarity_matrix[i][j]:.3f}"
                                    })
                        
                        if high_similarities:
                            st.write(f"🎯 High similarities (>{similarity_threshold}):")
                            st.dataframe(pd.DataFrame(high_similarities))
                        else:
                            st.write(f"📉 No similarities above {similarity_threshold}")
                    
                    # Show individual embeddings
                    if show_dimensions:
                        st.subheader("🔢 Embedding Values")
                        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                            with st.expander(f"Text {i+1}: {text[:50] + '...' if len(text) > 50 else text}"):
                                st.write(f"Shape: {embedding.shape}")
                                st.write(f"Mean: {np.mean(embedding):.4f}")
                                st.write(f"Std: {np.std(embedding):.4f}")
                                st.write(f"Min: {np.min(embedding):.4f}")
                                st.write(f"Max: {np.max(embedding):.4f}")
                                
                                # Show first 50 dimensions
                                st.write("First 50 dimensions:")
                                st.write(embedding[:50])
                    
                else:
                    # Single text processing
                    embedding = model.encode([input_text], normalize_embeddings=normalize_embeddings)[0]
                    
                    st.success("✅ Embedding generated successfully!")
                    st.write(f"📏 Embedding dimensions: **{len(embedding)}**")
                    st.write(f"📊 Mean: **{np.mean(embedding):.4f}**")
                    st.write(f"📊 Std: **{np.std(embedding):.4f}**")
                    
                    if show_dimensions:
                        st.subheader("🔢 Embedding Values")
                        st.write(embedding)
                
            except Exception as e:
                st.error(f"❌ Error generating embeddings: {e}")
    else:
        st.warning("⚠️ Please enter some text to embed.")

# Advanced features
st.subheader("🔧 Advanced Features")

col1, col2 = st.columns(2)

with col1:
    st.write("**Embedding Statistics**")
    if input_text and model:
        try:
            texts = [line.strip() for line in input_text.split('\\n') if line.strip()]
            embeddings = model.encode(texts[:5])  # Limit to 5 for performance
            
            # Calculate statistics
            all_embeddings = np.concatenate(embeddings)
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Total Embeddings", len(embeddings))
            with stats_col2:
                st.metric("Dimensions", len(embeddings[0]))
            with stats_col3:
                st.metric("Mean Value", f"{np.mean(all_embeddings):.4f}")
                
        except Exception as e:
            st.error(f"Error calculating statistics: {e}")

with col2:
    st.write("**Model Information**")
    try:
        # Get model info
        model_info = {
            "Model ID": MODEL_ID,
            "Device": DEVICE,
            "Batch Size": BATCH_SIZE,
            "Max Sequence Length": MAX_SEQ_LENGTH,
            "Normalization": "Enabled" if normalize_embeddings else "Disabled"
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")
            
    except Exception as e:
        st.error(f"Error getting model info: {e}")

# Footer
st.markdown("---")
st.markdown("🚀 **Powered by Terradev** | Multi-Cloud GPU Arbitrage Platform")
'''

class HardwareOptimizer:
    """Hardware tier optimization engine"""
    
    def __init__(self):
        self.template_generator = SmartTemplateGenerator()
    
    def get_hardware_recommendation(self, model_id: str, budget_constraint: Optional[float] = None) -> Dict[str, Any]:
        """Get hardware recommendation with cost breakdown"""
        model_spec = self.template_generator.analyze_model(model_id)
        if not model_spec:
            return {"error": f"Unable to analyze model: {model_id}"}
        
        suitable_hardware = self.template_generator.optimize_hardware(model_spec, budget_constraint)
        
        if not suitable_hardware:
            return {"error": f"No suitable hardware found for model: {model_id}"}
        
        best_hardware = suitable_hardware[0]
        cost_breakdown = self.template_generator.generate_cost_breakdown(model_spec, best_hardware)
        
        recommendation = {
            "model_id": model_id,
            "model_analysis": {
                "parameters": model_spec.parameters,
                "model_size_gb": model_spec.model_size_gb,
                "model_type": model_spec.model_type,
                "min_memory_gb": model_spec.min_memory_gb
            },
            "recommended_hardware": best_hardware.name,
            "recommendation_reason": f"Model requires {model_spec.min_memory_gb}GB memory, {best_hardware.name} provides {best_hardware.memory_gb}GB",
            "cost_breakdown": cost_breakdown,
            "memory_utilization": (model_spec.min_memory_gb / best_hardware.memory_gb) * 100,
            "performance_score": best_hardware.performance_score,
            "alternative_options": [
                {
                    "hardware": tier.name,
                    "hourly_cost": tier.hourly_cost,
                    "memory_gb": tier.memory_gb,
                    "monthly_cost_24_7": tier.hourly_cost * 24 * 30,
                    "monthly_cost_8h": tier.hourly_cost * 8 * 30,
                    "suitable": tier.memory_gb >= model_spec.min_memory_gb
                }
                for tier in suitable_hardware[1:5]  # Top 4 alternatives
            ]
        }
        
        return recommendation
    
    def compare_hardware_options(self, model_id: str) -> Dict[str, Any]:
        """Compare all hardware options for a model"""
        model_spec = self.template_generator.analyze_model(model_id)
        if not model_spec:
            return {"error": f"Unable to analyze model: {model_id}"}
        
        all_hardware = []
        for tier_name, tier in self.template_generator.hardware_tiers.items():
            suitable = tier.memory_gb >= model_spec.min_memory_gb
            cost_breakdown = self.template_generator.generate_cost_breakdown(model_spec, tier)
            
            all_hardware.append({
                "hardware": tier.name,
                "suitable": suitable,
                "memory_gb": tier.memory_gb,
                "gpu_type": tier.gpu_type,
                "hourly_cost": tier.hourly_cost,
                "performance_score": tier.performance_score,
                "memory_utilization": (model_spec.min_memory_gb / tier.memory_gb) * 100 if suitable else None,
                "monthly_cost_24_7": tier.hourly_cost * 24 * 30,
                "monthly_cost_8h": tier.hourly_cost * 8 * 30,
                "recommendation": "✅ Recommended" if suitable and tier.name == model_spec.recommended_hardware else ("✅ Suitable" if suitable else "❌ Insufficient memory")
            })
        
        # Sort by performance score (best first)
        all_hardware.sort(key=lambda x: x["performance_score"], reverse=True)
        
        return {
            "model_id": model_id,
            "model_requirements": {
                "parameters": model_spec.parameters,
                "min_memory_gb": model_spec.min_memory_gb,
                "model_size_gb": model_spec.model_size_gb
            },
            "hardware_comparison": all_hardware
        }
