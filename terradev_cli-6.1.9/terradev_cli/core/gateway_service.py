#!/usr/bin/env python3
"""
Gateway Service - API Gateway for Inference Serving

Provides OpenAI/Anthropic/custom API entry and exit points for inference workflows.
Integrates with Terradev's inference routing and KV cache management.

Features:
  - OpenAI-compatible API endpoints (/v1/chat/completions, /v1/completions)
  - Anthropic-compatible API endpoints (/v1/messages, /v1/messages/batches)
  - Custom workflow entry/exit points
  - Integration with inference router for intelligent routing
  - Support for streaming responses
  - Request/response transformation and validation
"""

import asyncio
import inspect
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# FastAPI 0.104.x with Starlette >=1.0: Starlette's Router no longer accepts
# on_startup/on_shutdown, but FastAPI's APIRouter still passes them. Patch the
# parent Router.__init__ to drop the legacy kwargs so the gateway app can load.
try:
    import starlette.routing

    _starlette_router_init = starlette.routing.Router.__init__
    if "on_startup" not in inspect.signature(_starlette_router_init).parameters:
        def _patched_router_init(self, *args, **kwargs):
            kwargs.pop("on_startup", None)
            kwargs.pop("on_shutdown", None)
            return _starlette_router_init(self, *args, **kwargs)

        starlette.routing.Router.__init__ = _patched_router_init
except Exception:  # noqa: BLE001
    pass

try:
    from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported API providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class GatewayConfig:
    """Gateway service configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    enable_openai: bool = True
    enable_anthropic: bool = True
    enable_custom: bool = True
    max_concurrent_requests: int = 100
    request_timeout: int = 120
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Inference routing integration
    enable_inference_router: bool = True
    default_model: str = "meta-llama/Llama-3.1-70B-Instruct"
    
    # Custom workflow endpoints
    custom_entry_points: Dict[str, str] = field(default_factory=dict)
    custom_exit_points: Dict[str, str] = field(default_factory=dict)


if FASTAPI_AVAILABLE:
    # Pydantic models for OpenAI API
    class OpenAIMessage(BaseModel):
        role: str
        content: str
    
    class OpenAIChatRequest(BaseModel):
        model: str
        messages: List[OpenAIMessage]
        temperature: float = 0.7
        max_tokens: int = 2048
        stream: bool = False
        top_p: float = 1.0
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
    
    class OpenAIChatResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[Dict[str, Any]]
        usage: Dict[str, int]
    
    # Pydantic models for Anthropic API
    class AnthropicMessage(BaseModel):
        role: str
        content: str
    
    class AnthropicRequest(BaseModel):
        model: str
        messages: List[AnthropicMessage]
        max_tokens: int = 2048
        temperature: float = 0.7
        top_p: float = 1.0
        stream: bool = False
    
    class AnthropicResponse(BaseModel):
        id: str
        type: str = "message"
        role: str = "assistant"
        content: List[Dict[str, Any]]
        model: str
        stop_reason: str = "end_turn"
        usage: Dict[str, int]


class GatewayService:
    """Main gateway service for inference serving"""
    
    def __init__(self, config: GatewayConfig):
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for gateway service. "
                "Install with: pip install fastapi uvicorn"
            )
        
        self.config = config
        self.app = FastAPI(
            title="Terradev Inference Gateway",
            description="API Gateway for inference serving with OpenAI/Anthropic/custom endpoints",
            version="1.0.0"
        )
        self._setup_middleware()
        self._setup_routes()
        
        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_count = 0
        
        # Inference router integration
        self.inference_router = None
        if config.enable_inference_router:
            try:
                from terradev_cli.core.inference_router import InferenceRouter
                self.inference_router = InferenceRouter()
                logger.info("Inference router integration enabled")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to initialize inference router: {e}")
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "active_requests": len(self.active_requests),
                "total_requests": self.request_count
            }
        
        # OpenAI-compatible endpoints
        if self.config.enable_openai:
            self._setup_openai_routes()
        
        # Anthropic-compatible endpoints
        if self.config.enable_anthropic:
            self._setup_anthropic_routes()
        
        # Custom workflow endpoints
        if self.config.enable_custom:
            self._setup_custom_routes()
        
        # Gateway management endpoints
        @self.app.get("/v1/gateway/status")
        async def gateway_status():
            return {
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "enable_openai": self.config.enable_openai,
                    "enable_anthropic": self.config.enable_anthropic,
                    "enable_custom": self.config.enable_custom,
                    "max_concurrent_requests": self.config.max_concurrent_requests,
                },
                "active_requests": len(self.active_requests),
                "total_requests": self.request_count,
                "inference_router_enabled": self.inference_router is not None,
            }
    
    def _setup_openai_routes(self):
        """Setup OpenAI-compatible API routes"""
        
        @self.app.post("/v1/chat/completions")
        async def openai_chat_completions(request: OpenAIChatRequest):
            """OpenAI-compatible chat completions endpoint"""
            request_id = str(uuid.uuid4())
            self.request_count += 1
            
            # Check concurrent request limit
            if len(self.active_requests) >= self.config.max_concurrent_requests:
                raise HTTPException(status_code=429, detail="Too many concurrent requests")
            
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "model": request.model,
                "provider": "openai",
            }
            
            try:
                # Process the request through inference router if available
                if self.inference_router:
                    response_data = await self._route_request(
                        request_id=request_id,
                        provider=APIProvider.OPENAI,
                        model=request.model,
                        messages=[{"role": m.role, "content": m.content} for m in request.messages],
                        parameters={
                            "temperature": request.temperature,
                            "max_tokens": request.max_tokens,
                            "top_p": request.top_p,
                        }
                    )
                else:
                    # Fallback to direct processing
                    response_data = await self._process_openai_request(request, request_id)
                
                if request.stream:
                    return StreamingResponse(
                        self._stream_openai_response(response_data, request_id),
                        media_type="text/event-stream"
                    )
                else:
                    return JSONResponse(content=response_data)
            
            finally:
                self.active_requests.pop(request_id, None)
        
        @self.app.post("/v1/completions")
        async def openai_completions(request: Request):
            """OpenAI-compatible completions endpoint"""
            # Similar implementation to chat completions
            return await openai_chat_completions(request)
    
    def _setup_anthropic_routes(self):
        """Setup Anthropic-compatible API routes"""
        
        @self.app.post("/v1/messages")
        async def anthropic_messages(request: AnthropicRequest):
            """Anthropic-compatible messages endpoint"""
            request_id = str(uuid.uuid4())
            self.request_count += 1
            
            if len(self.active_requests) >= self.config.max_concurrent_requests:
                raise HTTPException(status_code=429, detail="Too many concurrent requests")
            
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "model": request.model,
                "provider": "anthropic",
            }
            
            try:
                if self.inference_router:
                    response_data = await self._route_request(
                        request_id=request_id,
                        provider=APIProvider.ANTHROPIC,
                        model=request.model,
                        messages=[{"role": m.role, "content": m.content} for m in request.messages],
                        parameters={
                            "temperature": request.temperature,
                            "max_tokens": request.max_tokens,
                            "top_p": request.top_p,
                        }
                    )
                else:
                    response_data = await self._process_anthropic_request(request, request_id)
                
                if request.stream:
                    return StreamingResponse(
                        self._stream_anthropic_response(response_data, request_id),
                        media_type="text/event-stream"
                    )
                else:
                    return JSONResponse(content=response_data)
            
            finally:
                self.active_requests.pop(request_id, None)
    
    def _setup_custom_routes(self):
        """Setup custom workflow entry/exit points"""
        
        @self.app.post("/v1/custom/entry/{workflow_id}")
        async def custom_entry_point(workflow_id: str, request: Request):
            """Custom workflow entry point"""
            request_id = str(uuid.uuid4())
            self.request_count += 1
            
            # Get custom entry point configuration
            entry_config = self.config.custom_entry_points.get(workflow_id)
            if not entry_config:
                raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
            
            try:
                request_data = await request.json()
                
                # Process through custom workflow
                response_data = await self._process_custom_workflow(
                    workflow_id=workflow_id,
                    entry_point=entry_config,
                    request_data=request_data,
                    request_id=request_id
                )
                
                return JSONResponse(content=response_data)
            
            except Exception as e:  # noqa: BLE001
                logger.error(f"Custom workflow error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/custom/exit/{workflow_id}")
        async def custom_exit_point(workflow_id: str, request: Request):
            """Custom workflow exit point"""
            request_id = str(uuid.uuid4())
            
            exit_config = self.config.custom_exit_points.get(workflow_id)
            if not exit_config:
                raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
            
            try:
                request_data = await request.json()
                
                # Process exit point logic
                response_data = await self._process_custom_exit(
                    workflow_id=workflow_id,
                    exit_point=exit_config,
                    request_data=request_data,
                    request_id=request_id
                )
                
                return JSONResponse(content=response_data)
            
            except Exception as e:  # noqa: BLE001
                logger.error(f"Custom exit point error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _route_request(
        self,
        request_id: str,
        provider: APIProvider,
        model: str,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route request through inference router"""
        # This would integrate with the existing inference router
        # For now, return a mock response
        return {
            "id": request_id,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the inference router."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
                "completion_tokens": 20,
                "total_tokens": sum(len(m.get("content", "")) for m in messages) + 20
            }
        }
    
    async def _process_openai_request(
        self,
        request: OpenAIChatRequest,
        request_id: str
    ) -> Dict[str, Any]:
        """Process OpenAI request (fallback without inference router)"""
        # Mock implementation - in production, this would call actual inference
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock OpenAI response. Configure inference endpoints for real responses."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(m.content) for m in request.messages),
                "completion_tokens": 25,
                "total_tokens": sum(len(m.content) for m in request.messages) + 25
            }
        }
    
    async def _process_anthropic_request(
        self,
        request: AnthropicRequest,
        request_id: str
    ) -> Dict[str, Any]:
        """Process Anthropic request (fallback without inference router)"""
        return {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "This is a mock Anthropic response. Configure inference endpoints for real responses."
            }],
            "model": request.model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": sum(len(m.content) for m in request.messages),
                "output_tokens": 25
            }
        }
    
    async def _process_custom_workflow(
        self,
        workflow_id: str,
        entry_point: str,
        request_data: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Process custom workflow entry point"""
        # This would integrate with custom workflow logic
        return {
            "workflow_id": workflow_id,
            "request_id": request_id,
            "status": "processed",
            "entry_point": entry_point,
            "result": request_data
        }
    
    async def _process_custom_exit(
        self,
        workflow_id: str,
        exit_point: str,
        request_data: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Process custom workflow exit point"""
        return {
            "workflow_id": workflow_id,
            "request_id": request_id,
            "status": "completed",
            "exit_point": exit_point,
            "result": request_data
        }
    
    async def _stream_openai_response(
        self,
        response_data: Dict[str, Any],
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI-compatible response"""
        # Mock streaming implementation
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": response_data.get("model", ""),
            "choices": [{
                "index": 0,
                "delta": {"content": "This is a mock streaming response."},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final chunk
        chunk["choices"][0]["delta"] = {}
        chunk["choices"][0]["finish_reason"] = "stop"
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    async def _stream_anthropic_response(
        self,
        response_data: Dict[str, Any],
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream Anthropic-compatible response"""
        # Mock streaming implementation
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "This is a mock streaming response."}
        }
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': response_data})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop', 'stop_reason': 'end_turn'})}\n\n"
    
    async def start(self):
        """Start the gateway server"""
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def run_sync(self):
        """Run the gateway server synchronously"""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )


def create_gateway_config(
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_openai: bool = True,
    enable_anthropic: bool = True,
    enable_custom: bool = True,
    max_concurrent_requests: int = 100,
    request_timeout: int = 120,
    enable_cors: bool = True,
    cors_origins: Optional[List[str]] = None,
    enable_inference_router: bool = True,
    default_model: str = "meta-llama/Llama-3.1-70B-Instruct",
) -> GatewayConfig:
    """Create a gateway configuration"""
    return GatewayConfig(
        host=host,
        port=port,
        enable_openai=enable_openai,
        enable_anthropic=enable_anthropic,
        enable_custom=enable_custom,
        max_concurrent_requests=max_concurrent_requests,
        request_timeout=request_timeout,
        enable_cors=enable_cors,
        cors_origins=cors_origins or ["*"],
        enable_inference_router=enable_inference_router,
        default_model=default_model,
    )
