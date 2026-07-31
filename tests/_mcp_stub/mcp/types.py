"""Minimal MCP SDK types for Python 3.9 testing.

This module is a drop-in subset of the official ``mcp.types`` package used to
satisfy the Terradev test suite on Python versions where the real ``mcp`` SDK
cannot be installed (it requires Python >= 3.10).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextContent:
    type: str = "text"
    text: str = ""


@dataclass
class ImageContent:
    type: str = "image"
    data: str = ""
    mimeType: str = ""


@dataclass
class EmbeddedResource:
    type: str = "resource"
    resource: Any = None


@dataclass
class CallToolRequestParams:
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallToolRequest:
    method: str = ""
    params: Any = field(default_factory=CallToolRequestParams)

    def __post_init__(self):
        if isinstance(self.params, dict):
            self.params = CallToolRequestParams(**self.params)


@dataclass
class CallToolResult:
    content: List[Any] = field(default_factory=list)
    isError: bool = False


@dataclass
class Tool:
    name: str = ""
    description: str = ""
    inputSchema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    uri: str = ""
    name: str = ""
    description: str = ""
    mimeType: str = "text/plain"


@dataclass
class TextResourceContents:
    uri: str = ""
    mimeType: str = "text/plain"
    text: str = ""


@dataclass
class BlobResourceContents:
    uri: str = ""
    mimeType: str = "application/octet-stream"
    blob: str = ""


@dataclass
class ListToolsRequest:
    method: str = "tools/list"
    params: Optional[Dict[str, Any]] = None


@dataclass
class ListToolsResult:
    tools: List[Tool] = field(default_factory=list)


@dataclass
class ListResourcesRequest:
    method: str = "resources/list"
    params: Optional[Dict[str, Any]] = None


@dataclass
class ListResourcesResult:
    resources: List[Resource] = field(default_factory=list)


@dataclass
class ReadResourceRequest:
    method: str = "resources/read"
    params: Optional[Dict[str, Any]] = None


@dataclass
class ReadResourceResult:
    contents: List[Any] = field(default_factory=list)


@dataclass
class GetPromptRequest:
    method: str = "prompts/get"
    params: Optional[Dict[str, Any]] = None


@dataclass
class GetPromptResult:
    description: str = ""
    messages: List[Any] = field(default_factory=list)


@dataclass
class ListPromptsRequest:
    method: str = "prompts/list"
    params: Optional[Dict[str, Any]] = None


@dataclass
class ListPromptsResult:
    prompts: List[Any] = field(default_factory=list)
