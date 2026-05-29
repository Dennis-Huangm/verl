# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel


class OpenAIFunctionPropertySchema(BaseModel):
    type: str | list[str]
    description: str | None = None
    enum: list[object] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema


class OpenAIFunctionToolSchema(BaseModel):
    type: str
    function: OpenAIFunctionSchema


class ToolResponse(BaseModel):
    text: str | None = None
    image: list[object] | None = None


class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        self.name = self.tool_schema.function.name

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs):
        return instance_id or "test-instance", ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        return None


def _load_crop_tool_class():
    schemas_mod = types.ModuleType("verl.tools.schemas")
    schemas_mod.OpenAIFunctionParametersSchema = OpenAIFunctionParametersSchema
    schemas_mod.OpenAIFunctionPropertySchema = OpenAIFunctionPropertySchema
    schemas_mod.OpenAIFunctionSchema = OpenAIFunctionSchema
    schemas_mod.OpenAIFunctionToolSchema = OpenAIFunctionToolSchema
    schemas_mod.ToolResponse = ToolResponse

    base_tool_mod = types.ModuleType("verl.tools.base_tool")
    base_tool_mod.BaseTool = BaseTool

    previous = {
        name: sys.modules.get(name)
        for name in ("verl", "verl.tools", "verl.tools.schemas", "verl.tools.base_tool")
    }
    sys.modules["verl"] = types.ModuleType("verl")
    sys.modules["verl.tools"] = types.ModuleType("verl.tools")
    sys.modules["verl.tools.schemas"] = schemas_mod
    sys.modules["verl.tools.base_tool"] = base_tool_mod
    try:
        module_path = (
            Path(__file__).resolve().parents[2]
            / "verl"
            / "tools"
            / "crop_image_tool.py"
        )
        spec = importlib.util.spec_from_file_location(
            "crop_image_tool_under_test", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CropImageTool, module.CROP_TOOL_RESPONSE_TEXT
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


CropImageTool, CROP_TOOL_RESPONSE_TEXT = _load_crop_tool_class()

_REQUIRED_PARAMETERS = {"x1", "y1", "x2", "y2", "description", "image_index"}


def _tool() -> CropImageTool:
    return CropImageTool(config={"type": "native"}, tool_schema=None)


def _valid_parameters() -> dict[str, object]:
    return {
        "x1": 2,
        "y1": 3,
        "x2": 8,
        "y2": 9,
        "description": "pointer and nearby scale evidence",
        "image_index": 0,
    }


def test_crop_image_tool_schema_is_structured_crop_only() -> None:
    tool = _tool()
    schema = tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)

    assert tool.name == "crop_image_tool"
    assert schema["function"]["name"] == "crop_image_tool"
    parameters = schema["function"]["parameters"]
    assert set(parameters["required"]) == _REQUIRED_PARAMETERS
    assert set(parameters["properties"]) == _REQUIRED_PARAMETERS
    assert "code" not in parameters["properties"]


def test_crop_image_tool_executes_valid_crop() -> None:
    async def _run():
        tool = _tool()
        instance_id, _ = await tool.create()
        response, reward, metrics = await tool.execute(
            instance_id,
            _valid_parameters(),
            images=[Image.new("RGB", (20, 16), "white")],
        )
        await tool.release(instance_id)
        return response, reward, metrics

    response, reward, metrics = asyncio.run(_run())

    assert response.text == CROP_TOOL_RESPONSE_TEXT
    assert reward == 0.0
    assert metrics == {
        "success": True,
        "image_index": 0,
        "crop_width": 6,
        "crop_height": 6,
    }
    assert response.image is not None
    assert len(response.image) == 1
    assert response.image[0].size == (6, 6)


def test_crop_image_tool_rejects_invalid_rectangle() -> None:
    async def _run():
        tool = _tool()
        instance_id, _ = await tool.create()
        parameters = {**_valid_parameters(), "x2": 2}
        response, reward, metrics = await tool.execute(
            instance_id,
            parameters,
            images=[Image.new("RGB", (20, 16), "white")],
        )
        await tool.release(instance_id)
        return response, reward, metrics

    response, reward, metrics = asyncio.run(_run())

    assert response.image is None
    assert response.text is not None
    assert response.text.startswith("Invalid crop request:")
    assert reward == 0.0
    assert metrics["success"] is False


def test_crop_image_tool_rejects_unexpected_code_parameter() -> None:
    async def _run():
        tool = _tool()
        instance_id, _ = await tool.create()
        parameters = {
            **_valid_parameters(),
            "code": "result = image.crop((0, 0, 1, 1))",
        }
        response, _, metrics = await tool.execute(
            instance_id,
            parameters,
            images=[Image.new("RGB", (20, 16), "white")],
        )
        await tool.release(instance_id)
        return response, metrics

    response, metrics = asyncio.run(_run())

    assert response.image is None
    assert response.text == "Invalid crop request: Unexpected crop parameter(s): code"
    assert metrics == {"success": False, "error": "Unexpected crop parameter(s): code"}


def test_crop_image_tool_rejects_invalid_image_index() -> None:
    async def _run():
        tool = _tool()
        instance_id, _ = await tool.create()
        parameters = {**_valid_parameters(), "image_index": 1}
        response, _, metrics = await tool.execute(
            instance_id,
            parameters,
            images=[Image.new("RGB", (20, 16), "white")],
        )
        await tool.release(instance_id)
        return response, metrics

    response, metrics = asyncio.run(_run())

    assert response.image is None
    assert response.text == "Invalid crop request: Invalid image_index: 1"
    assert metrics == {"success": False, "error": "Invalid image_index: 1"}


def test_crop_image_tool_loads_from_native_yaml(tmp_path: Path) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("tensordict")
    from verl.tools.crop_image_tool import CropImageTool as RuntimeCropImageTool
    from verl.tools.tool_registry import load_all_tools

    config_path = tmp_path / "crop_image_tool_config.yaml"
    config_path.write_text(
        """
tools:
  - class_name: "verl.tools.crop_image_tool.CropImageTool"
    config:
      type: native
""".strip(),
        encoding="utf-8",
    )

    tools = load_all_tools(tool_config_path=str(config_path), function_tool_path=None)

    assert len(tools) == 1
    assert isinstance(tools[0], RuntimeCropImageTool)
    assert tools[0].name == "crop_image_tool"
