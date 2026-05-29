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

from typing import Any

from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

TOOL_NAME = "crop_image_tool"
REQUIRED_PARAMETERS = ["x1", "y1", "x2", "y2", "description", "image_index"]

# Tool-response text returned after a successful crop. This must stay byte-for-byte
# identical to ``NEUTRAL_OBSERVATION_TEXT`` used when building the Tool-SFT data
# (data_synthesis/export_tool_sft_json.py). The model is SFT-trained to receive this
# instructional scaffold after every crop; returning a different string at RL rollout
# time creates a train/inference distribution mismatch on the tool-response channel.
CROP_TOOL_RESPONSE_TEXT = (
    "Here is the cropped image returned by the tool. Analyze this visual evidence "
    "carefully. Continue reasoning in a think block to decide whether another "
    "crop is needed. If more visual evidence is needed, make another "
    "crop_image_tool call. Otherwise, provide the final reading in the answer "
    "block."
)


class CropImageTool(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name=TOOL_NAME,
                # Kept byte-for-byte identical to the Tool-SFT schema
                # (data_synthesis/tool_sft_schema.py::build_tool_schema) so the rendered
                # <tools> block matches between SFT and RL rollout.
                description=(
                    "Crop a rectangular region from an image for focused visual "
                    "inspection of analog meter evidence such as a pointer, scale, "
                    "tick labels, endpoint, meniscus, or selected multimeter mode."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "x1": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Left pixel coordinate of the crop rectangle.",
                        ),
                        "y1": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Top pixel coordinate of the crop rectangle.",
                        ),
                        "x2": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Right pixel coordinate of the crop rectangle.",
                        ),
                        "y2": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Bottom pixel coordinate of the crop rectangle.",
                        ),
                        "description": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Short description of the region or visual cue being cropped.",
                        ),
                        "image_index": OpenAIFunctionPropertySchema(
                            type="integer",
                            minimum=0,
                            description="Index of the image in the trajectory to crop, starting at 0.",
                        ),
                    },
                    required=REQUIRED_PARAMETERS,
                ),
            ),
        )

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        images = _image_data_from_kwargs(kwargs)
        if not images:
            return _invalid_response("No images are available to crop")

        parsed = _parse_parameters(parameters)
        if isinstance(parsed, str):
            return _invalid_response(parsed)

        image_index, bbox = parsed
        if image_index < 0 or image_index >= len(images):
            return _invalid_response(f"Invalid image_index: {image_index}")

        image = images[image_index]
        if not isinstance(image, Image.Image):
            return _invalid_response(f"Image at index {image_index} is not a PIL image")

        clamped = _clamp_bbox(bbox, image.size)
        if isinstance(clamped, str):
            return _invalid_response(clamped)

        cropped = image.crop(clamped)
        metrics = {
            "success": True,
            "image_index": image_index,
            "crop_width": cropped.size[0],
            "crop_height": cropped.size[1],
        }
        return ToolResponse(text=CROP_TOOL_RESPONSE_TEXT, image=[cropped]), 0.0, metrics


def _image_data_from_kwargs(kwargs: dict[str, Any]) -> list[Any]:
    agent_data = kwargs.get("agent_data")
    if agent_data is not None:
        image_data = getattr(agent_data, "image_data", None)
        if image_data is None:
            return []
        return image_data if isinstance(image_data, list) else [image_data]

    images = kwargs.get("images")
    if images is None:
        return []
    return images if isinstance(images, list) else [images]


def _parse_parameters(
    parameters: dict[str, Any],
) -> tuple[int, tuple[int, int, int, int]] | str:
    allowed = set(REQUIRED_PARAMETERS)
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        return f"Unexpected crop parameter(s): {', '.join(unexpected)}"

    missing = [name for name in REQUIRED_PARAMETERS if name not in parameters]
    if missing:
        return f"Missing crop parameter(s): {', '.join(missing)}"

    description = parameters["description"]
    if not isinstance(description, str) or not description.strip():
        return "description must be a non-empty string"
    if len(description) > 240:
        return "description must be at most 240 characters"

    try:
        x1 = _strict_int(parameters["x1"], "x1")
        y1 = _strict_int(parameters["y1"], "y1")
        x2 = _strict_int(parameters["x2"], "x2")
        y2 = _strict_int(parameters["y2"], "y2")
        image_index = _strict_int(parameters["image_index"], "image_index")
    except ValueError as exc:
        return str(exc)

    if x2 <= x1 or y2 <= y1:
        return "Invalid crop rectangle: x2/y2 must be greater than x1/y1"
    return image_index, (x1, y1, x2, y2)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _clamp_bbox(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int] | str:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    clamped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return "Invalid crop rectangle after clamping to image bounds"
    return clamped


def _invalid_response(message: str) -> tuple[ToolResponse, float, dict[str, Any]]:
    return (
        ToolResponse(text=f"Invalid crop request: {message}"),
        0.0,
        {"success": False, "error": message},
    )
