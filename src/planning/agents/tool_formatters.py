"""
Tool definition formatters for embedding in prompts.

This module provides formatters that convert tool definitions (list of dicts)
into string representations suitable for embedding in system messages.
"""

import json
from typing import Any


class GptOssFormatter:
    """Format tool definitions in TypeScript-style format for GPT-OSS models."""

    def format(self, tools: list[dict[str, Any]]) -> str:
        """
        Format tools as TypeScript-style namespace declarations.

        Args:
            tools: List of tool definition dictionaries, where each tool has a
                   'function' key containing name, description, and parameters

        Returns:
            TypeScript-style string representation of tools
        """
        lines = []

        for tool_def in tools:
            # Extract function definition
            tool = tool_def.get("function", tool_def)

            # Add tool description as comment
            lines.append(f"// {tool['description']}")

            # Start type declaration
            type_decl = f"type {tool['name']} = "

            # Check if tool has parameters
            parameters = tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required_params = parameters.get("required", [])

            if properties:
                lines.append(type_decl + "(_: {")

                # Render each parameter
                for param_name, param_spec in properties.items():
                    # Add parameter description as comment if present
                    if "description" in param_spec:
                        lines.append(f"// {param_spec['description']}")

                    # Parameter name with optional marker
                    is_required = param_name in required_params
                    optional_marker = "" if is_required else "?"

                    # Render TypeScript type
                    ts_type = self._render_typescript_type(param_spec, required_params)

                    # Add default value comment if present
                    default_comment = ""
                    if "default" in param_spec:
                        default_value = param_spec["default"]
                        if isinstance(default_value, str):
                            default_comment = f" // default: {json.dumps(default_value)}"
                        else:
                            default_comment = f" // default: {json.dumps(default_value)}"

                    lines.append(f"{param_name}{optional_marker}: {ts_type},{default_comment}")

                lines.append("}) => any;")
            else:
                lines.append(type_decl + "() => any;")

            # Add blank line after each tool
            lines.append("")

        return "```\n" + "\n".join(lines) + "\n```"

    def _render_typescript_type(
        self,
        param_spec: dict[str, Any],
        required_params: list[str]
    ) -> str:
        """
        Convert a parameter specification to TypeScript type notation.

        Args:
            param_spec: Parameter specification dictionary
            required_params: List of required parameter names

        Returns:
            TypeScript type string
        """
        param_type = param_spec.get("type")

        # Handle array types
        if param_type == "array":
            items = param_spec.get("items", {})
            if items:
                item_type = items.get("type")
                if item_type == "string":
                    base_type = "string[]"
                elif item_type in ("number", "integer"):
                    base_type = "number[]"
                elif item_type == "boolean":
                    base_type = "boolean[]"
                else:
                    inner_type = self._render_typescript_type(items, required_params)
                    if "object" in inner_type or len(inner_type) > 50:
                        base_type = "any[]"
                    else:
                        base_type = f"{inner_type}[]"
            else:
                base_type = "any[]"

            if param_spec.get("nullable"):
                return f"{base_type} | null"
            return base_type

        # Handle array of types (Union types)
        if isinstance(param_type, list):
            if len(param_type) > 1:
                return " | ".join(param_type)
            return param_type[0] if param_type else "any"

        # Handle oneOf schemas
        if "oneOf" in param_spec:
            one_of = param_spec["oneOf"]
            # Check for complex unions with multiple object variants
            has_object_variants = sum(1 for v in one_of if v.get("type") == "object")
            if has_object_variants > 0 and len(one_of) > 1:
                return "any"

            # Render union type
            types = []
            for variant in one_of:
                types.append(self._render_typescript_type(variant, required_params))
            return " | ".join(types)

        # Handle string types
        if param_type == "string":
            if "enum" in param_spec:
                enum_values = [f'"{val}"' for val in param_spec["enum"]]
                return " | ".join(enum_values)
            if param_spec.get("nullable"):
                return "string | null"
            return "string"

        # Handle number types
        if param_type in ("number", "integer"):
            return "number"

        # Handle boolean type
        if param_type == "boolean":
            return "boolean"

        # Handle object types
        if param_type == "object":
            properties = param_spec.get("properties")
            if properties:
                prop_lines = []
                nested_required = param_spec.get("required", [])
                for prop_name, prop_spec in properties.items():
                    is_required = prop_name in nested_required
                    optional_marker = "" if is_required else "?"
                    prop_type = self._render_typescript_type(prop_spec, nested_required)
                    prop_lines.append(f"{prop_name}{optional_marker}: {prop_type}")
                return "{\n" + ",\n".join(prop_lines) + "\n}"
            return "object"

        # Default fallback
        return "any"


class JsonToolFormatter:
    """Format tool definitions as JSON string."""

    def format(self, tools: list[dict[str, Any]]) -> str:
        """
        Format tools as a JSON string.

        Args:
            tools: List of tool definition dictionaries

        Returns:
            JSON string representation of tools
        """
        return f"```json\n{json.dumps(tools, indent=2)}\n```"


# Registry mapping formatter IDs to formatter classes
FORMATTER_REGISTRY: dict[str, type] = {
    "json": JsonToolFormatter,
    "gpt-oss": GptOssFormatter,
}


def get_formatter(formatter_id: str = "json"):
    """
    Get a formatter instance by ID.

    Args:
        formatter_id: ID of the formatter to retrieve (default: "json")

    Returns:
        Instance of the requested formatter

    Raises:
        ValueError: If formatter_id is not found in registry
    """
    if formatter_id not in FORMATTER_REGISTRY:
        raise ValueError(
            f"Unknown formatter ID: '{formatter_id}'. "
            f"Available formatters: {list(FORMATTER_REGISTRY.keys())}"
        )
    return FORMATTER_REGISTRY[formatter_id]()
