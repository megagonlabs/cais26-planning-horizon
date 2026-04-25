"""
Tool registry for configuration-based tool management.

This module provides the ToolRegistry class for managing tools through
configuration loading with OpenAI function schema generation for agent
selection.
"""

from typing import Any
import logging

from ..tools.base_tools import Tool
from ..tools.collection import TOOLS, TOOL_FACTORIES


logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing tools through configuration-based loading.

    The ToolRegistry provides configuration-driven management of tool
    components for the multi-agent system. It loads and instantiates
    tools from configuration dictionaries, maintains collections of
    tools, and generates OpenAI function calling schemas for agent
    selection.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: dict[str, Tool] = {}
        self.tool_set_registry: dict[str, list[str]] = {}  # tool_set_id -> list of tool_ids
        self.factory_instances: dict[str, Any] = {}  # tool_set_id -> factory instance

    def load_from_config(self, config_dict: dict[str, Any]) -> None:
        """
        Load and instantiate tools from configuration dictionary.

        Args:
            config_dict: Dictionary containing tool configurations with
                        tool IDs as keys and settings as values

        Raises:
            ValueError: If tool configuration is invalid
            ImportError: If tool class cannot be imported
            RuntimeError: If tool instantiation fails
        """
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration must be a dictionary")

        logger.info(f"Loading {len(config_dict)} tools from configuration")

        for tool_id, tool_config in config_dict.items():
            self._load_tool(tool_id, tool_config)

        logger.info(f"Successfully loaded {len(self.tools)} tools")

    def _load_tool(self, tool_id: str, tool_config: dict[str, Any]) -> None:
        """
        Load and instantiate a single tool or tool set from configuration.

        Args:
            tool_id: Unique identifier for the tool or tool set
            tool_config: Configuration dictionary for the tool

        Raises:
            ValueError: If tool configuration is missing required fields
            ImportError: If tool class cannot be imported
        """
        if not isinstance(tool_config, dict):
            raise ValueError(f"Tool configuration for '{tool_id}' must be a dictionary")

        # Handle individual tool loading
        # Get tool type and parameters
        tool_type = tool_config.get("type")
        if not tool_type:
            raise ValueError(f"Tool '{tool_id}' missing required 'type' field")

        # Check if this is a tool set (bulk registration via factory)
        if tool_type in TOOL_FACTORIES:
            self._load_tool_set(tool_id, tool_config)
            return

        tool_creator = TOOLS.get(tool_type)
        if not tool_creator:
            raise ValueError(
                f"Could not find a tool with the type '{tool_type}'. "
                f"Available tool types are: {', '.join(TOOLS.keys())}"
            )

        # Instantiate the tool
        tool = tool_creator(**tool_config.get("parameters", {}))
        self.register(tool_id, tool)

    def _load_tool_set(self, tool_set_id: str, tool_config: dict[str, Any]) -> None:
        """
        Load a tool set using its factory class.

        Args:
            tool_set_id: ID of the tool set (not currently used, for logging)
            tool_config: Configuration dictionary with 'type' and 'parameters'

        Raises:
            ValueError: If tool set type is unknown or configuration is invalid
            RuntimeError: If factory fails to create tools
        """
        tool_set_type = tool_config.get("type")

        if not isinstance(tool_set_type, str):
            raise ValueError(f"Tool set '{tool_set_id}' missing required 'type' field")

        if tool_set_type not in TOOL_FACTORIES:
            raise ValueError(f"Unknown tool set '{tool_set_type}'. Available tool sets: {list(TOOL_FACTORIES.keys())}")

        # Get factory class (not instance)
        factory_class = TOOL_FACTORIES[tool_set_type]

        # Instantiate factory
        factory = factory_class()

        # Store factory instance for later reuse
        self.factory_instances[tool_set_id] = factory

        # Get shared parameters
        shared_params = tool_config.get("parameters", {})

        # Create all tools from factory
        try:
            tools_dict = factory.create_all_tools(**shared_params)
        except Exception as e:
            logger.error(
                f"Failed to create tools from factory '{tool_set_type}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Tool set '{tool_set_id}' factory failed: {e}") from e

        # Track tool IDs created by this tool set
        tool_ids = []

        # Register each tool
        for tool_id, tool in tools_dict.items():
            self.register(tool_id, tool)
            tool_ids.append(tool_id)
            logger.debug(f"Registered tool '{tool_id}' from set '{tool_set_type}'")

        # Store tool set mapping
        self.tool_set_registry[tool_set_id] = tool_ids

        logger.info(f"Successfully loaded tool set '{tool_set_id}' ({tool_set_type}): {len(tools_dict)} tools")

    def register(self, tool_id: str, tool: Tool) -> None:
        """
        Register a tool instance with the registry.

        Args:
            tool_id: Unique identifier for the tool
            tool: Tool instance to register

        Raises:
            ValueError: If tool_id is already registered or tool is invalid
        """
        if not isinstance(tool, Tool):
            raise ValueError(f"Tool must be an instance of Tool class, got {type(tool)}")

        if tool_id in self.tools:
            raise ValueError(f"Tool ID '{tool_id}' is already registered")

        self.tools[tool_id] = tool
        logger.debug(f"Registered tool '{tool_id}': {tool.name}")

    def get_tool(self, tool_id: str) -> Tool:
        """
        Get a tool by its ID.

        Args:
            tool_id: Unique identifier for the tool

        Returns:
            Tool: The tool instance

        Raises:
            KeyError: If tool is not found
        """
        if tool_id not in self.tools:
            error_message = f"Unknown tool '{tool_id}'. Available tools: {list(self.get_all_tools().keys())}"
            raise KeyError(error_message)
        return self.tools[tool_id]

    def close_all(self) -> None:
        """
        Close all registered tools.

        This method iterates through all registered tools and calls their
        close() method to release resources.
        """
        for tool_id, tool in self.tools.items():
            try:
                tool.close()
            except Exception as e:
                logger.warning(f"Error closing tool '{tool_id}': {e}")

    def get_tool_ids_by_tool_set(self, tool_set_id: str) -> list[str]:
        """
        Get all tool IDs created by a specific tool set.

        Args:
            tool_set_id: Tool set identifier (e.g., "kopl_schema_free_tools")

        Returns:
            List of tool IDs created by this tool set

        Raises:
            KeyError: If tool set ID not found
        """
        if tool_set_id not in self.tool_set_registry:
            raise KeyError(
                f"Tool set '{tool_set_id}' not found. Available tool sets: {list(self.tool_set_registry.keys())}"
            )
        return self.tool_set_registry[tool_set_id].copy()

    def get_factory_instance(self, tool_set_id: str) -> Any:
        """
        Get the factory instance used to create a specific tool set.

        This allows other components (e.g., AgentRegistry) to reuse the
        factory instance instead of creating a new one.

        Args:
            tool_set_id: Tool set identifier (e.g., "kopl_schema_free_tools")

        Returns:
            Factory instance that created this tool set

        Raises:
            KeyError: If tool set ID not found
        """
        if tool_set_id not in self.factory_instances:
            raise KeyError(
                f"Factory for tool set '{tool_set_id}' not found. "
                f"Available tool sets: {list(self.factory_instances.keys())}"
            )
        return self.factory_instances[tool_set_id]

    def get_all_tools(self) -> dict[str, Tool]:
        """
        Get all registered tools.

        Returns:
            dict[str, Tool]: Dictionary mapping tool IDs to tool instances
        """
        return self.tools.copy()

    def get_tool_names(self) -> list[str]:
        """
        Get the names of all registered tools.

        Returns:
            list[str]: List of tool IDs
        """
        return list(self.tools.keys())

    def get_openai_schema(self) -> list[dict[str, Any]]:
        """
        Generate OpenAI function calling schemas for all registered tools.

        Returns:
            list[dict[str, Any]]: List of OpenAI function calling format schemas
        """
        schemas = []
        for tool_id, tool in self.get_all_tools().items():
            # Get the tool spec and ensure it follows OpenAI format
            spec = tool.spec(tool_id=tool_id)

            # Ensure parameters field exists
            if "parameters" not in spec["function"]:
                spec["function"]["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }

            schemas.append(spec)

        return schemas

    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()
        logger.debug("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self.tools)

    def __contains__(self, tool_id: str) -> bool:
        """Check if a tool is registered."""
        return tool_id in self.tools

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry(tools={list(self.tools.keys())})"
