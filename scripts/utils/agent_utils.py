"""
Shared utilities for Meta Agent initialization and execution.

This module provides common functions used by both example_run_kqa_pro.py and run.py
scripts to avoid code duplication.
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml

from planning.agents.meta_agents.meta_sh import SHMetaAgent
from planning.agents.meta_agents.meta_fh import FHMetaAgent
from planning.environment import Environment
from planning.services.llm_registry import LLMProviderRegistry


def _register_omegaconf_resolvers() -> None:
    """Register custom OmegaConf resolvers used by the project config."""
    OmegaConf.register_new_resolver(
        "basename",
        lambda path: path.split("/")[-1] if path else "",
        replace=True,
    )


def _load_hydra_experiment_config(config_path: Path) -> dict[str, Any]:
    """Compose a full experiment config from a Hydra experiment file path."""
    project_root = Path(__file__).resolve().parents[2]
    conf_dir = project_root / "conf"
    experiment_dir = conf_dir / "experiment"
    experiment_name = config_path.relative_to(experiment_dir).with_suffix("").as_posix()

    _register_omegaconf_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment_name}"])

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise ValueError(f"Expected dict config from Hydra compose, got {type(resolved_cfg)}")
    return resolved_cfg


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file or compose a Hydra experiment config.

    Args:
        config_path: Path to YAML config

    Returns:
        dict[str, Any]: Parsed configuration dictionary
    """
    config_path = config_path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = project_root / "conf" / "experiment"

    if config_path.is_relative_to(experiment_dir):
        return _load_hydra_experiment_config(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm_registry_from_config(cfg: dict[str, Any]) -> LLMProviderRegistry:
    """
    Create an LLM provider registry from configuration.

    Args:
        cfg: Configuration dictionary containing llm_providers section

    Returns:
        LLMProviderRegistry: Configured provider registry

    Raises:
        RuntimeError: If no llm_providers configuration found
    """
    load_dotenv(override=True)
    providers_cfg = cfg.get("llm_providers", {})
    if not providers_cfg:
        raise RuntimeError("No llm_providers configuration found")

    return LLMProviderRegistry.from_config(providers_cfg)


def create_meta_agent(
    agent_type: str,
    cfg: dict[str, Any],
    llm_call,
) -> SHMetaAgent | FHMetaAgent:
    """
    Factory function to create Meta Agent based on type.

    Args:
        agent_type: Type of agent ('sh' or 'fh')
        cfg: Configuration dictionary
        llm_call: LLM call function

    Returns:
        BaseMetaAgent: Instantiated Meta Agent

    Raises:
        ValueError: If agent_type is unknown or required config is missing
    """
    meta_config = cfg.get("meta_agent", {})

    base_kwargs = {
        "llm_call": llm_call,
        "llm_options": meta_config.get("model", {}),
        "name": meta_config.get("name", "MetaAgent"),
        "max_steps": meta_config.get("max_steps", 10),
        "max_retries": meta_config.get("max_retries", 3),
        "tool_formatter_id": meta_config.get("tool_formatter_id", "json"),
    }

    if agent_type == "sh":
        plan_step_params = meta_config.get("generation", {}).get("plan_step")
        if plan_step_params is None:
            raise ValueError("SH config requires generation.plan_step")
        return SHMetaAgent(
            **base_kwargs,
            use_builtin_tool_input=meta_config.get("use_builtin_tool_input", False),
            use_builtin_tool_output=meta_config.get("use_builtin_tool_output", False),
            plan_step_params=plan_step_params,
        )
    elif agent_type == "fh":
        generation = meta_config.get("generation", {})
        plan_params = generation.get("plan")
        revision_params = generation.get("revision")
        return FHMetaAgent(
            **base_kwargs,
            use_builtin_tool_input=meta_config.get("use_builtin_tool_input", False),
            plan_params=plan_params,
            revision_params=revision_params,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Expected 'sh' or 'fh'.")


def setup_environment(
    cfg: dict[str, Any],
    llm_registry: LLMProviderRegistry,
    debug: bool = False,
) -> Environment:
    """
    Create and configure environment from configuration.

    Args:
        cfg: Configuration dictionary
        llm_registry: LLM provider registry for agent assignment
        debug: Enable debug mode

    Returns:
        Environment: Configured environment with agents
    """
    env = Environment(debug=debug)
    env_config = cfg.get("environment", {})
    env.load_from_config(env_config)
    env.assign_llm_calls(llm_registry)
    return env


def get_meta_agent_llm_call(
    cfg: dict[str, Any],
    llm_registry: LLMProviderRegistry,
):
    """
    Get LLM call function for Meta Agent from configuration.

    Args:
        cfg: Configuration dictionary
        llm_registry: LLM provider registry

    Returns:
        LLM call function for Meta Agent
    """
    meta_agent_cfg = cfg.get("meta_agent", {})
    meta_llm_provider = meta_agent_cfg.get("llm_provider", "default")
    return llm_registry.get_call(meta_llm_provider)
