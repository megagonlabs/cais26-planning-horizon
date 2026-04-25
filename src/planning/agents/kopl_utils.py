"""
KoPL-specific utilities for processing system outputs and data structures.

This module provides functions to handle KoPL (Knowledge-oriented Programming Language)
data structures, including value classes, entities, and entity facts. These utilities
are used to convert structured KoPL outputs into human-readable strings for evaluation
and display purposes.
"""

import json
import re

# Mapping of KoPL operators to expected value types for disambiguation
## Note: year/date are compatible
OPERATOR_TO_VALUE_TYPE = {
    "filter_str": {"string"},
    "filter_num": {"quantity"},
    "filter_year": {"year", "date"},
    "filter_date": {"year", "date"},
    "qfilter_str": {"string"},
    "qfilter_num": {"quantity"},
    "qfilter_year": {"year", "date"},
    "qfilter_date": {"year", "date"},
}

# Operators that return scalar results (not entity lists)
SCALAR_RESULT_OPERATORS = {
    "count",
    "query_attr_qualifier",
    "query_attr_under_condition",
    "query_attr",
    "query_name",
    "query_relation_qualifier",
    "query_relation",
    "select_among",
    "select_between",
    "verify_date",
    "verify_num",
    "verify_str",
    "verify_year",
}


def is_kopl_value_class(data: dict) -> bool:
    """
    Check if the given dictionary represents a KoPL ValueClass instance.

    A KoPL ValueClass is a dictionary with exactly three keys: "type", "value", and "unit".
    This is used to identify structured data returned by the KoPL system.

    Args:
        data (dict): The dictionary to check.

    Returns:
        bool: True if the dictionary matches the ValueClass structure, False otherwise.

    Example:
        >>> is_kopl_value_class({"type": "string", "value": "example", "unit": None})
        True
    """
    # Must have exactly three keys
    if not isinstance(data, dict):
        return False
    if len(data) != 3:
        return False
    # Must contain the required keys
    if "type" not in data or "value" not in data or "unit" not in data:
        return False
    return True


def kopl_value_class_to_str(data: dict) -> str:
    """
    Convert a KoPL ValueClass dictionary to a human-readable string.

    If the "unit" is not None, it includes the unit in the output; otherwise, just the value.

    Args:
        data (dict): A dictionary representing a KoPL ValueClass.

    Returns:
        str: The string representation of the value, with unit if present.

    Example:
        >>> kopl_value_class_to_str({"type": "number", "value": 42, "unit": "kg"})
        '42 kg'
        >>> kopl_value_class_to_str({"type": "string", "value": "hello", "unit": None})
        'hello'
    """
    value_type = data["type"]
    value = data["value"]
    value_unit = data["unit"]
    if value_type == "quantity":  # quantity-specific formatting
        # x.0+ -> x
        if re.match(r"^-?\d+\.0+$", str(value)):
            value = str(int(float(value)))
    if value_unit is not None and value_unit != "1":
        return f"{value} {value_unit}"
    return str(value)


def is_kopl_entity(data: dict) -> bool:
    """
    Check if the given dictionary represents a KoPL entity with no facts.

    A KoPL entity dictionary has "entities" as a list and "facts" as None.

    Args:
        data (dict): The dictionary to check.

    Returns:
        bool: True if it matches the entity structure, False otherwise.

    Example:
        >>> is_kopl_entity({"entities": ["Q8416"], "facts": None})
        True
    """
    if not isinstance(data, dict):
        return False
    # Must have both keys
    if "entities" not in data or "facts" not in data:
        return False
    # "entities" must be a list
    if not isinstance(data["entities"], list):
        return False
    # "facts" must be None
    if data["facts"] is not None:
        return False
    return True


def kopl_entity_to_str(data: dict) -> str | None:
    """
    Convert a KoPL entity dictionary to a string representation.

    Joins the list of entities into a comma-separated string.

    Args:
        data (dict): A dictionary representing a KoPL entity.

    Returns:
        str: The string "Entities: " followed by the comma-separated entity IDs.

    Example:
        >>> kopl_entity_to_str({"entities": ["Q8416"], "facts": None})
        'Entities: Q8416'
    """
    if len(data["entities"]) == 0:
        return None
    entities_str = ", ".join(data["entities"])
    return f"Entities: {entities_str}"


def is_kopl_entity_facts(data: dict) -> bool:
    """
    Check if the given dictionary represents a KoPL entity with facts.

    A KoPL entity facts dictionary has "entities" as a list and "facts" as a list of fact dictionaries.

    Args:
        data (dict): The dictionary to check.

    Returns:
        bool: True if it matches the entity facts structure, False otherwise.

    Example:
        >>> is_kopl_entity_facts({"entities": ["Q8416"], "facts": [{"key": "name", "value": {...}}]})
        True
    """
    if not isinstance(data, dict):
        return False
    # Must have both keys
    if "entities" not in data or "facts" not in data:
        return False
    # Both must be lists
    if not isinstance(data["entities"], list):
        return False
    if not isinstance(data["facts"], list):
        return False
    return True


def kopl_entity_facts_to_str(data: dict) -> str | None:
    """
    Convert a KoPL entity facts dictionary to a detailed string representation.

    Processes each fact, including its value and any qualifiers, into a formatted string.

    Args:
        data (dict): A dictionary representing a KoPL entity with facts.

    Returns:
        str: A semicolon-separated string of facts, each with key, value, and qualifiers.

    Example:
        Input: {'entities': ['Q8416'], 'facts': [{'key': 'official website', 'value': {'type': 'string', 'value': 'http://example.com', 'unit': None}, 'qualifiers': {'language': [{'type': 'string', 'value': 'English', 'unit': None}]}}]}
        Output: "- official website: http://example.com\n  - [language: English]"
    """
    facts_strs = []
    for fact in data["facts"]:
        # Extract the key of the fact
        is_attribute = False
        if "key" in fact:  # attribute
            is_attribute = True
            key = fact["key"]
        elif "relation" in fact:  # relation
            key = fact["relation"]
        else:
            return None
        # Get the value, defaulting to empty dict
        if is_attribute:
            value = fact.get("value", {})
            # Convert value to string, handling ValueClass if needed
            if is_kopl_value_class(value):
                value_str = kopl_value_class_to_str(value)
            else:
                value_str = str(value)
        else:
            value_str = fact["object"]
        # Start building the fact string
        fact_str = f"- {key}: {value_str}"
        # Process qualifiers if present
        for qualifier_key, qualifier_values in fact.get("qualifiers", {}).items():
            qualifier_strs = []
            for qualifier in qualifier_values:
                # Convert each qualifier to string
                if is_kopl_value_class(qualifier):
                    qualifier_strs.append(kopl_value_class_to_str(qualifier))
                else:
                    qualifier_strs.append(str(qualifier))
            # Append qualifiers to the fact string
            fact_str += f"\n  - [{qualifier_key}: {', '.join(qualifier_strs)}]"
        facts_strs.append(fact_str)
    # Join all facts with semicolons
    return "; ".join(facts_strs)


def postprocess_kopl_answer(answer: str) -> str | None:
    """
    Postprocess the system's answer string to handle KoPL data structures.

    Attempts to parse the answer as JSON and convert KoPL-specific structures to
    readable strings. If parsing fails, returns the original string.

    Args:
        answer (str): The raw answer string from the system output.

    Returns:
        str: The postprocessed answer string, potentially converted from KoPL formats.

    Raises:
        NotImplementedError: If the parsed dict does not match known KoPL structures.
    """
    try:
        # Attempt to evaluate the string as a Python literal
        val = json.loads(answer)
        if len(val) == 0:
            return None
        if isinstance(val, list):
            if is_kopl_value_class(val[0]):
                # List of KoPL ValueClass - convert each to string
                return ", ".join(kopl_value_class_to_str(v) for v in val)
            return ", ".join(str(v) for v in val)
        if isinstance(val, dict):
            # Handle different KoPL dictionary types
            if is_kopl_value_class(val):
                return kopl_value_class_to_str(val)
            elif is_kopl_entity(val):
                return kopl_entity_to_str(val)
            elif is_kopl_entity_facts(val):
                return kopl_entity_facts_to_str(val)
            else:
                # Unknown dict format - raise error for debugging
                raise NotImplementedError("Unhandled dict format")
        # For non-dict literals, convert to string
        return str(val)
    except json.JSONDecodeError:
        # If not a valid literal, return as-is
        return answer
