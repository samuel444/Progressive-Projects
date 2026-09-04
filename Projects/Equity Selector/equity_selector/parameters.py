"""Safe, lossless model-parameter parsing across Python and SQLite/JSON."""

import ast
import copy
import json
import math
from collections.abc import Mapping

import numpy as np


def _literal(node):
    if isinstance(node, ast.Name) and node.id in {"nan", "NaN", "null"}:
        return None
    if isinstance(node, ast.Name) and node.id in {"true", "false"}:
        return node.id == "true"
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_literal(v) for v in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "np"
            and func.attr in {"float64", "float32", "int64", "int32", "bool_"}
            and len(node.args) == 1
            and not node.keywords
        ):
            value = _literal(node.args[0])
            if value is None and func.attr in {"float64", "float32"}:
                return None  # Legacy np.float64(nan) is the nullable-parameter sentinel.
            return {"float64": float, "float32": float, "int64": int, "int32": int, "bool_": bool}[
                func.attr
            ](value)
        raise ValueError("Executable expressions are not model parameters")
    return ast.literal_eval(node)


def parse_parameters(value):
    """Read a dictionary, JSON, or legacy Python repr without executing code.

    Quoted strings are never rewritten. JSON list representations of sklearn's
    tuple-valued hidden_layer_sizes are restored at this boundary.
    """
    if value is None or isinstance(value, (float, np.floating)) and np.isnan(value):
        return {}
    if isinstance(value, Mapping):
        parsed = copy.deepcopy(dict(value))
    elif isinstance(value, str):
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = _literal(ast.parse(value, mode="eval").body)
            except (SyntaxError, TypeError, ValueError) as error:
                raise ValueError("Invalid model parameter dictionary") from error
    else:
        raise ValueError("Model parameters must be a dictionary or dictionary text")
    if not isinstance(parsed, dict):
        raise ValueError("Model parameters must parse to a dictionary")
    result = {}
    for key, value in parsed.items():
        key = str(key).removeprefix("model__")
        if key in result:
            raise ValueError(f"Duplicate normalized parameter: {key}")
        if key == "hidden_layer_sizes" and isinstance(value, list):
            value = tuple(value)
        if key == "class_weight" and isinstance(value, dict):
            # This application uses numeric class labels; JSON stringifies keys.
            value = {
                int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k: v
                for k, v in value.items()
            }
        result[key] = value
    return result


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        raise ValueError("Infinite model parameter")
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def parameters_to_json(value):
    return json.dumps(
        _json_value(parse_parameters(value)), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def configuration_key(model):
    return str(model["name"]), parameters_to_json(model["params"])


def parameter_key(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def same_parameter(left, right):
    return parameter_key(left) == parameter_key(right)


def unique_models(models):
    seen = set()
    result = []
    for model in models:
        key = configuration_key(model)
        if key not in seen:
            seen.add(key)
            result.append(copy.deepcopy(model))
    return result
