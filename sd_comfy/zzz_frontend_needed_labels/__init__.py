"""
Append " (new frontend needed)" to display names of nodes that use
widgets the pinned ComfyUI frontend (e.g. 1.25.10) does not render well.

Drop into ComfyUI/custom_nodes/ (zzz_ prefix preferred).
Disable with FRONTEND_NEEDED_LABELS=0.

Runs once on import (for already-loaded core nodes) and again in a
background thread after NODE_CLASS_MAPPINGS stops growing, because
custom_nodes load order is filesystem order — not alphabetical.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

SUFFIX = " (new frontend needed)"
_INFO_MARKERS = (
    "COMFY_DYNAMICCOMBO_V3",
    "IMAGECOMPARE",
    "AUDIO_RECORD",
)
_INCLUDE_COLOR = os.environ.get("FRONTEND_NEEDED_LABELS_COLOR", "0") == "1"
_EXCLUDE_API = os.environ.get("FRONTEND_NEEDED_LABELS_EXCLUDE_API", "1") != "0"
_ENABLED = os.environ.get("FRONTEND_NEEDED_LABELS", "1") != "0"

_LABELED: set[str] = set()
_LOCK = threading.Lock()


def _label_display_name(name: str | None, node_id: str) -> str:
    base = name or node_id
    if base.endswith(SUFFIX):
        return base
    return f"{base}{SUFFIX}"


def _info_needs_new_frontend(info: dict[str, Any]) -> bool:
    if _EXCLUDE_API and info.get("api_node"):
        return False
    blob = repr(info.get("input", {}))
    if any(m in blob for m in _INFO_MARKERS):
        return True
    if _INCLUDE_COLOR and ("COLOR" in blob or "Color" in blob):
        return True
    return False


def _resolve_define_schema(cls: type) -> Callable:
    for klass in cls.__mro__:
        raw = klass.__dict__.get("define_schema")
        if raw is None:
            continue
        if isinstance(raw, classmethod):
            return raw.__func__
        if isinstance(raw, staticmethod):
            return raw.__func__
        if callable(raw):
            return raw
    raise AttributeError(f"{cls.__name__} has no define_schema")


def _wrap_define_schema(cls: type, node_id: str) -> None:
    if getattr(cls, "_frontend_needed_label_wrapped", False):
        return
    original = _resolve_define_schema(cls)

    @classmethod
    def define_schema(wrapped_cls):  # type: ignore[no-untyped-def]
        schema = original(wrapped_cls)
        schema.display_name = _label_display_name(
            schema.display_name, getattr(schema, "node_id", None) or node_id
        )
        return schema

    cls.define_schema = define_schema
    cls._frontend_needed_label_wrapped = True
    cls.SCHEMA = None


def apply_frontend_needed_labels() -> int:
    import nodes

    labeled = 0
    with _LOCK:
        for node_id, cls in list(nodes.NODE_CLASS_MAPPINGS.items()):
            try:
                if node_id in _LABELED:
                    continue

                needs = False
                if hasattr(cls, "GET_NODE_INFO_V1"):
                    info = cls.GET_NODE_INFO_V1()
                    if not isinstance(info, dict):
                        info = {
                            "input": getattr(info, "input", {}),
                            "display_name": getattr(info, "display_name", None),
                            "api_node": getattr(info, "api_node", False),
                            "name": node_id,
                        }
                    needs = _info_needs_new_frontend(info)
                    if needs:
                        _wrap_define_schema(cls, node_id)
                        info2 = cls.GET_NODE_INFO_V1()
                        if isinstance(info2, dict):
                            display = info2.get("display_name") or _label_display_name(
                                info.get("display_name"), node_id
                            )
                        else:
                            display = _label_display_name(
                                getattr(info2, "display_name", None) or info.get("display_name"),
                                node_id,
                            )
                        nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] = display
                elif hasattr(cls, "INPUT_TYPES"):
                    if _EXCLUDE_API and getattr(cls, "API_NODE", False):
                        continue
                    blob = repr(cls.INPUT_TYPES())
                    needs = any(m in blob for m in _INFO_MARKERS) or (
                        _INCLUDE_COLOR and "COLOR" in blob
                    )
                    if needs:
                        display = _label_display_name(
                            nodes.NODE_DISPLAY_NAME_MAPPINGS.get(node_id), node_id
                        )
                        nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] = display

                if needs:
                    _LABELED.add(node_id)
                    labeled += 1
            except Exception as e:
                logging.warning("frontend_needed_labels: skip %s: %s", node_id, e)

    return labeled


def _deferred_pass() -> None:
    import time
    import nodes

    prev = -1
    stable = 0
    for _ in range(180):
        time.sleep(1)
        cur = len(nodes.NODE_CLASS_MAPPINGS)
        if cur == prev and cur > 0:
            stable += 1
            if stable >= 5:
                break
        else:
            stable = 0
            prev = cur

    n = apply_frontend_needed_labels()
    sample = ", ".join(sorted(_LABELED)[:8])
    more = f" …(+{len(_LABELED)-8})" if len(_LABELED) > 8 else ""
    logging.info(
        "frontend_needed_labels: deferred pass +%s (total %s) [%s%s]",
        n,
        len(_LABELED),
        sample,
        more,
    )


if _ENABLED:
    try:
        n = apply_frontend_needed_labels()
        logging.info("frontend_needed_labels: initial pass tagged %s node(s)", n)
        threading.Thread(target=_deferred_pass, name="frontend_needed_labels", daemon=True).start()
    except Exception as e:
        logging.warning("frontend_needed_labels: failed to apply: %s", e)
else:
    logging.info("frontend_needed_labels: disabled (FRONTEND_NEEDED_LABELS=0)")
