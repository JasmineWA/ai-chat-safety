from __future__ import annotations

from typing import Any


_LOCAL_MODEL_INSTANCE: Any = None


def reload_rules():
    # 输出侧不再使用规则库，保留空函数兼容现有调用
    return None


def _get_local_model():
    global _LOCAL_MODEL_INSTANCE
    if _LOCAL_MODEL_INSTANCE is None:
        from local_model import create_output_local_model

        _LOCAL_MODEL_INSTANCE = create_output_local_model()
    return _LOCAL_MODEL_INSTANCE


def check_output(
    text: str,
    *,
    user_query: str = "",
    local_model=None,
) -> dict[str, Any]:
    original_text = text or ""
    if local_model is None:
        local_model = _get_local_model()

    model_input = original_text
    if user_query.strip():
        model_input = f"人类问题: {user_query.strip()}\n大模型回答: {original_text}"

    model_result = local_model.predict(model_input, [])
    risk_level = model_result.get("risk_level", "safe")
    action = model_result.get("action", "pass")
    safe_text = original_text if action == "pass" else "该回复可能涉及违法违规或不当内容，系统已进行安全替换。"

    return {
        "direction": "output",
        "has_risk": risk_level != "safe",
        "risk_category": model_result.get("risk_category", "safe"),
        "risk_subcategory": model_result.get("risk_subcategory", ""),
        "risk_level": risk_level,
        "score": model_result.get("score", 0),
        "action": action,
        "matched_rules": [],
        "safe_text": safe_text,
        "message": "输出侧模型检测完成。",
        "detectors": {
            "local_model": {
                "risk_category": model_result.get("risk_category", "safe"),
                "risk_level": risk_level,
                "action": action,
                "score": model_result.get("score", 0),
                "matched_rules": [],
            }
        },
    }
