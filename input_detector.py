"""输入侧内容安全检测器 v5。

v5 统一架构：
1. 与输出侧共用 detection_core 中的 RiskRule、FastMatcher、工具函数
2. 两级检测：规则快速匹配 → 本地小模型二次过滤
3. 规则词库从 SQLite 数据库加载，支持运行时增删改查
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from detection_core import (
    RiskRule, FastMatcher,
    build_rule_result, merge_rule_and_model, compact_result,
)


# =========================== 规则加载 ===========================

def _load_input_rules_from_db() -> list[RiskRule]:
    try:
        import db_sqlite as db
        db.seed_input_rules()
        rows = db.get_input_rules()
        if rows:
            return [
                RiskRule(
                    rule_id=r["rule_id"],
                    category=r["category"],
                    subcategory=r.get("subcategory", ""),
                    pattern=r["pattern"],
                    level=r.get("level", "medium"),
                    action=r["action"],
                    score=r.get("score", r.get("base_score", 1)),
                    template_key=r.get("template_key", "default"),
                    is_keyword=bool(r.get("is_keyword", 0)),
                )
                for r in rows
            ]
    except Exception:
        pass
    return _fallback_input_rules()


def _fallback_input_rules() -> list[RiskRule]:
    try:
        import json as _json, os as _os
        base = _os.path.dirname(_os.path.abspath(__file__))
        path = _os.path.join(base, "data", "rules_input.json")
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return [
            RiskRule(
                rule_id=r["rule_id"],
                category=r["category"],
                subcategory=r.get("subcategory", ""),
                pattern=r["pattern"],
                level=r.get("level", "medium"),
                action=r["action"],
                score=r.get("score", r.get("base_score", 1)),
                template_key=r.get("template_key", "default"),
                is_keyword=bool(r.get("is_keyword", 0)),
            )
            for r in data
        ]
    except Exception:
        return []


# =========================== 全局缓存 ===========================

_RULES_CACHE: Optional[list[RiskRule]] = None
_MATCHER_CACHE: Optional[FastMatcher] = None
_RULES_LOCK = threading.RLock()


def _get_rules() -> list[RiskRule]:
    global _RULES_CACHE
    if _RULES_CACHE is None:
        with _RULES_LOCK:
            if _RULES_CACHE is None:
                _RULES_CACHE = _load_input_rules_from_db()
    return _RULES_CACHE


def _get_matcher() -> FastMatcher:
    global _MATCHER_CACHE
    if _MATCHER_CACHE is None:
        with _RULES_LOCK:
            if _MATCHER_CACHE is None:
                _MATCHER_CACHE = FastMatcher(_get_rules())
    return _MATCHER_CACHE


def reload_rules():
    global _RULES_CACHE, _MATCHER_CACHE
    with _RULES_LOCK:
        try:
            import db_sqlite as db
            db.seed_input_rules()
        except Exception:
            pass
        _RULES_CACHE = _load_input_rules_from_db()
        _MATCHER_CACHE = FastMatcher(_RULES_CACHE)


# =========================== 本地小模型 ===========================

_local_model_instance: Any = None


def _get_local_model():
    global _local_model_instance
    if _local_model_instance is None:
        from local_model import create_input_local_model
        _local_model_instance = create_input_local_model()
    return _local_model_instance


# =========================== 主检测入口 ===========================

def check_input(
    text: str,
    *,
    local_model=None,
) -> dict[str, Any]:
    """执行输入侧两级检测：规则快速匹配 → 本地小模型二次过滤。

    Args:
        text: 用户输入原文
        local_model: 本地小模型实例（可选，未传入则自动创建）

    Returns:
        统一 result 字典，包含 risk_level、action、safe_text、matched_rules 等
    """
    original_text = text or ""

    # 第一级：规则快速匹配
    matcher = _get_matcher()
    matched = matcher.search(original_text)
    rule_result = build_rule_result(matched, "input", original_text)

    # 第二级：本地小模型二次过滤（即使规则未命中，也让模型参与判定）
    if local_model is None:
        local_model = _get_local_model()
    rule_matches = rule_result.get("rule_matches", [])
    model_result = local_model.predict(original_text, rule_matches)

    return merge_rule_and_model(rule_result, model_result, "input", original_text)
