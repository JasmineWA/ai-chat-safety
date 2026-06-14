"""检测核心模块 — 统一的数据结构、匹配引擎和工具函数。

输入侧和输出侧共用此模块提供的 RiskRule、FastMatcher、风险等级计算、
脱敏函数和安全模板。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

# =========================== 常量 ===========================

LEVEL_ORDER = {"safe": 0, "low": 1, "medium": 2, "high": 3}
ACTION_ORDER = {"pass": 0, "warn": 1, "mask": 2, "replace": 3, "block": 4}

SAFE_RESULT_TEMPLATE = {
    "has_risk": False,
    "risk_category": "安全",
    "risk_subcategory": "无",
    "risk_level": "safe",
    "score": 0,
    "action": "pass",
    "matched_rules": [],
    "safe_text": "",
    "message": "未命中风险规则，已正常放行。",
}


# =========================== 统一规则数据结构 ===========================

@dataclass(frozen=True)
class RiskRule:
    """统一的风险检测规则，输入侧和输出侧共用。"""
    rule_id: str
    category: str
    subcategory: str
    pattern: str
    level: str          # safe / low / medium / high
    action: str         # pass / warn / mask / replace / block
    score: int
    template_key: str = "default"
    flags: int = re.IGNORECASE
    is_keyword: bool = False


# =========================== 安全模板 ===========================

def _load_safety_templates() -> dict[str, str]:
    try:
        import json as _json, os as _os
        base = _os.path.dirname(_os.path.abspath(__file__))
        path = _os.path.join(base, "data", "safety_templates.json")
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {
            "attack": "该回复可能包含可被滥用的攻击方法或恶意代码，系统已进行拦截。",
            "privacy": "该回复中包含疑似个人隐私或敏感凭证，系统已自动脱敏后展示。",
            "fraud": "该回复可能包含诈骗、诱导转账或套取敏感信息的内容，系统已进行拦截。",
            "professional": "该回复涉及医疗、法律或金融等专业领域，仅供参考，建议咨询具备资质的专业人士。",
            "illegal": "该回复可能涉及违法违规或不当内容，系统已进行安全替换。",
            "default": "该回复可能包含不适合直接展示的高风险内容，系统已进行安全处理。",
        }


SAFETY_TEMPLATES = _load_safety_templates()


# =========================== 快速匹配引擎 ===========================

class FastMatcher:
    """多模式合并匹配器，同时支持正则和关键词规则，单次扫描完成全部匹配。"""

    def __init__(self, rules: Sequence[RiskRule]):
        self._keyword_rules: list[tuple[RiskRule, list[str]]] = []
        self._regex_rules: list[RiskRule] = []
        self._combined_pattern: Optional[re.Pattern] = None
        self._group_to_rule: dict[str, RiskRule] = {}
        self._build(rules)

    def _build(self, rules: Sequence[RiskRule]):
        parts: list[str] = []
        for r in rules:
            if r.is_keyword:
                keywords = [k.strip().lower() for k in r.pattern.split("|")]
                self._keyword_rules.append((r, keywords))
            else:
                safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", r.rule_id)
                base_id = safe_id
                n = 1
                while safe_id in self._group_to_rule:
                    safe_id = f"{base_id}_{n}"
                    n += 1
                self._group_to_rule[safe_id] = r
                parts.append(f"(?P<{safe_id}>{r.pattern})")
                self._regex_rules.append(r)

        if parts:
            self._combined_pattern = re.compile("|".join(parts), re.IGNORECASE)

    def search(self, text: str) -> list[RiskRule]:
        """单次扫描文本，返回命中的 RiskRule 列表（去重）。"""
        matched: list[RiskRule] = []
        seen: set[str] = set()
        text_lower = text.lower()

        for rule, keywords in self._keyword_rules:
            for kw in keywords:
                if kw in text_lower:
                    if rule.rule_id not in seen:
                        matched.append(rule)
                        seen.add(rule.rule_id)
                    break

        if self._combined_pattern:
            for m in self._combined_pattern.finditer(text):
                safe_id = m.lastgroup
                if safe_id and safe_id in self._group_to_rule:
                    rule = self._group_to_rule[safe_id]
                    if rule.rule_id not in seen:
                        matched.append(rule)
                        seen.add(rule.rule_id)

        return matched


# =========================== 工具函数 ===========================

def score_to_level(score: int, strongest_level: str) -> str:
    """根据总分和最强规则等级确定最终风险等级。"""
    if LEVEL_ORDER.get(strongest_level, 0) >= LEVEL_ORDER["high"] or score >= 5:
        return "high"
    if LEVEL_ORDER.get(strongest_level, 0) >= LEVEL_ORDER["medium"] or score >= 3:
        return "medium"
    if score >= 1:
        return "low"
    return "safe"


def select_main_rule(rules: list[RiskRule]) -> RiskRule:
    """从命中规则列表中选出最重要的规则。"""
    return max(rules, key=lambda r: (LEVEL_ORDER.get(r.level, 0), ACTION_ORDER.get(r.action, 0), r.score))


def select_rule_action(rules: list[RiskRule], level: str) -> str:
    """根据命中规则和风险等级确定处置动作。"""
    if not rules:
        return "pass"
    strongest_action = max((r.action for r in rules), key=lambda a: ACTION_ORDER.get(a, 0))
    if level == "high" and ACTION_ORDER.get(strongest_action, 0) < ACTION_ORDER["replace"]:
        return "replace"
    return strongest_action


def resolve_template_key(category_or_key: str) -> str:
    """将风险类别映射到安全模板 key。"""
    if category_or_key in SAFETY_TEMPLATES:
        return category_or_key
    mapping = {
        "隐私": "privacy",
        "诈骗": "fraud",
        "攻击": "attack",
        "恶意": "attack",
        "违法": "illegal",
        "暴力": "illegal",
        "专业": "professional",
        "色情": "illegal",
        "虚假": "illegal",
    }
    for kw, key in mapping.items():
        if kw in str(category_or_key):
            return key
    return "default"


def build_safe_text(text: str, action: str, template_key: str = "default",
                    risk_category: str = "") -> str:
    """根据处置动作构建安全文本。"""
    if action == "pass":
        return text
    if action == "warn":
        key = template_key if template_key in SAFETY_TEMPLATES else resolve_template_key(template_key)
        return f"{text}\n\n安全提示：{SAFETY_TEMPLATES.get(key, SAFETY_TEMPLATES['default'])}"
    if action == "mask":
        return mask_sensitive_text(text)
    # replace / block: 用安全模板替换全文
    key = resolve_template_key(template_key or risk_category)
    return SAFETY_TEMPLATES.get(key, SAFETY_TEMPLATES["default"])


def build_message(direction: str, level: str, action: str,
                  risk_category: str = "", risk_subcategory: str = "",
                  confidence: float = 0, model_used: bool = True) -> str:
    """构建人类可读的检测结果消息。"""
    action_text = {"pass": "已放行", "warn": "已提示", "mask": "已脱敏",
                   "replace": "已替换", "block": "已拦截"}.get(action, action)
    direction_text = "输入" if direction == "input" else "输出"
    if model_used:
        return (f"{direction_text}侧两级检测完成：{risk_category}/{risk_subcategory}，"
                f"置信度{confidence:.0%}，处理结果：{action_text}。")
    else:
        return (f"{direction_text}侧规则检测命中：{risk_category}/{risk_subcategory}，"
                f"处理结果：{action_text}。")


# =========================== 脱敏函数 ===========================

def mask_sensitive_text(text: str) -> str:
    """通用正则脱敏：手机号、身份证、银行卡、邮箱、API Key。"""
    masked = text
    # 手机号
    masked = re.sub(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)", r"\1****\3", masked)
    # 身份证号（18位 + 15位）
    masked = re.sub(
        r"(?<!\d)([1-9]\d{5})(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\d{3}[0-9Xx])(?!\d)",
        r"\1********\5", masked)
    masked = re.sub(
        r"(?<!\d)([1-9]\d{5})\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}(?!\d)",
        r"\1******\3", masked)
    # 银行卡
    masked = re.sub(r"(?<!\d)((?:\d[ -]?){13,19})(?!\d)",
                    lambda m: _mask_bank_card(m.group(1)), masked)
    # 邮箱
    masked = re.sub(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
                    r"\1***\2", masked)
    # API Key / Token / Password
    masked = re.sub(
        r"\b(api[_-]?key|secret[_-]?key|access[_-]?token|bearer|password|passwd|pwd)\b(\s*[:=]\s*['\"]?)[A-Za-z0-9_\-./+=]{8,}",
        r"\1\2******", masked, flags=re.IGNORECASE)
    return masked


def _mask_bank_card(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 13 or len(digits) > 19:
        return value
    return f"{digits[:4]} **** **** {digits[-4:]}"


# =========================== 规则命中结果构建 ===========================

def build_rule_result(matched: list[RiskRule], direction: str,
                      text: str = "") -> dict:
    """根据命中规则列表构建统一的规则检测结果字典。"""
    if not matched:
        result = dict(SAFE_RESULT_TEMPLATE)
        result["direction"] = direction
        result["safe_text"] = text
        result["message"] = (
            "AI 回复未命中输出侧风险规则，已正常放行。" if direction == "output"
            else "用户输入未命中输入侧风险规则，已正常放行。"
        )
        return result

    main_rule = select_main_rule(matched)
    score = sum(r.score for r in matched)
    level = score_to_level(score, main_rule.level)
    action = select_rule_action(matched, level)

    return {
        "direction": direction,
        "has_risk": True,
        "risk_category": main_rule.category,
        "risk_subcategory": main_rule.subcategory,
        "risk_level": level,
        "score": score,
        "action": action,
        "matched_rules": [r.rule_id for r in matched],
        "safe_text": "",
        "message": (
            f"命中输出侧规则 {main_rule.rule_id}，"
            f"风险类型为{main_rule.category}/{main_rule.subcategory}。"
            if direction == "output"
            else f"命中输入侧规则 {main_rule.rule_id}，"
            f"风险类型为{main_rule.category}/{main_rule.subcategory}。"
        ),
        "template_key": main_rule.template_key,
        "rule_matches": [
            {
                "rule_id": r.rule_id,
                "category": r.category,
                "subcategory": r.subcategory,
                "level": r.level,
                "score": r.score,
                "action": r.action,
            }
            for r in matched
        ],
    }


# =========================== 两级检测决策 ===========================

def merge_rule_and_model(rule_result: dict, model_result: dict,
                         direction: str, original_text: str = "") -> dict:
    """合并规则检测和本地小模型的结果，采用更高风险的一侧作为主判定。"""
    rule_level = rule_result.get("risk_level", "safe")
    model_level = model_result.get("risk_level", "safe")
    rule_action = rule_result.get("action", "pass")
    model_action = model_result.get("action", "pass")

    rule_rank = (
        LEVEL_ORDER.get(rule_level, 0),
        int(rule_result.get("score", 0)),
        ACTION_ORDER.get(rule_action, 0),
    )
    model_rank = (
        LEVEL_ORDER.get(model_level, 0),
        int(model_result.get("score", 0)),
        ACTION_ORDER.get(model_action, 0),
    )

    primary = model_result if model_rank > rule_rank else rule_result
    secondary = rule_result if primary is model_result else model_result

    final_level = primary.get("risk_level", "safe")
    final_action = primary.get("action", "pass")
    final_category = primary.get("risk_category") or secondary.get("risk_category", "安全")
    final_subcategory = primary.get("risk_subcategory") or secondary.get("risk_subcategory", "无")
    final_score = max(int(rule_result.get("score", 0)), int(model_result.get("score", 0)))
    final_template_key = primary.get("template_key", "default")

    safe_text = build_safe_text(
        original_text or rule_result.get("safe_text", ""),
        final_action,
        final_template_key,
        final_category,
    )

    return {
        "direction": direction,
        "has_risk": final_level != "safe",
        "risk_category": final_category,
        "risk_subcategory": final_subcategory,
        "risk_level": final_level,
        "score": final_score,
        "action": final_action,
        "matched_rules": list(rule_result.get("matched_rules", [])),
        "safe_text": safe_text,
        "message": build_message(
            direction,
            final_level,
            final_action,
            final_category,
            final_subcategory,
            model_result.get("confidence", 0),
        ),
        "detectors": {
            "rule": compact_result(rule_result),
            "local_model": compact_result(model_result),
        },
    }


# =========================== 结果精简 ===========================

def compact_result(result: dict) -> dict:
    """将完整检测结果精简为摘要字典（用于 detectors 嵌套）。"""
    return {
        "has_risk": bool(result.get("has_risk")),
        "risk_category": result.get("risk_category", "安全"),
        "risk_subcategory": result.get("risk_subcategory", "无"),
        "risk_level": result.get("risk_level", "safe"),
        "score": result.get("score", 0),
        "action": result.get("action", "pass"),
        "matched_rules": result.get("matched_rules", []),
    }
