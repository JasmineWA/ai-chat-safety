import concurrent.futures
import csv
import io
import json
import os
import re
from uuid import uuid4

from flask import Flask, Response, jsonify, render_template, request

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

import db_sqlite as db
from input_detector import check_input, reload_rules as reload_input_rules
from logger import get_logger
from local_model import create_input_local_model, create_output_local_model
from output_checker_v3 import check_output


logger = get_logger(__name__)
app = Flask(__name__, static_folder="static", template_folder="templates")


DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "30"))
DEEPSEEK_CLIENT = (
    OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=DEEPSEEK_TIMEOUT,
        max_retries=0,
    )
    if OpenAI is not None
    else None
)
DEEPSEEK_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

SYSTEM_PROMPT = (
    "你是一个安全、合规的 AI 助手，禁止生成违法犯罪、隐私泄露、攻击指导、诈骗、仇恨歧视、"
    "自残伤害等高风险内容。请在合规范围内提供帮助。"
)
MAX_HISTORY_ROUNDS = 10
COMMON_SHORT_INPUT_WHITELIST = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "在不在",
    "收到",
    "好的",
    "好",
    "嗯",
    "谢谢",
    "感谢",
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
    "晚安",
    "再见",
    "拜拜",
}
conversation_history: dict[str, list[dict[str, str]]] = {}

input_local_model = create_input_local_model()
output_local_model = create_output_local_model()

IP_BLACKLIST = set(filter(None, os.getenv("IP_BLACKLIST", "").split(",")))
IP_WHITELIST = set(filter(None, os.getenv("IP_WHITELIST", "").split(",")))


def _check_ip_access() -> bool:
    client_ip = request.remote_addr
    if IP_WHITELIST and client_ip not in IP_WHITELIST:
        return False
    if client_ip in IP_BLACKLIST:
        return False
    return True


def _init_app():
    with app.app_context():
        db.init_db()
        db.seed_all_rules()


def _normalize_session_id(value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    return f"session_{uuid4().hex}"


def _restore_history_from_db(session_id: str) -> list[dict[str, str]]:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in db.fetch_session_messages(session_id):
        if row["direction"] == "input":
            content = row.get("masked_content") or row.get("content") or ""
            history.append({"role": "user", "content": content})
        else:
            content = row.get("masked_content") or row.get("content") or ""
            history.append({"role": "assistant", "content": content})
    return history


def _get_history(session_id: str) -> list[dict[str, str]]:
    if session_id not in conversation_history:
        conversation_history[session_id] = _restore_history_from_db(session_id)
    return conversation_history[session_id]


def _trim_history(session_id: str):
    history = conversation_history.get(session_id, [])
    if len(history) > MAX_HISTORY_ROUNDS * 2 + 1:
        conversation_history[session_id] = [history[0]] + history[-(MAX_HISTORY_ROUNDS * 2) :]


def _mask_long_digits(text: str) -> str:
    def repl(match):
        value = match.group(1)
        if len(value) <= 4:
            return value
        if len(value) <= 6:
            return f"{value[:2]}**{value[-2:]}"
        return f"{value[:3]}{'*' * max(2, len(value) - 5)}{value[-2:]}"

    return re.sub(r"(?<!\d)(\d{7,19})(?!\d)", repl, text or "")


def _mask_loose_emails(text: str) -> str:
    def repl(match):
        local_part = match.group(1)
        domain = match.group(2)
        if len(local_part) <= 1:
            return f"{local_part}***@{domain}"
        return f"{local_part[:1]}***@{domain}"

    return re.sub(
        r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)(?![A-Za-z0-9._%+-])",
        repl,
        text or "",
    )


def _mask_generic_token(value: str) -> str:
    if len(value) <= 4:
        return value[:1] + "*" * max(1, len(value) - 1)
    return f"{value[:2]}{'*' * max(2, len(value) - 4)}{value[-2:]}"


def _mask_test_privacy_tokens(text: str) -> str:
    result = text or ""
    result = re.sub(
        r"((?:手机号|手机号码|电话|电话号码|联系方式)\D{0,6})([A-Za-z0-9._%+-]{4,})",
        lambda m: f"{m.group(1)}{_mask_long_digits(m.group(2)) if m.group(2).isdigit() else _mask_generic_token(m.group(2))}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"((?:身份证号|身份证号码)\D{0,6})([A-Za-z0-9._%+-]{4,})",
        lambda m: f"{m.group(1)}{_mask_generic_token(m.group(2))}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"((?:邮箱|邮件|email)\D{0,6})([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)",
        lambda m: f"{m.group(1)}{_mask_loose_emails(m.group(2))}",
        result,
        flags=re.IGNORECASE,
    )
    return result


def _mask_test_privacy_tokens_v2(text: str) -> str:
    result = text or ""

    result = re.sub(
        r"((?:手机号|手机号码|电话|电话号码|联系方式)\D{0,6})([A-Za-z0-9._%+-]{4,})",
        lambda m: f"{m.group(1)}{_mask_generic_token(m.group(2))}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"((?:身份证号|身份证号码)\D{0,6})([A-Za-z0-9._%+-]{4,})",
        lambda m: f"{m.group(1)}{_mask_generic_token(m.group(2))}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"((?:邮箱|邮件|email)\D{0,6})([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)",
        lambda m: f"{m.group(1)}{_mask_loose_emails(m.group(2))}",
        result,
        flags=re.IGNORECASE,
    )
    return result


def _has_test_sensitive_context(text: str) -> bool:
    value = (text or "").strip()
    patterns = [
        r"(?<!\d)1[3-9]\d{9}(?!\d)",
        r"(?<!\d)\d{7,19}(?!\d)",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*",
        r"(身份证|银行卡|密码|验证码|邮箱|手机|手机号|手机号码|电话|电话号码|联系方式|token|api[_-]?key|secret)",
    ]
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)


def _ensure_input_safe_text(original_text: str, input_result: dict) -> str:
    safe_text = input_result.get("safe_text", original_text) or original_text
    if input_result.get("action") == "mask" and safe_text == original_text:
        safe_text = _mask_test_privacy_tokens_v2(_mask_loose_emails(_mask_long_digits(original_text)))
    return safe_text


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").strip().split())


def _is_whitelisted_short_input(text: str) -> bool:
    return _normalize_whitespace(text) in COMMON_SHORT_INPUT_WHITELIST


def _normalize_input_result(original_text: str, input_result: dict) -> dict:
    result = dict(input_result)
    if _is_whitelisted_short_input(original_text):
        result["has_risk"] = False
        result["risk_category"] = "安全"
        result["risk_subcategory"] = "常见短句白名单"
        result["risk_level"] = "safe"
        result["score"] = 0
        result["action"] = "pass"
        result["matched_rules"] = []
        result["safe_text"] = original_text
        result["message"] = "输入命中常见短句白名单，已直接放行。"
        return result

    category = result.get("risk_category", "")
    subcategory = result.get("risk_subcategory", "")

    if category == "Privacy_And_Property_Risk" or subcategory == "privacy_and_property":
        masked_text = _mask_test_privacy_tokens_v2(_mask_loose_emails(_mask_long_digits(original_text)))
        if masked_text != original_text:
            result["action"] = "mask"
            result["safe_text"] = masked_text

    return result


def _has_obvious_sensitive_pattern_for_guard(text: str) -> bool:
    value = (text or "").strip()
    patterns = [
        r"(?<!\d)1[3-9]\d{9}(?!\d)",
        r"(?<!\d)\d{7,19}(?!\d)",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*",
        r"(身份证|银行卡|密码|验证码|邮箱|手机|手机号|token|api[_-]?key|secret)",
    ]
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)


def _apply_short_text_false_positive_guard(original_text: str, input_result: dict) -> dict:
    result = dict(input_result)
    normalized = _normalize_whitespace(original_text)
    if not normalized:
        return result
    if len(normalized) > 12:
        return result
    if result.get("matched_rules"):
        return result
    if _has_test_sensitive_context(normalized):
        return result
    if result.get("risk_level") not in {"low", "medium"}:
        return result
    if result.get("action") not in {"warn", "mask", "replace"}:
        return result

    result["has_risk"] = False
    result["risk_category"] = "安全"
    result["risk_subcategory"] = "短文本误判校正"
    result["risk_level"] = "safe"
    result["score"] = 0
    result["action"] = "pass"
    result["matched_rules"] = []
    result["safe_text"] = original_text
    result["message"] = "输入无规则命中且文本较短，已按常规无风险进行校正。"
    return result


def _build_output_message(blocked_reply: str, output_result: dict | None, raw_text: str = ""):
    if output_result is None:
        return {
            "text": blocked_reply,
            "raw_text": raw_text or blocked_reply,
            "safe_text": blocked_reply,
            "risk_level": None,
            "risk_subcategory": None,
            "action": None,
            "score": None,
        }
    return {
        "text": blocked_reply,
        "raw_text": raw_text or blocked_reply,
        "safe_text": blocked_reply,
        "risk_level": output_result.get("risk_level"),
        "risk_subcategory": output_result.get("risk_subcategory"),
        "action": output_result.get("action"),
        "score": output_result.get("score"),
    }


def generate_ai_reply(user_message: str, session_id: str) -> str:
    if OpenAI is None or DEEPSEEK_CLIENT is None:
        raise RuntimeError("openai package is not installed")

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    history = _get_history(session_id)
    history.append({"role": "user", "content": user_message})
    logger.info("Calling DeepSeek for session=%s model=%s", session_id, DEEPSEEK_MODEL)

    def _do_request():
        return DEEPSEEK_CLIENT.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=history,
            temperature=0.2,
            max_tokens=1024,
            timeout=DEEPSEEK_TIMEOUT,
        )

    try:
        future = DEEPSEEK_EXECUTOR.submit(_do_request)
        response = future.result(timeout=DEEPSEEK_TIMEOUT + 2)
        reply = (response.choices[0].message.content or "").strip()
        result = reply or "抱歉，我暂时无法生成有效回复。"
        history.append({"role": "assistant", "content": result})
        _trim_history(session_id)
        return result
    except Exception as exc:
        history.pop()
        logger.exception("DeepSeek request failed")
        raise RuntimeError(f"DeepSeek request failed: {exc}") from exc


@app.route("/")
def chat_page():
    return render_template("chat.html")


@app.route("/admin")
def admin_page():
    return render_template("admin.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not _check_ip_access():
        return jsonify({"error": "Access denied"}), 403

    payload = request.get_json(silent=True) or {}
    user_input = (payload.get("message") or "").strip()
    session_id = _normalize_session_id(payload.get("session_id"))
    if not user_input:
        return jsonify({"error": "message is required"}), 400

    input_result = check_input(user_input, local_model=input_local_model)
    input_result = _normalize_input_result(user_input, input_result)
    input_result = _apply_short_text_false_positive_guard(user_input, input_result)
    input_safe_text = _ensure_input_safe_text(user_input, input_result)
    input_chat_id = db.insert_chat_log(
        session_id=session_id,
        direction="input",
        content=user_input,
        masked_content=input_safe_text,
    )
    db.insert_risk_log(
        direction="input",
        original_text_preview=user_input,
        risk_level=input_result.get("risk_level", "safe"),
        score=input_result.get("score", 0),
        final_action=input_result.get("action", "pass"),
        matched_rules=input_result.get("matched_rules", []),
        detector_details=input_result.get("detectors", {}),
        risk_category=input_result.get("risk_category", "安全"),
        risk_subcategory=input_result.get("risk_subcategory", ""),
        chat_log_id=input_chat_id,
    )

    input_payload = {
        "text": user_input,
        "safe_text": input_safe_text,
        "risk_level": input_result.get("risk_level"),
        "risk_category": input_result.get("risk_category"),
        "risk_subcategory": input_result.get("risk_subcategory"),
        "action": input_result.get("action"),
        "score": input_result.get("score"),
    }

    if input_result.get("action") == "block" or input_result.get("risk_level") == "high":
        blocked_reply = input_result.get("safe_text") or "检测到高风险输入，系统已拦截。"
        output_chat_id = db.insert_chat_log(
            session_id=session_id,
            direction="output",
            content=blocked_reply,
            masked_content=blocked_reply,
        )
        db.insert_risk_log(
            direction="output",
            original_text_preview=blocked_reply,
            risk_level="safe",
            score=0,
            final_action="pass",
            matched_rules=[],
            detector_details={},
            risk_category="安全",
            risk_subcategory="系统拦截回复",
            chat_log_id=output_chat_id,
        )
        return jsonify(
            {
                "session_id": session_id,
                "input": input_payload,
                "output": None,
                "reply": blocked_reply,
                "blocked": True,
            }
        )

    safe_input_for_model = input_safe_text if input_result.get("action") in {"mask", "replace"} else user_input
    try:
        ai_raw_reply = generate_ai_reply(safe_input_for_model, session_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    output_result = check_output(
        ai_raw_reply,
        user_query=user_input,
        local_model=output_local_model,
    )

    final_action = output_result.get("action", "pass")
    final_reply = output_result.get("safe_text", ai_raw_reply) if final_action in {"replace", "block", "mask", "warn"} else ai_raw_reply

    output_chat_id = db.insert_chat_log(
        session_id=session_id,
        direction="output",
        content=ai_raw_reply,
        masked_content=final_reply,
    )
    db.insert_risk_log(
        direction="output",
        original_text_preview=ai_raw_reply,
        risk_level=output_result.get("risk_level", "safe"),
        score=output_result.get("score", 0),
        final_action=final_action,
        matched_rules=output_result.get("matched_rules", []),
        detector_details=output_result.get("detectors", {}),
        risk_category=output_result.get("risk_category", "安全"),
        risk_subcategory=output_result.get("risk_subcategory", ""),
        chat_log_id=output_chat_id,
    )

    return jsonify(
        {
            "session_id": session_id,
            "input": input_payload,
            "output": {
                "raw_text": ai_raw_reply,
                "safe_text": final_reply,
                "risk_level": output_result.get("risk_level"),
                "risk_subcategory": output_result.get("risk_subcategory"),
                "action": output_result.get("action"),
                "score": output_result.get("score"),
            },
            "reply": final_reply,
            "blocked": False,
        }
    )


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    limit = int(request.args.get("limit", 100))
    keyword = (request.args.get("keyword") or "").strip().lower()
    sessions = db.fetch_sessions(limit=limit)
    if keyword:
        sessions = [
            item
            for item in sessions
            if keyword in (item.get("title", "") or "").lower()
        ]
    return jsonify({"data": sessions})


@app.route("/api/sessions/new", methods=["POST"])
def create_session():
    session_id = _normalize_session_id(None)
    conversation_history[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return jsonify({"session_id": session_id})


@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session_messages(session_id: str):
    rows = db.fetch_session_messages(session_id)
    messages = []
    for row in rows:
        safe_text = row.get("masked_content") or row.get("content") or ""
        display_text = safe_text if row["direction"] == "output" else row.get("content", "")
        try:
            matched_rules = json.loads(row.get("matched_rules") or "[]")
        except Exception:
            matched_rules = []
        messages.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": "user" if row["direction"] == "input" else "assistant",
                "direction": row["direction"],
                "text": display_text,
                "raw_text": row.get("content") or "",
                "safe_text": safe_text,
                "risk_level": row.get("risk_level"),
                "risk_category": row.get("risk_category"),
                "risk_subcategory": row.get("risk_subcategory"),
                "action": row.get("final_action"),
                "score": row.get("score"),
                "matched_rules": matched_rules,
                "created_at": row.get("created_at"),
            }
        )
    return jsonify({"session_id": session_id, "messages": messages})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session_api(session_id: str):
    db.delete_session(session_id)
    conversation_history.pop(session_id, None)
    return jsonify({"message": "Session deleted", "session_id": session_id})


@app.route("/api/messages/<int:message_id>", methods=["DELETE"])
def delete_message_api(message_id: int):
    result = db.delete_chat_message(message_id)
    if result is None:
        return jsonify({"error": "Message not found"}), 404

    session_id = result["session_id"]
    conversation_history.pop(session_id, None)
    return jsonify({"message": "Message deleted", "session_id": session_id, "message_id": message_id})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    direction = request.args.get("direction")
    risk_level = request.args.get("risk_level")
    offset = (page - 1) * per_page

    logs = db.fetch_risk_logs(
        limit=per_page,
        offset=offset,
        direction=direction,
        risk_level=risk_level,
    )
    for log in logs:
        try:
            log["matched_rules"] = json.loads(log.get("matched_rules") or "[]")
        except Exception:
            log["matched_rules"] = []
        try:
            log["detector_details"] = json.loads(log.get("detector_details") or "{}")
        except Exception:
            log["detector_details"] = {}

    total = db.count_risk_logs(direction=direction, risk_level=risk_level)
    return jsonify({"page": page, "per_page": per_page, "total": total, "data": logs})


@app.route("/api/logs/export", methods=["GET"])
def export_logs():
    direction = request.args.get("direction")
    risk_level = request.args.get("risk_level")
    logs = db.fetch_risk_logs(limit=10000, offset=0, direction=direction, risk_level=risk_level)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "time",
            "direction",
            "risk_level",
            "risk_category",
            "risk_subcategory",
            "score",
            "action",
            "preview",
            "matched_rules",
        ]
    )
    for log in logs:
        writer.writerow(
            [
                log.get("id", ""),
                log.get("created_at", ""),
                log.get("direction", ""),
                log.get("risk_level", ""),
                log.get("risk_category", ""),
                log.get("risk_subcategory", ""),
                log.get("score", ""),
                log.get("final_action", ""),
                log.get("original_text_preview", ""),
                log.get("matched_rules", ""),
            ]
        )

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=risk_logs.csv"},
    )


@app.route("/api/logs", methods=["DELETE"])
def clear_logs():
    db.clear_all_logs()
    conversation_history.clear()
    return jsonify({"message": "All logs cleared"})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    stats = db.get_risk_statistics()
    stats["category_distribution"] = db.get_category_distribution()
    return jsonify(stats)


@app.route("/api/rules", methods=["GET"])
def get_rules():
    rule_type = request.args.get("type", "all")
    result = {}
    if rule_type in {"all", "input"}:
        result["input"] = db.get_input_rules()
    if rule_type in {"all", "semantic"}:
        result["semantic"] = db.get_semantic_examples()
    return jsonify(result)


@app.route("/api/rules/reload", methods=["POST"])
def reload_rules_api():
    db.seed_all_rules()
    reload_input_rules()
    return jsonify({"message": "Rules reloaded"})


@app.route("/api/rules/input", methods=["POST"])
def add_input_rule_api():
    data = request.get_json(silent=True) or {}
    try:
        db.add_input_rule(
            rule_id=data["rule_id"],
            category=data["category"],
            subcategory=data.get("subcategory", ""),
            pattern=data["pattern"],
            level=data.get("level", "medium"),
            action=data.get("action", "warn"),
            score=data.get("score", 1),
            template_key=data.get("template_key", "default"),
            is_keyword=data.get("is_keyword", 0),
        )
        reload_input_rules()
        return jsonify({"message": "Input rule added"}), 201
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/rules/input/<rule_id>", methods=["PUT"])
def update_input_rule_api(rule_id: str):
    data = request.get_json(silent=True) or {}
    try:
        db.update_input_rule(rule_id, **data)
        reload_input_rules()
        return jsonify({"message": "Input rule updated"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/rules/input/<rule_id>", methods=["DELETE"])
def delete_input_rule_api(rule_id: str):
    db.delete_input_rule(rule_id)
    reload_input_rules()
    return jsonify({"message": "Input rule deleted"})


_init_app()


if __name__ == "__main__":
    logger.info("Starting AI safety system")
    app.run(debug=True, port=5000)
