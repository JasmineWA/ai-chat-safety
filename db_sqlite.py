import json
import os
import sqlite3
from contextlib import contextmanager


DB_FILE = os.getenv("DB_FILE", "ai_chat_safety.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('input', 'output')),
                content TEXT NOT NULL,
                masked_content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS risk_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_log_id INTEGER,
                direction TEXT NOT NULL CHECK(direction IN ('input', 'output')),
                original_text_preview TEXT NOT NULL,
                risk_level TEXT NOT NULL CHECK(risk_level IN ('safe', 'low', 'medium', 'high')),
                risk_category TEXT NOT NULL DEFAULT '',
                risk_subcategory TEXT NOT NULL DEFAULT '',
                score INTEGER NOT NULL DEFAULT 0,
                final_action TEXT NOT NULL CHECK(final_action IN ('pass', 'warn', 'mask', 'replace', 'block')),
                matched_rules TEXT,
                detector_details TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_log_id) REFERENCES chat_logs(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_logs(session_id);
            CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_logs(created_at);
            CREATE INDEX IF NOT EXISTS idx_risk_direction ON risk_logs(direction);
            CREATE INDEX IF NOT EXISTS idx_risk_level ON risk_logs(risk_level);
            CREATE INDEX IF NOT EXISTS idx_risk_category ON risk_logs(risk_category);
            CREATE INDEX IF NOT EXISTS idx_risk_created ON risk_logs(created_at);

            CREATE TABLE IF NOT EXISTS input_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                subcategory TEXT NOT NULL DEFAULT '',
                pattern TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'medium' CHECK(level IN ('safe', 'low', 'medium', 'high')),
                action TEXT NOT NULL DEFAULT 'warn' CHECK(action IN ('pass', 'warn', 'mask', 'replace', 'block')),
                score INTEGER NOT NULL DEFAULT 1,
                template_key TEXT NOT NULL DEFAULT 'default',
                is_keyword INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS semantic_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT NOT NULL DEFAULT '',
                level TEXT NOT NULL DEFAULT 'high',
                action TEXT NOT NULL DEFAULT 'replace',
                score INTEGER NOT NULL DEFAULT 5,
                template_key TEXT NOT NULL DEFAULT 'default'
            );
            """
        )
        conn.execute("DROP TABLE IF EXISTS output_rules")


def insert_chat_log(session_id: str, direction: str, content: str, masked_content: str | None = None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO chat_logs (session_id, direction, content, masked_content) VALUES (?,?,?,?)",
            (session_id, direction, content, masked_content),
        )
        return cur.lastrowid


def insert_risk_log(
    direction: str,
    original_text_preview: str,
    risk_level: str,
    score: int,
    final_action: str,
    matched_rules: list,
    detector_details: dict | None = None,
    chat_log_id: int | None = None,
    risk_category: str = "",
    risk_subcategory: str = "",
):
    matched_rules_json = json.dumps(matched_rules or [], ensure_ascii=False)
    detector_json = json.dumps(detector_details or {}, ensure_ascii=False)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO risk_logs (
                chat_log_id, direction, original_text_preview, risk_level,
                risk_category, risk_subcategory, score, final_action,
                matched_rules, detector_details
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                chat_log_id,
                direction,
                (original_text_preview or "")[:255],
                risk_level,
                risk_category,
                risk_subcategory,
                score,
                final_action,
                matched_rules_json,
                detector_json,
            ),
        )


def fetch_session_messages(session_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.session_id,
                c.direction,
                c.content,
                c.masked_content,
                c.created_at,
                r.risk_level,
                r.risk_category,
                r.risk_subcategory,
                r.score,
                r.final_action,
                r.matched_rules,
                r.detector_details
            FROM chat_logs c
            LEFT JOIN risk_logs r ON r.chat_log_id = c.id
            WHERE c.session_id = ?
            ORDER BY c.id ASC
            """,
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_sessions(limit=100):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.session_id,
                MIN(c.created_at) AS created_at,
                MAX(c.created_at) AS updated_at,
                COUNT(*) AS message_count,
                COALESCE(
                    (
                        SELECT substr(ci.content, 1, 40)
                        FROM chat_logs ci
                        WHERE ci.session_id = c.session_id AND ci.direction = 'input'
                        ORDER BY ci.id ASC
                        LIMIT 1
                    ),
                    '新对话'
                ) AS title
            FROM chat_logs c
            GROUP BY c.session_id
            ORDER BY MAX(c.id) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: str):
    with get_connection() as conn:
        chat_ids = [
            row["id"]
            for row in conn.execute("SELECT id FROM chat_logs WHERE session_id = ?", (session_id,)).fetchall()
        ]
        if chat_ids:
            conn.executemany("DELETE FROM risk_logs WHERE chat_log_id = ?", [(chat_id,) for chat_id in chat_ids])
        conn.execute("DELETE FROM chat_logs WHERE session_id = ?", (session_id,))


def delete_chat_message(message_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT session_id, direction FROM chat_logs WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return None

        conn.execute("DELETE FROM risk_logs WHERE chat_log_id = ?", (message_id,))
        conn.execute("DELETE FROM chat_logs WHERE id = ?", (message_id,))
        return {"session_id": row["session_id"], "direction": row["direction"]}


def clear_all_logs():
    with get_connection() as conn:
        conn.execute("DELETE FROM risk_logs")
        conn.execute("DELETE FROM chat_logs")


def fetch_risk_logs(limit=50, offset=0, direction=None, risk_level=None):
    query = "SELECT * FROM risk_logs"
    params = []
    conditions = []
    if direction:
        conditions.append("direction = ?")
        params.append(direction)
    if risk_level:
        conditions.append("risk_level = ?")
        params.append(risk_level)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_risk_logs(direction=None, risk_level=None):
    query = "SELECT COUNT(*) AS cnt FROM risk_logs"
    params = []
    conditions = []
    if direction:
        conditions.append("direction = ?")
        params.append(direction)
    if risk_level:
        conditions.append("risk_level = ?")
        params.append(risk_level)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    return row["cnt"]


def get_risk_statistics():
    with get_connection() as conn:
        level_rows = conn.execute(
            "SELECT risk_level, COUNT(*) AS cnt FROM risk_logs GROUP BY risk_level"
        ).fetchall()
        direction_rows = conn.execute(
            "SELECT direction, COUNT(*) AS cnt FROM risk_logs GROUP BY direction"
        ).fetchall()
        action_rows = conn.execute(
            "SELECT final_action, COUNT(*) AS cnt FROM risk_logs GROUP BY final_action"
        ).fetchall()
    return {
        "level_distribution": {row["risk_level"]: row["cnt"] for row in level_rows},
        "direction_distribution": {row["direction"]: row["cnt"] for row in direction_rows},
        "action_distribution": {row["final_action"]: row["cnt"] for row in action_rows},
    }


def get_category_distribution():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT risk_category, COUNT(*) AS cnt
            FROM risk_logs
            WHERE risk_category IS NOT NULL AND risk_category != ''
            GROUP BY risk_category
            ORDER BY cnt DESC
            """
        ).fetchall()
    return {row["risk_category"]: row["cnt"] for row in rows}


def _load_json_data(filename: str):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_input_rules():
    data = _load_json_data("rules_input.json")
    extra_test_rules = [
        {
            "rule_id": "SP_PRIV_04",
            "category": "Privacy_And_Property_Risk",
            "subcategory": "privacy_and_property",
            "pattern": "(手机号|手机号码|电话|电话号码|身份证号|身份证号码|邮箱|邮件|email|联系方式).{0,8}[A-Za-z0-9@._%+-]{4,}",
            "level": "medium",
            "action": "mask",
            "score": 3,
            "template_key": "privacy",
            "is_keyword": 0,
        }
    ]
    existing_rule_ids = {item["rule_id"] for item in data}
    for rule in extra_test_rules:
        if rule["rule_id"] not in existing_rule_ids:
            data.append(rule)
    with get_connection() as conn:
        existing_rows = conn.execute("SELECT rule_id FROM input_rules").fetchall()
        existing_ids = {row["rule_id"] for row in existing_rows}
        data_ids = {row["rule_id"] for row in data}

        stale_ids = sorted(existing_ids - data_ids)
        if stale_ids:
            conn.executemany(
                "DELETE FROM input_rules WHERE rule_id = ?",
                [(rule_id,) for rule_id in stale_ids],
            )

        for item in data:
            conn.execute(
                """
                INSERT INTO input_rules (
                    rule_id, category, subcategory, pattern, level,
                    action, score, template_key, is_keyword
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    category=excluded.category,
                    subcategory=excluded.subcategory,
                    pattern=excluded.pattern,
                    level=excluded.level,
                    action=excluded.action,
                    score=excluded.score,
                    template_key=excluded.template_key,
                    is_keyword=excluded.is_keyword
                """,
                (
                    item["rule_id"],
                    item["category"],
                    item.get("subcategory", ""),
                    item["pattern"],
                    item.get("level", "medium"),
                    item.get("action", "warn"),
                    item.get("score", item.get("base_score", 1)),
                    item.get("template_key", "default"),
                    int(bool(item.get("is_keyword", 0))),
                ),
            )
    return len(data)


def seed_semantic_examples():
    data = _load_json_data("semantic_examples.json")
    with get_connection() as conn:
        existing_rows = conn.execute("SELECT example_id FROM semantic_examples").fetchall()
        existing_ids = {row["example_id"] for row in existing_rows}
        data_ids = {row["example_id"] for row in data}

        stale_ids = sorted(existing_ids - data_ids)
        if stale_ids:
            conn.executemany(
                "DELETE FROM semantic_examples WHERE example_id = ?",
                [(example_id,) for example_id in stale_ids],
            )

        for item in data:
            conn.execute(
                """
                INSERT INTO semantic_examples (
                    example_id, text, category, subcategory, level,
                    action, score, template_key
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(example_id) DO UPDATE SET
                    text=excluded.text,
                    category=excluded.category,
                    subcategory=excluded.subcategory,
                    level=excluded.level,
                    action=excluded.action,
                    score=excluded.score,
                    template_key=excluded.template_key
                """,
                (
                    item["example_id"],
                    item["text"],
                    item["category"],
                    item.get("subcategory", ""),
                    item.get("level", "high"),
                    item.get("action", "replace"),
                    item.get("score", 5),
                    item.get("template_key", "default"),
                ),
            )
    return len(data)


def seed_all_rules():
    return {
        "input_rules": seed_input_rules(),
        "semantic_examples": seed_semantic_examples(),
    }


def get_input_rules():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM input_rules ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def get_semantic_examples():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM semantic_examples ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def add_input_rule(rule_id, category, subcategory="", pattern="", level="medium", action="warn", score=1, template_key="default", is_keyword=0):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO input_rules (
                rule_id, category, subcategory, pattern, level,
                action, score, template_key, is_keyword
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (rule_id, category, subcategory, pattern, level, action, score, template_key, is_keyword),
        )


def update_input_rule(rule_id, **kwargs):
    allowed = {"category", "subcategory", "pattern", "level", "action", "score", "template_key", "is_keyword"}
    updates = {key: value for key, value in kwargs.items() if key in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [rule_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE input_rules SET {set_clause} WHERE rule_id = ?", values)


def delete_input_rule(rule_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM input_rules WHERE rule_id = ?", (rule_id,))
