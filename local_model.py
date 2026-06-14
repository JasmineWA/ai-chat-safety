from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence

from logger import get_logger

_logger = get_logger(__name__)


INPUT_LABEL_CONFIG: dict[str, dict[str, Any]] = {
    "Instruction_Attack": {
        "risk_level": "high",
        "risk_subcategory": "instruction_attack",
        "action": "block",
    },
    "Social_Manipulation_And_Harmful_Guidance": {
        "risk_level": "high",
        "risk_subcategory": "social_manipulation",
        "action": "block",
    },
    "Illegal_And_Criminal_Content": {
        "risk_level": "high",
        "risk_subcategory": "illegal_activity",
        "action": "block",
    },
    "Abuse_And_Hate": {
        "risk_level": "medium",
        "risk_subcategory": "abuse_and_hate",
        "action": "replace",
    },
    "Privacy_And_Property_Risk": {
        "risk_level": "medium",
        "risk_subcategory": "privacy_and_property",
        "action": "mask",
    },
    "Mental_Health_Risk": {
        "risk_level": "medium",
        "risk_subcategory": "mental_health",
        "action": "warn",
    },
    "Benign_Safety_Discussion": {
        "risk_level": "safe",
        "risk_subcategory": "benign_safety_discussion",
        "action": "pass",
    },
}

OUTPUT_LABEL_CONFIG: dict[str, dict[str, Any]] = {
    "safe": {
        "risk_level": "safe",
        "risk_subcategory": "safe_output",
        "action": "pass",
    },
    "unsafe": {
        "risk_level": "high",
        "risk_subcategory": "unsafe_output",
        "action": "replace",
    },
}

DEFAULT_INPUT_ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "FineTuning",
    "models",
    "risk_classifier_input_safety_prompts",
    "model.onnx",
)
DEFAULT_OUTPUT_ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "FineTuning",
    "models",
    "risk_classifier_output_pku_zh",
    "model.onnx",
)


class BaseLocalModel:
    def predict(self, text: str, rule_matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


def _load_risk_keywords() -> dict[str, tuple[str, str, int]]:
    try:
        import json as _json

        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "data", "risk_keywords.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        return {k: tuple(v) for k, v in raw.items()}
    except Exception:
        return {}


class HeuristicLocalModel(BaseLocalModel):
    def __init__(self):
        self._keywords_lower = {k.lower(): v for k, v in _load_risk_keywords().items()}

    def predict(self, text: str, rule_matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        text_lower = (text or "").lower()
        keyword_hits: dict[str, int] = {}
        keyword_subcategory: dict[str, str] = {}
        benign_hits = 0

        for keyword, (cat, subcat, severity) in self._keywords_lower.items():
            if keyword in text_lower:
                if cat == "Benign_Safety_Discussion":
                    benign_hits += severity
                    continue
                keyword_hits[cat] = keyword_hits.get(cat, 0) + severity
                keyword_subcategory[cat] = subcat

        if not keyword_hits:
            return {
                "risk_level": "safe",
                "risk_category": "安全",
                "risk_subcategory": "正常对话",
                "score": 0,
                "action": "pass",
                "confidence": 1.0,
            }

        # 濡傛灉鍙槸鑹€у畨鍏ㄨ璁烘垨鏃犲叾浠栭珮椋庨櫓鍛戒腑锛岀洿鎺ユ寜 safe 澶勭悊
        if benign_hits and not keyword_hits:
            return {
                "risk_level": "safe",
                "risk_category": "Benign_Safety_Discussion",
                "risk_subcategory": "benign_safety_discussion",
                "score": 0,
                "action": "pass",
                "confidence": 0.9,
            }

        main_cat = max(keyword_hits, key=keyword_hits.get)
        total_score = keyword_hits[main_cat]
        if total_score <= 1:
            level, action = "low", "warn"
        elif total_score <= 2:
            level, action = "medium", "warn"
        elif total_score <= 4:
            level, action = "medium", "replace"
        else:
            level, action = "high", "replace"

        return {
            "risk_level": level,
            "risk_category": main_cat,
            "risk_subcategory": keyword_subcategory.get(main_cat, ""),
            "score": min(total_score, 5),
            "action": action,
            "confidence": 0.6,
        }


class OnnxLocalModel(BaseLocalModel):
    def __init__(
        self,
        model_path: str | None = None,
        *,
        default_path: str,
        label_config: dict[str, dict[str, Any]],
    ):
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []
        self._id2label: dict[int, str] = {}
        self._loaded = False
        self._load_error: str | None = None
        self._default_path = default_path
        self._label_config = label_config
        self._load(model_path or default_path)

    def _load(self, model_path: str):
        path = model_path.strip() if model_path else self._default_path
        if not os.path.isfile(path):
            self._load_error = f"ONNX model not found: {path}"
            return

        try:
            import json as _json
            import onnxruntime as ort
            from transformers import AutoTokenizer

            model_dir = os.path.dirname(path)
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._input_names = [inp.name for inp in self._session.get_inputs()]

            label_cfg_path = os.path.join(model_dir, "label_config.json")
            if os.path.isfile(label_cfg_path):
                with open(label_cfg_path, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
                self._id2label = {int(k): v for k, v in cfg["ID2LABEL"].items()}
            else:
                self._id2label = {i: label for i, label in enumerate(self._label_config.keys())}

            self._loaded = True
        except Exception as exc:
            self._load_error = f"ONNX load failed: {exc}"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def predict(self, text: str, rule_matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not self._loaded:
            return _fallback_result(f"ONNX model not loaded: {self._load_error}")

        try:
            import numpy as np

            enc = self._tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                padding="max_length",
                max_length=256,
            )
            logits = self._session.run(
                None,
                {
                    self._input_names[0]: enc["input_ids"].astype(np.int64, copy=False),
                    self._input_names[1]: enc["attention_mask"].astype(np.int64, copy=False),
                },
            )[0]
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp / exp.sum(axis=-1, keepdims=True)
            pred_id = int(np.argmax(logits, axis=-1)[0])
            confidence = float(probs[0, pred_id])
            label = self._id2label.get(pred_id, next(iter(self._label_config.keys())))
            cfg = self._label_config.get(label, next(iter(self._label_config.values())))

            return {
                "risk_level": cfg["risk_level"],
                "risk_category": label,
                "risk_subcategory": cfg["risk_subcategory"],
                "score": _risk_score(cfg["risk_level"]),
                "action": cfg["action"],
                "confidence": round(confidence, 4),
            }
        except Exception as exc:
            return _fallback_result(f"ONNX inference failed: {exc}")


def _risk_score(risk_level: str) -> int:
    return {"safe": 0, "low": 1, "medium": 3, "high": 5}.get(risk_level, 0)


def _fallback_result(reason: str = "") -> dict[str, Any]:
    return {
        "risk_level": "safe",
        "risk_category": "安全",
        "risk_subcategory": "正常对话（模型降级）",
        "score": 0,
        "action": "pass",
        "confidence": 0.0,
        "_fallback": True,
        "_reason": reason,
    }


def create_local_model(
    model_path: str | None = None,
    *,
    default_path: str,
    label_config: dict[str, dict[str, Any]],
) -> BaseLocalModel:
    model = OnnxLocalModel(
        model_path,
        default_path=default_path,
        label_config=label_config,
    )
    if model.is_loaded:
        _logger.info("已加载 ONNX 模型 → %s", os.path.basename(model_path or default_path))
        return model
    fallback = HeuristicLocalModel()
    _logger.warning("ONNX 不可用 (%s)，回退到 HeuristicLocalModel", model.load_error)
    return fallback


def create_input_local_model() -> BaseLocalModel:
    model_path = os.getenv("LOCAL_INPUT_MODEL_PATH", "").strip() or None
    return create_local_model(
        model_path,
        default_path=DEFAULT_INPUT_ONNX_PATH,
        label_config=INPUT_LABEL_CONFIG,
    )


def create_output_local_model() -> BaseLocalModel:
    model_path = os.getenv("LOCAL_OUTPUT_MODEL_PATH", "").strip() or None
    return create_local_model(
        model_path,
        default_path=DEFAULT_OUTPUT_ONNX_PATH,
        label_config=OUTPUT_LABEL_CONFIG,
    )
