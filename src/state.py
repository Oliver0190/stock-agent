"""每日去重状态. state.json 存当天已成功发送的 (job, symbol) 组合."""
import json
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent / "state.json"


def _load() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def already_sent_today(job: str, symbol: str) -> bool:
    state = _load()
    today = datetime.now().strftime("%Y-%m-%d")
    return state.get(job, {}).get(symbol) == today


def mark_sent(job: str, symbol: str) -> None:
    state = _load()
    today = datetime.now().strftime("%Y-%m-%d")
    state.setdefault(job, {})[symbol] = today
    _save(state)


# ---------- 盘中事件去重 (同一事件同一天只发一次) ----------

def event_fired_today(symbol: str, event_id: str) -> bool:
    state = _load()
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{symbol}:{event_id}"
    return state.get("intraday_events", {}).get(key) == today


def mark_event_fired(symbol: str, event_id: str) -> None:
    state = _load()
    today = datetime.now().strftime("%Y-%m-%d")
    state.setdefault("intraday_events", {})[f"{symbol}:{event_id}"] = today
    _save(state)
