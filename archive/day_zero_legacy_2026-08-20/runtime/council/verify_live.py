"""Live verify: talk to the running council server and watch the draft evolve.

Usage: python verify_live.py ["first message"] ["second message"]
Reports roster, per-tick draft evolution, and history commit after the
second message (Jeff's living-draft doctrine: draft commits when he speaks).
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8788"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return json.load(r)


def post(path, payload, timeout=15):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def draft_of(st):
    for key in ("response_draft", "draft"):
        if isinstance(st.get(key), str):
            return st[key]
    canon = st.get("canonical") or st.get("canonical_state") or {}
    return str(canon.get("response_draft", ""))


def history_of(st):
    canon = st.get("canonical") or st.get("canonical_state") or {}
    return str(st.get("conversation_history") or canon.get("conversation_history", ""))


def watch(seconds, label):
    print(f"--- watching draft for {seconds}s ({label}) ---")
    end = time.time() + seconds
    last = None
    while time.time() < end:
        try:
            st = get("/api/status")
        except Exception as exc:
            print(f"  (status error: {exc})")
            time.sleep(2)
            continue
        d = draft_of(st)
        if d != last:
            print(f"  tick {st.get('tick'):>5} crown={st.get('consolidator_core')} draft: {d!r}")
            last = d
        time.sleep(1.0)
    return last


def main():
    msg1 = sys.argv[1] if len(sys.argv) > 1 else "How are you?"
    msg2 = sys.argv[2] if len(sys.argv) > 2 else "Thank you"

    st = get("/api/status")
    cores = [(c.get("id"), f"{c.get('size')}D") for c in st.get("cores", [])]
    print("cores:", cores)
    print("running:", st.get("running"), "tick:", st.get("tick"))
    if not st.get("running"):
        try:
            print(post("/api/control", {"action": "start"}, timeout=300))
        except Exception as exc:
            print(f"(start request returned: {exc}; polling until running)")
            for _ in range(60):
                time.sleep(5)
                if get("/api/status").get("running"):
                    break
        time.sleep(2)

    print("send:", repr(msg1))
    post("/api/chat", {"text": msg1})
    watch(45, "turn 1")

    print("send:", repr(msg2))
    post("/api/chat", {"text": msg2})
    watch(25, "turn 2")

    st = get("/api/status")
    print("history:", repr(history_of(st)))


if __name__ == "__main__":
    main()
