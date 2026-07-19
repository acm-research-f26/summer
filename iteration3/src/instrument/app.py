
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, redirect, request, send_file, session, url_for

from config import PATHS, SEED
from src.instrument.storage import append_response, assign_conditions, new_response_id

app = Flask(__name__)
app.secret_key = "iteration3-instrument"  # research shell, session-order only

_STIMULI = None


def stimuli() -> pd.DataFrame:
    global _STIMULI
    if _STIMULI is None:
        _STIMULI = pd.read_csv(PATHS["stimuli"])
    return _STIMULI


def _order_for(pid: str) -> list:
    """Seeded per-participant presentation order (independent of conditions)."""
    pid_hash = int(hashlib.sha256(f"order:{pid}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([SEED, pid_hash])
    ids = stimuli()["screen_id"].tolist()
    rng.shuffle(ids)
    return ids


_PAGE = """<!doctype html><html><head><title>Screen study</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto;
        color: #0b0b0b; background: #fcfcfb; padding: 0 1rem; }}
 img {{ max-width: 100%; border: 1px solid #e1e0d9; }}
 .btn {{ font-size: 1rem; padding: .6rem 1.4rem; margin: .5rem .5rem 0 0;
        cursor: pointer; }}
 .muted {{ color: #898781; font-size: .85rem; }}
</style></head><body>{body}</body></html>"""


@app.route("/")
def index():
    body = """
    <h2>Screen study</h2>
    <p>You will see a series of app/website screenshots. For each one, imagine
    you encountered it while browsing and choose what you would actually do.</p>
    <form method="post" action="/start">
      <label>Participant ID (leave blank to auto-generate):
        <input name="participant_id"></label>
      <button class="btn" type="submit">Begin</button>
    </form>"""
    return _PAGE.format(body=body)


@app.route("/start", methods=["POST"])
def start():
    pid = request.form.get("participant_id", "").strip() or f"p_{uuid.uuid4().hex[:8]}"
    session["pid"] = pid
    session["order"] = _order_for(pid)
    session["idx"] = 0
    return redirect(url_for("screen"))


@app.route("/screen")
def screen():
    if "pid" not in session:
        return redirect(url_for("index"))
    idx, order = session["idx"], session["order"]
    if idx >= len(order):
        body = ("<h2>Done — thank you!</h2><p class='muted'>Your responses "
                "have been recorded.</p>")
        return _PAGE.format(body=body)
    sid = order[idx]
    body = f"""
    <p class="muted">Screen {idx + 1} of {len(order)}</p>
    <img src="/stimulus/{sid}" alt="screenshot">
    <p>Would you accept the offer / continue with what this screen is
    steering you toward?</p>
    <form method="post" action="/respond">
      <input type="hidden" name="screen_id" value="{sid}">
      <input type="hidden" name="shown_at" value="{time.time()}">
      <button class="btn" name="decision" value="1" type="submit">Yes, accept</button>
      <button class="btn" name="decision" value="0" type="submit">No, decline</button>
    </form>"""
    return _PAGE.format(body=body)


@app.route("/respond", methods=["POST"])
def respond():
    pid = session["pid"]
    sid = request.form["screen_id"]
    conditions = assign_conditions(pid, stimuli()["screen_id"].tolist(), SEED)
    cond = conditions[sid]
    row = stimuli().set_index("screen_id").loc[sid]
    path = row["manipulated_path"] if cond == "manipulated" else row["neutralized_path"]
    shown = path if Path(path).exists() else f"placeholder({path})"

    append_response({
        "response_id": new_response_id(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "participant_id": pid,
        "source": "human",
        "screen_id": sid,
        "condition": cond,
        "image_shown": shown,
        "decision": int(request.form["decision"]),
        "rt_ms": int((time.time() - float(request.form["shown_at"])) * 1000),
    })
    session["idx"] += 1
    return redirect(url_for("screen"))


@app.route("/stimulus/<sid>")
def stimulus_image(sid):
    conditions = assign_conditions(session["pid"], stimuli()["screen_id"].tolist(), SEED)
    row = stimuli().set_index("screen_id").loc[sid]
    path = row["manipulated_path"] if conditions[sid] == "manipulated" else row["neutralized_path"]
    if Path(path).exists():
        return send_file(path)
    # serve a labeled placeholder until the neutralized images exist
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">'
        '<rect width="100%" height="100%" fill="#e1e0d9"/>'
        f'<text x="50%" y="50%" text-anchor="middle" fill="#52514e" '
        f'font-family="sans-serif" font-size="16">neutralized placeholder — '
        f"{sid}</text></svg>"
    )
    return app.response_class(svg, mimetype="image/svg+xml")


if __name__ == "__main__":
    print(f"[instrument] stimuli: {PATHS['stimuli']}")
    print(f"[instrument] responses -> {PATHS['responses']}")
    app.run(host="127.0.0.1", port=5000, debug=False)
