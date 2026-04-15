import os
import requests
from typing import Optional, Dict, Any

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _build_raw_data_summary(
    release_metrics: Dict[str, Any],
    load_metrics: Optional[Dict[str, Any]],
    followthrough_metrics: Optional[Dict[str, Any]]
) -> str:
    lines = []

    lines.append("=== RELEASE FRAME ===")
    elbow = release_metrics.get("elbow_angle_deg")
    if elbow is not None:
        lines.append(f"- Elbow angle: {elbow:.1f} deg")
    knee = release_metrics.get("knee_angle_deg")
    if knee is not None:
        lines.append(f"- Knee angle: {knee:.1f} deg")
    wrist_above = release_metrics.get("wrist_above_shoulder")
    if wrist_above is not None:
        lines.append(f"- Wrist above shoulder: {'yes' if wrist_above else 'no'}")
    elbow_offset = release_metrics.get("elbow_offset_px")
    flare_threshold = release_metrics.get("flare_threshold_px")
    if elbow_offset is not None and flare_threshold is not None:
        flared = abs(elbow_offset) > flare_threshold
        lines.append(f"- Elbow alignment: {'flared' if flared else 'stacked'}")

    elbow_load = None

    if load_metrics:
        lines.append("")
        lines.append("=== LOAD POSITION ===")
        knee_left = load_metrics.get("knee_angle_left")
        knee_right = load_metrics.get("knee_angle_right")
        if knee_left is not None:
            lines.append(f"- Left knee: {knee_left:.1f} deg")
        if knee_right is not None:
            lines.append(f"- Right knee: {knee_right:.1f} deg")
        elbow_load = load_metrics.get("elbow_angle_deg")
        if elbow_load is not None:
            lines.append(f"- Elbow at load: {elbow_load:.1f} deg")
        hip_square = load_metrics.get("hip_square")
        if hip_square is not None:
            lines.append(f"- Hips square: {'yes' if hip_square else 'no'}")
        balance = load_metrics.get("balance_ok")
        if balance is not None:
            lines.append(f"- Balanced: {'yes' if balance else 'no'}")

        lines.append("")
        lines.append("=== TRANSITION ===")
        knee_angles = [k for k in [knee_left, knee_right] if k is not None]
        knee_load_min = min(knee_angles) if knee_angles else None
        knee_release = release_metrics.get("knee_angle_deg")
        if knee_load_min and knee_release:
            lines.append(f"- Knee extension from load to release: {knee_release - knee_load_min:.1f} deg")
        elbow_release = release_metrics.get("elbow_angle_deg")
        if elbow_load and elbow_release:
            lines.append(f"- Elbow extension from load to release: {elbow_release - elbow_load:.1f} deg")

    if followthrough_metrics:
        lines.append("")
        lines.append("=== FOLLOW THROUGH ===")
        wrist_snapped = followthrough_metrics.get("wrist_snapped")
        if wrist_snapped is not None:
            lines.append(f"- Wrist snap: {'yes' if wrist_snapped else 'no'}")
        elbow_ft = followthrough_metrics.get("elbow_angle_deg")
        if elbow_ft is not None:
            lines.append(f"- Elbow at follow through: {elbow_ft:.1f} deg")
        balance_ft = followthrough_metrics.get("balance_ok")
        if balance_ft is not None:
            lines.append(f"- Balanced after release: {'yes' if balance_ft else 'no'}")

    return "\n".join(lines)


def _call_groq_api(prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": GROQ_MODEL,
        "max_tokens": 500,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a basketball shooting coach. Write a concise coaching report in exactly 3 short paragraphs: "
                    "1) One sentence overall assessment. "
                    "2) Two specific things the player is doing well. "
                    "3) One or two things to improve with one concrete drill or cue. "
                    "Total response must be under 120 words. Be direct and specific. No fluff."
                )
            },
            {"role": "user", "content": prompt}
        ],
    }
    response = requests.post(GROQ_URL, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_feedback(
    release_metrics: Dict[str, Any],
    load_metrics: Optional[Dict[str, Any]] = None,
    followthrough_metrics: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
):
    raw_summary = _build_raw_data_summary(release_metrics, load_metrics, followthrough_metrics)

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return None

    prompt = (
        "Basketball shooting measurements:\n\n"
        f"{raw_summary}\n\n"
        "Write a concise personalized coaching report."
    )

    try:
        coaching = _call_groq_api(prompt, key)
    except requests.exceptions.HTTPError as e:
        print(f"\nGroq API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        print(f"\nError: {e}")
        return None

    return coaching