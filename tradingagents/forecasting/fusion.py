"""Fuse the quant model and the agent forecast into one call, deterministically.

The quant model is the only component with a MEASURED directional edge (AUC>0.5);
the agents have none for direction but own ranges, narrative, and contextual
overrides. So the fused direction/confidence is a weighted blend of the two
implied P(up) values — favouring the quant by default — while the agents' expected
price, ranges, and prose are preserved. Where the two disagree on direction, the
fused conviction is trimmed (a real signal to trust the call less).

Pure and testable. The weight is tunable and can later be set per horizon from
each source's measured track-record skill.
"""

from __future__ import annotations

from tradingagents.agents.schemas import FORECAST_HORIZONS, Direction

DEFAULT_QUANT_WEIGHT = 0.6
_FLAT_BAND = 0.015        # fused P(up) within 0.5 +/- this -> Flat
_DISAGREE_CONF_CAP = 55   # cap fused confidence when quant & desk point opposite ways


def agent_implied_p_up(direction: Direction, confidence: int) -> float:
    """Convert an agent's (direction, confidence 0-100) into an implied P(up)."""
    c = confidence / 100.0
    if direction == Direction.UP:
        return c
    if direction == Direction.DOWN:
        return 1.0 - c
    return 0.5  # Flat carries no directional information


def _direction_from_p(p_up: float) -> Direction:
    if p_up > 0.5 + _FLAT_BAND:
        return Direction.UP
    if p_up < 0.5 - _FLAT_BAND:
        return Direction.DOWN
    return Direction.FLAT


def fuse_forecast(forecast, quant_probs: dict[str, dict],
                  weight: float = DEFAULT_QUANT_WEIGHT) -> tuple[object, list[dict]]:
    """Blend the quant P(up) into the agent forecast's direction + confidence (in place).

    Returns ``(forecast, sidebyside)`` where ``sidebyside`` is a per-horizon list of
    {horizon, quant_dir, quant_p, agent_dir, agent_conf, fused_dir, fused_conf,
    disagree} for display and logging. Horizons with no quant prediction are left
    exactly as the agents produced them. Expected price, ranges, and prose untouched.
    """
    sidebyside: list[dict] = []
    for label, _ in FORECAST_HORIZONS:
        agent_dir = getattr(forecast, f"direction_{label}")
        agent_conf = getattr(forecast, f"confidence_{label}")
        q = quant_probs.get(label)
        if not q:
            sidebyside.append({
                "horizon": label, "quant_dir": None, "quant_p": None,
                "agent_dir": agent_dir.value, "agent_conf": agent_conf,
                "fused_dir": agent_dir.value, "fused_conf": agent_conf, "disagree": False,
            })
            continue

        quant_p = q["prob_up"]
        fused_p = weight * quant_p + (1 - weight) * agent_implied_p_up(agent_dir, agent_conf)
        fused_dir = _direction_from_p(fused_p)
        fused_conf = round(100 * max(fused_p, 1 - fused_p))
        disagree = (
            q["direction"] in ("Up", "Down")
            and agent_dir.value in ("Up", "Down")
            and q["direction"] != agent_dir.value
        )
        if disagree:
            fused_conf = min(fused_conf, _DISAGREE_CONF_CAP)

        setattr(forecast, f"direction_{label}", fused_dir)
        setattr(forecast, f"confidence_{label}", fused_conf)
        sidebyside.append({
            "horizon": label, "quant_dir": q["direction"], "quant_p": quant_p,
            "agent_dir": agent_dir.value, "agent_conf": agent_conf,
            "fused_dir": fused_dir.value, "fused_conf": fused_conf, "disagree": disagree,
        })
    return forecast, sidebyside


def render_fusion_block(sidebyside: list[dict]) -> str:
    """Side-by-side Quant / Desk / Fused table with disagreement flags."""
    if not sidebyside:
        return ""
    lines = [
        "**Quant vs Desk vs Fused (⚠ = the model and the desk disagree → "
        "conviction trimmed):**",
        "",
        "| Horizon | Quant model | Desk (agents) | Fused (final) |",
        "| --- | --- | --- | --- |",
    ]
    for r in sidebyside:
        quant = f"{r['quant_dir']} (P={r['quant_p']:.2f})" if r.get("quant_dir") else "n/a"
        agent = f"{r['agent_dir']} ({r['agent_conf']}%)"
        fused = f"{r['fused_dir']} ({r['fused_conf']}%)" + (" ⚠" if r["disagree"] else "")
        lines.append(f"| {r['horizon']} | {quant} | {agent} | {fused} |")
    return "\n".join(lines)
