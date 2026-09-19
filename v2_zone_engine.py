"""
ProactiveGuard v2.0 — Module 6: Zone Intelligence Engine
=========================================================
Zone-level microclimate modeling and coordinated risk detection.

Features:
- Per-zone WBGT (not one number for whole site)
- Coordinated spike detection (environmental event vs individual)
- Worker rotation optimization across zones
- Zone evacuation decision engine
- Predictive WBGT 30min ahead using solar position

Run: python v2_zone_engine.py
Output: outputs/v2_zone_analysis.png
        data/v2_zone_risk.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False

os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SITE, ZONE, AND WORKER DEFINITIONS
# Imported from v2_config.py — the single source of truth shared by
# the data engine, the LLM engine, the zone engine, and the dashboard.
# Do not redeclare these here; edit v2_config.py instead.
# ══════════════════════════════════════════════════════════════
from v2_config import (SITES, ZONE_WBGT_OFFSET, ZONE_CAPACITY,
                        WORKERS_BY_ID as WORKER_PROFILES)


# ══════════════════════════════════════════════════════════════
# ZONE MICROCLIMATE WBGT
# ══════════════════════════════════════════════════════════════
def calculate_zone_wbgt(base_wbgt, zone, solar_radiation=800):
    """
    Calculates zone-specific WBGT from site base reading.
    Different zones on same site have different radiant loads.
    This is what separates real-world deployment from theory.
    """
    offset = ZONE_WBGT_OFFSET.get(zone, 0.0)
    # Solar radiation amplifies the offset
    solar_factor = np.clip(solar_radiation / 800, 0.3, 1.5)
    zone_wbgt = base_wbgt + offset * solar_factor
    return float(np.clip(zone_wbgt, 20, 52))


def get_site_zone_wbgt_map(base_wbgt, solar_radiation):
    """Returns WBGT for all zones at current conditions."""
    return {
        zone: calculate_zone_wbgt(base_wbgt, zone, solar_radiation)
        for zone in ZONE_WBGT_OFFSET.keys()
    }


# ══════════════════════════════════════════════════════════════
# PREDICTIVE WBGT — 30 MINUTES AHEAD
# Current systems react to current WBGT.
# We predict what WBGT will be in 30 minutes using
# solar position trajectory. No competitor does this.
# ══════════════════════════════════════════════════════════════
def predict_wbgt_30min(current_wbgt, site_id, current_hour_float,
                        current_humidity, steps=30):
    """
    Predicts WBGT trajectory for next 30 minutes.
    Uses solar altitude change rate to project environmental heat.
    """
    predictions = []
    wbgt = current_wbgt
    humidity = current_humidity

    for step in range(1, steps + 1):
        future_hour = current_hour_float + step / 60

        if EPHEM_AVAILABLE:
            try:
                site = SITES.get(site_id, SITES["S001"])
                obs = ephem.Observer()
                obs.lat = site["lat"]
                obs.lon = site["lon"]
                h = int(future_hour)
                m = int((future_hour - h) * 60)
                obs.date = f"2024/7/15 {h:02d}:{m:02d}:00"
                sun = ephem.Sun()
                sun.compute(obs)
                alt = float(sun.alt) * 180 / np.pi
                solar = float(np.clip(950 * (np.cos(np.radians(90 - alt)) ** 0.6)
                                      if alt > 0 else 0, 0, 1050))
            except Exception:
                solar = max(0, 800 * np.sin(np.pi * (future_hour - 6) / 10))
        else:
            solar = max(0, 800 * np.sin(np.pi * (future_hour - 6) / 10))

        # WBGT rises with solar radiation
        ambient_delta = solar / 800 * 0.08
        wbgt = float(np.clip(wbgt + ambient_delta + np.random.normal(0, 0.05),
                             20, 52))
        predictions.append({
            "minute_ahead": step,
            "predicted_wbgt": round(wbgt, 2),
            "predicted_solar": round(float(solar), 1),
        })

    return predictions


# ══════════════════════════════════════════════════════════════
# COORDINATED RISK DETECTION
# Novel feature — no competitor has this.
# 3+ workers in same zone spiking = environmental event.
# ══════════════════════════════════════════════════════════════
def detect_coordinated_risk(worker_risks, worker_zones,
                             threshold=60, min_workers=3):
    """
    Detects zone-level environmental events.

    Individual pattern:  1-2 workers elevated = personal physiology
    Environmental event: 3+ workers same zone elevated simultaneously

    This distinction is critical:
    - Individual: send worker for rest break
    - Environmental: evacuate entire zone
    """
    zone_elevated = {}
    for wid, risk in worker_risks.items():
        zone = worker_zones.get(wid, "unknown")
        if zone not in zone_elevated:
            zone_elevated[zone] = []
        if risk >= threshold:
            zone_elevated[zone].append({"worker_id": wid, "risk": risk})

    events = []
    for zone, elevated_workers in zone_elevated.items():
        if len(elevated_workers) >= min_workers:
            avg_risk  = np.mean([w["risk"] for w in elevated_workers])
            max_risk  = max(w["risk"] for w in elevated_workers)
            severity  = "critical" if max_risk >= 80 else "high"
            events.append({
                "zone":              zone,
                "event_type":        "environmental",
                "severity":          severity,
                "n_workers":         len(elevated_workers),
                "workers":           elevated_workers,
                "avg_risk":          round(float(avg_risk), 1),
                "max_risk":          round(float(max_risk), 1),
                "action":            (
                    f"ZONE EVACUATION: {len(elevated_workers)} workers in "
                    f"{zone.replace('_',' ')} simultaneously elevated. "
                    f"Environmental cause suspected. "
                    f"Evacuate zone immediately and inspect for hazards."
                ),
            })

    return events


# ══════════════════════════════════════════════════════════════
# WORKER ROTATION OPTIMIZER
# Recommends optimal zone rotation to reduce heat exposure.
# Balances productivity (workers stay on task) with safety.
# ══════════════════════════════════════════════════════════════
def optimize_worker_rotation(worker_risks, worker_zones,
                              zone_wbgt_map, minutes_on_shift):
    """
    Recommends zone rotations for workers at elevated risk.

    Strategy:
    - Workers at risk >= 55: move to cooler zone
    - Workers at risk >= 70: immediate rest in shade/indoor
    - New workers (accl_days < 14): proactive rotation at risk >= 45
    - Respect zone capacity limits
    """
    recommendations = []
    zone_occupancy = {}
    for wid, zone in worker_zones.items():
        zone_occupancy[zone] = zone_occupancy.get(zone, 0) + 1

    for wid, risk in worker_risks.items():
        wp           = WORKER_PROFILES.get(wid, {})
        current_zone = worker_zones.get(wid, "direct_sun")
        accl         = wp.get("accl_days", 30)
        name         = wp.get("name", wid)

        # Determine if rotation needed
        rotation_threshold = 45 if accl < 14 else 55
        if risk < rotation_threshold:
            continue

        # Find best available cooler zone
        current_wbgt = zone_wbgt_map.get(current_zone, 35)
        best_zone    = current_zone
        best_wbgt    = current_wbgt

        for target_zone, target_wbgt in sorted(
                zone_wbgt_map.items(), key=lambda x: x[1]):
            if target_zone == current_zone:
                continue
            capacity = ZONE_CAPACITY.get(target_zone, 5)
            occupancy = zone_occupancy.get(target_zone, 0)
            if occupancy < capacity and target_wbgt < current_wbgt:
                best_zone = target_zone
                best_wbgt = target_wbgt
                break

        if best_zone != current_zone:
            wbgt_reduction = current_wbgt - best_wbgt
            urgency = "IMMEDIATE" if risk >= 70 else "SOON" if risk >= 55 else "PROACTIVE"
            recommendations.append({
                "worker_id":        wid,
                "worker_name":      name,
                "current_zone":     current_zone,
                "recommended_zone": best_zone,
                "current_wbgt":     round(current_wbgt, 1),
                "target_wbgt":      round(best_wbgt, 1),
                "wbgt_reduction":   round(wbgt_reduction, 1),
                "risk_score":       risk,
                "urgency":          urgency,
                "rationale":        (
                    f"Move {name} from {current_zone.replace('_',' ')} "
                    f"(WBGT {current_wbgt:.1f}°C) to "
                    f"{best_zone.replace('_',' ')} "
                    f"(WBGT {best_wbgt:.1f}°C). "
                    f"Reduces heat exposure by {wbgt_reduction:.1f}°C."
                ),
            })

    return sorted(recommendations, key=lambda x: x["risk_score"], reverse=True)


# ══════════════════════════════════════════════════════════════
# ZONE EVACUATION ENGINE
# Decides when to evacuate a zone vs individual intervention.
# ══════════════════════════════════════════════════════════════
def evaluate_zone_evacuation(zone, n_workers_elevated,
                              avg_risk, max_risk,
                              wbgt, coordinated_event):
    """
    Returns evacuation decision with justification.
    Three levels: Monitor | Partial | Full Evacuation
    """
    if coordinated_event and max_risk >= 75:
        return {
            "decision":      "FULL_EVACUATION",
            "urgency":       "IMMEDIATE",
            "justification": (
                f"Zone {zone}: {n_workers_elevated} workers simultaneously "
                f"at critical risk. Environmental hazard confirmed. "
                f"Full zone evacuation required immediately."
            ),
        }
    elif coordinated_event and avg_risk >= 60:
        return {
            "decision":      "PARTIAL_EVACUATION",
            "urgency":       "WITHIN_10_MIN",
            "justification": (
                f"Zone {zone}: Multiple workers elevated simultaneously. "
                f"Rotate high-risk workers to shade. "
                f"Investigate environmental cause."
            ),
        }
    elif max_risk >= 85:
        return {
            "decision":      "INDIVIDUAL_EVACUATION",
            "urgency":       "IMMEDIATE",
            "justification": (
                f"One or more workers at critical risk (>{max_risk:.0f}). "
                f"Remove from zone immediately. Medical assessment required."
            ),
        }
    else:
        return {
            "decision":      "MONITOR",
            "urgency":       "ROUTINE",
            "justification": f"Zone {zone}: Elevated but not critical. Increase monitoring frequency.",
        }


# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════
def plot_zone_analysis(zone_wbgt_map, worker_risks,
                        worker_zones, rotation_recs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("ProactiveGuard v2.0 — Zone Intelligence Analysis",
                 color="#ecf0f1", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="#95a5a6", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2c3e50")
        ax.grid(True, alpha=0.15, color="#2c3e50")

    # Panel 1: Zone WBGT comparison
    ax1 = axes[0]
    zones  = list(zone_wbgt_map.keys())
    wbgts  = [zone_wbgt_map[z] for z in zones]
    colors = ["#e74c3c" if w > 35 else "#f39c12" if w > 32
              else "#2ecc71" for w in wbgts]
    bars = ax1.bar([z.replace("_", "\n") for z in zones],
                   wbgts, color=colors, alpha=0.85)
    ax1.axhline(32, color="#f39c12", linestyle="--",
                alpha=0.7, linewidth=1, label="Danger threshold")
    ax1.axhline(28, color="#2ecc71", linestyle="--",
                alpha=0.5, linewidth=1, label="Caution threshold")
    for bar, w in zip(bars, wbgts):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{w:.1f}°C", ha="center", color="#ecf0f1", fontsize=8)
    ax1.set_ylabel("WBGT (°C)", color="#ecf0f1", fontsize=9)
    ax1.set_title("Zone Microclimate WBGT\n(same site, different heat loads)",
                  color="#ecf0f1", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, labelcolor="#95a5a6",
               facecolor="#1e2130", edgecolor="#2c3e50")

    # Panel 2: Worker risk by zone
    ax2 = axes[1]
    zone_list  = sorted(set(worker_zones.values()))
    zone_risks = {z: [] for z in zone_list}
    for wid, risk in worker_risks.items():
        zone = worker_zones.get(wid, "direct_sun")
        zone_risks[zone].append(risk)

    x_pos = np.arange(len(zone_list))
    for i, zone in enumerate(zone_list):
        risks = zone_risks.get(zone, [0])
        for j, r in enumerate(risks):
            color = ("#e74c3c" if r >= 70 else "#f39c12"
                     if r >= 55 else "#2ecc71")
            ax2.scatter(i + np.random.uniform(-0.2, 0.2),
                        r, color=color, s=80, alpha=0.85, zorder=3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([z.replace("_", "\n") for z in zone_list],
                        color="#ecf0f1", fontsize=8)
    ax2.axhline(70, color="#e74c3c", linestyle="--",
                alpha=0.6, linewidth=1, label="Danger (70)")
    ax2.axhline(55, color="#f39c12", linestyle="--",
                alpha=0.5, linewidth=1, label="Warning (55)")
    ax2.set_ylabel("Worker Risk Score", color="#ecf0f1", fontsize=9)
    ax2.set_title("Worker Risk Distribution by Zone",
                  color="#ecf0f1", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, labelcolor="#95a5a6",
               facecolor="#1e2130", edgecolor="#2c3e50")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("outputs/v2_zone_analysis.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("  Saved: outputs/v2_zone_analysis.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("ProactiveGuard v2.0 — Module 6: Zone Intelligence Engine")
    print("Microclimate WBGT | Coordinated Detection | Rotation Optimizer")
    print("=" * 65)

    # Current conditions
    base_wbgt      = 37.5
    solar_radiation = 820.0
    current_hour   = 10.5
    humidity       = 52.0

    print("\n[1/5] Zone microclimate WBGT calculation...")
    zone_wbgt_map = get_site_zone_wbgt_map(base_wbgt, solar_radiation)
    print(f"      Base site WBGT: {base_wbgt}°C")
    for zone, wbgt in zone_wbgt_map.items():
        bar = "█" * int((wbgt - 25) * 2)
        print(f"      {zone:20}: {wbgt:.1f}°C {bar}")

    print("\n[2/5] Predictive WBGT (30 min ahead)...")
    wbgt_preds = predict_wbgt_30min(base_wbgt, "S001",
                                     current_hour, humidity)
    p15 = wbgt_preds[14]
    p30 = wbgt_preds[29]
    print(f"      Current WBGT    : {base_wbgt:.1f}°C")
    print(f"      +15 min WBGT    : {p15['predicted_wbgt']:.2f}°C")
    print(f"      +30 min WBGT    : {p30['predicted_wbgt']:.2f}°C")
    print(f"      Trend            : {'↑ Rising' if p30['predicted_wbgt'] > base_wbgt else '↓ Falling'}")

    print("\n[3/5] Coordinated risk detection...")
    # Real per-worker current risk, read from the actual digital twin
    # output -- not a hand-typed demo scenario. Previously this was a
    # hardcoded dict the original author's own comment already
    # admitted was fake ("Simulate scenario: 4 workers in direct_sun
    # all elevated"), unconnected to whatever the real pipeline
    # actually produced.
    predictions_path = "data/v2_twin_predictions.csv"
    worker_risks = {}
    if os.path.exists(predictions_path):
        twin_df = pd.read_csv(predictions_path)
        for wid in twin_df["worker_id"].unique():
            w_data = twin_df[twin_df["worker_id"] == wid]
            worker_risks[wid] = round(
                float(w_data["danger_prob_now"].max()) * 100, 1)
    if not worker_risks:
        print(f"      ⚠️  {predictions_path} not found — run "
              f"v2_data_engine.py and v2_digital_twin.py first for "
              f"real per-worker risk. Using illustrative placeholder "
              f"values for this demo run only.")
        worker_risks = {
            "W001": 72, "W002": 68, "W003": 45,
            "W004": 82, "W005": 38, "W006": 74,
            "W007": 55, "W008": 71, "W009": 32, "W010": 79,
        }
    worker_zones = {wid: wp["zone"] for wid, wp in WORKER_PROFILES.items()}
    events = detect_coordinated_risk(worker_risks, worker_zones,
                                      threshold=60, min_workers=3)
    if events:
        for event in events:
            print(f"\n      🚨 ZONE EVENT DETECTED")
            print(f"         Zone     : {event['zone']}")
            print(f"         Workers  : {event['n_workers']} elevated")
            print(f"         Avg risk : {event['avg_risk']}")
            print(f"         Action   : {event['action'][:80]}...")
    else:
        print("      No coordinated events detected")

    print("\n[4/5] Worker rotation optimization...")
    rotation_recs = optimize_worker_rotation(
        worker_risks, worker_zones, zone_wbgt_map, minutes_on_shift=240)
    for rec in rotation_recs[:3]:
        print(f"\n      [{rec['urgency']}] {rec['worker_name']}")
        print(f"         {rec['current_zone']} → {rec['recommended_zone']}")
        print(f"         WBGT reduction: {rec['wbgt_reduction']:.1f}°C")
        print(f"         {rec['rationale'][:70]}...")

    print("\n[5/5] Zone evacuation evaluation...")
    if events:
        for event in events:
            evac = evaluate_zone_evacuation(
                event["zone"], event["n_workers"],
                event["avg_risk"], event["max_risk"],
                zone_wbgt_map.get(event["zone"], 35),
                coordinated_event=True)
            print(f"\n      Zone: {event['zone']}")
            print(f"      Decision  : {evac['decision']}")
            print(f"      Urgency   : {evac['urgency']}")
            print(f"      {evac['justification'][:80]}...")

    # Generate visualization
    plot_zone_analysis(zone_wbgt_map, worker_risks,
                       worker_zones, rotation_recs)

    # Save zone risk data
    records = []
    for wid, risk in worker_risks.items():
        wp   = WORKER_PROFILES.get(wid, {})
        zone = worker_zones.get(wid, "direct_sun")
        records.append({
            "worker_id":   wid,
            "worker_name": wp.get("name", wid),
            "zone":        zone,
            "zone_wbgt":   zone_wbgt_map.get(zone, base_wbgt),
            "risk_score":  risk,
            "wbgt_15min":  p15["predicted_wbgt"],
            "wbgt_30min":  p30["predicted_wbgt"],
        })
    pd.DataFrame(records).to_csv("data/v2_zone_risk.csv", index=False)

    print("\n" + "=" * 65)
    print("MODULE 6 COMPLETE")
    print("=" * 65)
    print(f"  Zone WBGT map       : {len(zone_wbgt_map)} zones calculated")
    print(f"  Coordinated events  : {len(events)}")
    print(f"  Rotation recs       : {len(rotation_recs)}")
    print(f"  WBGT +30min         : {p30['predicted_wbgt']:.2f}°C predicted")
    print(f"  Saved: outputs/v2_zone_analysis.png")
    print(f"  Saved: data/v2_zone_risk.csv")
    print("\nNext: python v2_rl_scheduler.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
