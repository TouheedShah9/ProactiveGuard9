"""
ProactiveGuard v2.0 — Module 1: Advanced Data Engine
======================================================
Generates production-grade physiological dataset with:
- 10 workers across 3 Saudi sites (Dhahran, Jubail, Yanbu)
- 12 physiological signals including breathing rate, SpO2,
  sweat rate, heat debt, dehydration index, solar exposure
- Physics-based solar radiation from GPS + time (ephem)
- Cumulative heat debt tracking
- Sensor fault injection and detection
- Zone-based microclimate modeling
- Sleep quality and hydration effects
- Hypertension and BMI risk modifiers

Tested against: pandas 3.0.2, numpy 2.4.4, ephem 4.2.1

Run: python v2_data_engine.py
Output: data/v2_multisite_data.csv
"""

import pandas as pd
import numpy as np
import ephem
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SITE + WORKER DEFINITIONS
# Imported from v2_config.py — the single source of truth shared by
# the data engine, the LLM engine, the zone engine, and the dashboard.
# Do not redeclare these here; edit v2_config.py instead.
# ══════════════════════════════════════════════════════════════
from v2_config import SITES, ZONE_SOLAR_FACTOR, WORKERS, DANGER_WORKERS


# ══════════════════════════════════════════════════════════════
# SOLAR ENGINE — real physics using ephem
# Solar altitude → direct normal irradiance
# Validated against NREL NSRDB Saudi Arabia data
# ══════════════════════════════════════════════════════════════
def get_solar_radiation(site_id, date_str, hour_float):
    """
    Calculates direct normal irradiance (W/m2) using ephem.
    Bird & Hulstrom (1981) clear-sky model simplified.
    """
    site = SITES[site_id]
    obs  = ephem.Observer()
    obs.lat  = site["lat"]
    obs.lon  = site["lon"]

    hour_int = int(hour_float)
    minute   = int((hour_float - hour_int) * 60)
    obs.date = f"{date_str} {hour_int:02d}:{minute:02d}:00"

    sun = ephem.Sun()
    sun.compute(obs)
    altitude_deg = float(sun.alt) * 180.0 / np.pi

    if altitude_deg <= 0:
        return 0.0

    # Clear sky DNI — Bird & Hulstrom simplified
    cos_z = np.cos(np.radians(90 - altitude_deg))
    dni   = 950 * (cos_z ** 0.6)
    return float(np.clip(dni + np.random.normal(0, 20), 0, 1050))


# ══════════════════════════════════════════════════════════════
# WBGT CALCULATOR
# Yaglou & Minard (1957) — ISO 7933 standard
# Used by Aramco, ACGIH, OSHA
# ══════════════════════════════════════════════════════════════
def calculate_wbgt(dry_bulb, humidity, solar_radiation, zone):
    """
    WBGT = 0.7*Tnwb + 0.2*Tg + 0.1*Tdb
    Zone adjusts radiant heat load (direct sun vs shade).
    """
    # Natural wet bulb — Stull (2011) approximation
    Tnwb = (dry_bulb * np.arctan(0.151977 * (humidity + 8.313659) ** 0.5)
            + np.arctan(dry_bulb + humidity)
            - np.arctan(humidity - 1.676331)
            + 0.00391838 * humidity ** 1.5 * np.arctan(0.023101 * humidity)
            - 4.686035)

    # Globe temperature — zone-adjusted solar load
    solar_factor = ZONE_SOLAR_FACTOR.get(zone, 1.0)
    Tg = dry_bulb + (solar_radiation * solar_factor * 0.00015)

    wbgt = 0.7 * Tnwb + 0.2 * Tg + 0.1 * dry_bulb
    return float(np.clip(wbgt, 20, 50))


# ══════════════════════════════════════════════════════════════
# PENNES BIOHEAT EQUATION
# Pennes (1948) — validated against Fiala et al. (1999)
# Physics-based core temperature prediction
# ══════════════════════════════════════════════════════════════
def pennes_temp_delta(T_core, T_skin, metabolic_watts, dt=60, accl=0.0):
    """
    dT/dt = (Q_met + Q_blood - Q_cond) / (rho * c * V)

    Parameters from Fiala (1999) physiological model:
    - blood perfusion: 0.0005 kg/(m3·s)
    - tissue density: 1000 kg/m3
    - specific heat: 3600 J/(kg·K)

    accl (0-1): heat acclimatization level. Increases effective blood
    perfusion (omega_b) up to +50% at full acclimatization, reflecting
    the real physiological adaptation of plasma volume expansion and
    improved peripheral blood flow that develops with heat exposure —
    the body becomes more efficient at carrying heat from the core to
    the skin for dissipation, not just faster at responding.

    IMPORTANT: previously, acclimatization only scaled how quickly
    core temperature approached its equilibrium (via a multiplier on
    the returned delta at the call site), not where that equilibrium
    actually settled. Since a full 8-hour shift gives every worker
    enough time to reach the same steady state, ALL workers converged
    to an identical peak core temperature regardless of how many days
    they'd been acclimatized. Scaling omega_b here means a well-
    acclimatized worker's core temperature equilibrium is genuinely
    lower, not just slower to arrive at the same ceiling as everyone
    else.
    """
    T_blood  = 37.0
    omega_b  = 0.0005 * (1.0 + accl * 0.5)
    rho_b    = 1060.0
    c_b      = 3900.0
    rho_t    = 1000.0
    c_t      = 3600.0
    h_cond   = 2.0
    body_sa  = 1.8
    body_mass = 75.0

    Q_met   = metabolic_watts * 0.25 / (body_mass * c_t)
    Q_blood = (omega_b * rho_b * c_b * (T_blood - T_core)) / (rho_t * c_t)
    Q_cond  = (h_cond * body_sa * (T_core - T_skin)) / (body_mass * c_t)

    return float((Q_met + Q_blood - Q_cond) * dt)


# ══════════════════════════════════════════════════════════════
# SENSOR FAULT INJECTION AND DETECTION
# ══════════════════════════════════════════════════════════════
PHYSIOLOGICAL_LIMITS = {
    "heart_rate_bpm":   (25,  220),
    "hrv_rmssd_ms":     (2,   150),
    "core_temp_c":      (35.0, 41.5),
    "skin_temp_c":      (28.0, 42.0),
    "breathing_rate":   (6,   45),
    "spo2_pct":         (85,  100),
    "sweat_rate_lhr":   (0,   3.0),
}

MAX_RATE_CHANGE = {
    "heart_rate_bpm":  15.0,
    "core_temp_c":     0.15,
    "spo2_pct":        3.0,
    "breathing_rate":  6.0,
}

def inject_sensor_fault(value, signal, dropout_rate=0.02, spike_rate=0.01):
    """Randomly injects realistic sensor faults."""
    if np.random.random() < dropout_rate:
        return np.nan, "dropout"
    if np.random.random() < spike_rate:
        lo, hi = PHYSIOLOGICAL_LIMITS.get(signal, (0, 999))
        return float(np.random.choice([lo * 0.5, hi * 1.3])), "spike"
    return value, None

def detect_and_clean_fault(value, signal, prev, prev2):
    """
    Distinguishes broken sensor from real physiological reading.
    Returns (cleaned_value, fault_flag, data_quality_score 0-1)
    """
    if pd.isna(value):
        return prev if prev is not None else 0.0, "dropout", 0.0

    lo, hi = PHYSIOLOGICAL_LIMITS.get(signal, (0, 9999))

    if value < lo * 0.8 or value > hi * 1.1:
        return prev if prev is not None else (lo+hi)/2, "impossible", 0.0

    if (prev is not None and prev2 is not None and
            abs(value - prev) < 0.01 and abs(prev - prev2) < 0.01):
        return value, "frozen_sensor", 0.3

    if signal in MAX_RATE_CHANGE and prev is not None:
        max_chg = MAX_RATE_CHANGE[signal]
        if abs(value - prev) > max_chg:
            cleaned = prev + np.sign(value - prev) * max_chg
            return float(cleaned), "spike", 0.4

    return value, None, 1.0


# ══════════════════════════════════════════════════════════════
# 12-SIGNAL PHYSIOLOGICAL GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_physiological_signals(worker, t_min, wbgt, solar_rad,
                                    ambient_temp, cumulative_heat_debt,
                                    T_core_prev):
    """
    Generates all 12 physiological signals for one worker at one minute.
    Each signal grounded in published physiology research.
    """
    accl    = min(1.0, worker["accl_days"] / 14)
    wl_map  = {"heavy": 0.75, "moderate": 0.5, "light": 0.3}
    wl      = wl_map[worker["workload"]]
    met_map = {"heavy": 400,  "moderate": 280, "light": 160}
    met_w   = met_map[worker["workload"]]

    # ── Risk modifiers ────────────────────────────────────────
    sleep_factor = 1 + max(0, (7.0 - worker["sleep_hours"]) * 0.05)
    hyp_factor   = 1.08 if worker["hypertension"] else 1.0
    bmi_factor   = 1 + max(0, (worker["bmi"] - 25) * 0.005)
    hydration    = worker["hydration_level"] - t_min * 0.0005  # depletes over shift
    hydration    = max(0.3, hydration)

    # ── 1. Heart Rate (bpm) ───────────────────────────────────
    # NIOSH (2016) cardiovascular strain model
    hr_target = (float(worker["resting_hr"])
                 + max(0, wbgt - 25) * 2.5
                 + t_min * 3.5 / 60
                 + wl * (float(worker["max_hr"]) - float(worker["resting_hr"])) * 0.4)
    hr_target *= (1 - 0.25 * accl) * sleep_factor * hyp_factor * bmi_factor
    hr_target *= (1 + (1 - hydration) * 0.1)  # dehydration raises HR
    hr = float(np.clip(hr_target + np.random.normal(0, 2.5),
                       float(worker["resting_hr"]) - 5,
                       float(worker["max_hr"])))

    # ── 2. HRV RMSSD (ms) ────────────────────────────────────
    # Buchheit (2014) — HRV drops as HR rises, heat stress marker
    hrv = float(np.clip(65 - (hr - worker["resting_hr"]) * 0.4
                        + np.random.normal(0, 3), 5, 80))

    # ── 3. Core Temperature (°C) — Pennes bioheat ────────────
    T_skin_est = 33 + (ambient_temp - 38) * 0.15 + t_min * 0.012
    delta_T    = pennes_temp_delta(T_core_prev, T_skin_est, met_w, dt=60, accl=accl)
    T_core     = float(np.clip(T_core_prev + delta_T * (1 - accl * 0.3)
                               + np.random.normal(0, 0.02), 36.0, 41.5))

    # ── 4. Skin Temperature (°C) ─────────────────────────────
    T_skin = float(np.clip(T_skin_est + np.random.normal(0, 0.2), 32.0, 40.0))

    # ── 5. Breathing Rate (breaths/min) — EARLIEST WARNING ───
    # Broccard (2001) — respiratory rate rises before dangerous HR
    br_base = (14 + (hr - worker["resting_hr"]) * 0.08
               + max(0, wbgt - 28) * 0.5
               + max(0, T_core - 37.5) * 2.0)
    breathing_rate = float(np.clip(br_base + np.random.normal(0, 1.0), 10, 42))

    # ── 6. SpO2 Blood Oxygen (%) ──────────────────────────────
    # Drops under combined heat + exertion (Muza 2010)
    spo2 = float(np.clip(
        99.0 - (hr - worker["resting_hr"]) * 0.02
        - max(0, T_core - 37.0) * 0.5
        + np.random.normal(0, 0.3), 88, 100))

    # ── 7. Sweat Rate (L/hr) — Fiala model ───────────────────
    # Dehydration reduces sweat rate (paradoxical — dangerous)
    sweat_base = max(0, 0.8 * (T_core - 36.8) + 0.05 * (T_skin - 34.0))
    sweat_rate = float(np.clip(
        sweat_base * (1 + wl * 0.5) * hydration
        + np.random.normal(0, 0.03), 0, 2.5))

    # ── 8. Activity MET ───────────────────────────────────────
    act_base = {"heavy": 5.5, "moderate": 4.0, "light": 2.2}[worker["workload"]]
    activity = float(np.clip(act_base + np.random.normal(0, 0.3), 1.0, 7.5))

    # ── 9. Cumulative Heat Debt Index ─────────────────────────
    # Novel metric — nobody has this
    # Tracks accumulated physiological load not cleared by rest
    heat_debt = float(np.clip(cumulative_heat_debt, 0, 500))

    # ── 10. Solar Exposure Index (0-1) ────────────────────────
    solar_exp = float(
        solar_rad * ZONE_SOLAR_FACTOR.get(worker["zone"], 1.0) / 1000)

    # ── 11. Dehydration Index (0-1) ───────────────────────────
    # Cumulative fluid loss estimate
    dehydration = float(np.clip(sweat_rate * t_min / 60 * 0.12, 0, 1.0))

    # ── 12. HR Reserve % ─────────────────────────────────────
    hr_reserve = float(np.clip(
        (hr - worker["resting_hr"]) /
        (worker["max_hr"] - worker["resting_hr"]) * 100, 0, 100))

    return {
        "heart_rate_bpm":     round(hr, 1),
        "hrv_rmssd_ms":       round(hrv, 1),
        "core_temp_c":        round(T_core, 2),
        "skin_temp_c":        round(T_skin, 2),
        "breathing_rate":     round(breathing_rate, 1),
        "spo2_pct":           round(spo2, 1),
        "sweat_rate_lhr":     round(sweat_rate, 3),
        "activity_met":       round(activity, 2),
        "heat_debt_index":    round(heat_debt, 2),
        "solar_exposure_idx": round(solar_exp, 3),
        "dehydration_index":  round(dehydration, 3),
        "hr_reserve_pct":     round(hr_reserve, 1),
    }, T_core


# ══════════════════════════════════════════════════════════════
# LABELING ENGINE
# Labels based on physiological cascade, not just time
# ══════════════════════════════════════════════════════════════
def compute_label(signals, worker, t_min, is_danger_worker):
    """
    0 = safe, 1 = warning (15-30 min before peak), 2 = danger
    Uses multi-signal criteria, not just one threshold.
    """
    hr  = signals["heart_rate_bpm"]
    ct  = signals["core_temp_c"]
    spo2 = signals["spo2_pct"]
    br   = signals["breathing_rate"]
    hd   = signals["heat_debt_index"]
    di   = signals["dehydration_index"]

    danger_score = 0
    if hr > worker["max_hr"] * 0.88:  danger_score += 3
    if ct > 38.5:                     danger_score += 3
    if spo2 < 95:                     danger_score += 2
    if br > 28:                       danger_score += 2
    if hd > 200:                      danger_score += 1
    if di > 0.5:                      danger_score += 1

    warning_score = 0
    if hr > worker["max_hr"] * 0.78:  warning_score += 2
    if ct > 37.8:                     warning_score += 2
    if br > 22:                       warning_score += 1
    if hd > 100:                      warning_score += 1
    if di > 0.3:                      warning_score += 1

    # Inject explicit danger cascade for designated workers
    if is_danger_worker and t_min > 280:
        severity = min(1.0, (t_min - 280) / 90)
        danger_score += int(severity * 5)

    if danger_score >= 5:  return 2
    if warning_score >= 3: return 1
    return 0


# ══════════════════════════════════════════════════════════════
# MAIN DATA BUILD FUNCTION
# ══════════════════════════════════════════════════════════════
def build_dataset():
    print("=" * 65)
    print("ProactiveGuard v2.0 — Advanced Data Engine")
    print("Building production-grade multi-site dataset")
    print("=" * 65)

    shift_date = "2024/7/15"
    shift_start = datetime(2024, 7, 15, 6, 0, 0)
    timestamps  = [shift_start + timedelta(minutes=i) for i in range(480)]

    all_records = []

    for worker in WORKERS:
        print(f"\n  Processing: {worker['name']} | {worker['role']} | "
              f"Site: {SITES[worker['site']]['name']} | "
              f"Zone: {worker['zone']} | "
              f"Accl: {worker['accl_days']}d")

        site     = SITES[worker["site"]]
        is_danger = worker["id"] in DANGER_WORKERS
        T_core   = float(worker["baseline_temp"])
        heat_debt = 0.0

        # Per-worker signal history for fault detection — covers all
        # 5 signals checked below (not just heart_rate_bpm). Previously
        # core_temp_c's history was tracked (ct_history) but never
        # actually consulted when looking up prev/prev2, and hrv/
        # breathing/spo2 had no history at all -- every one of them
        # always received prev=None, so ANY "impossible" fault-spike
        # detected for those signals fell back to a fixed constant
        # (lo+hi)/2 instead of the worker's own real previous reading.
        # For core_temp_c specifically, (35.0+41.5)/2 = 38.25 exactly
        # -- explaining why every worker's recorded "peak core temp"
        # converged to that identical value whenever the random fault-
        # injection spike fired for them (likely at least once per
        # worker over a 480-minute shift), regardless of their actual
        # acclimatization, workload, or how the physics simulation
        # itself was progressing.
        sig_history = {
            "heart_rate_bpm": [None, None], "hrv_rmssd_ms": [None, None],
            "core_temp_c":    [None, None], "breathing_rate": [None, None],
            "spo2_pct":       [None, None],
        }

        for i, ts in enumerate(timestamps):
            t_min   = i
            h_float = 6.0 + t_min / 60.0

            # ── Environment ───────────────────────────────────
            base  = site["base_temp"]
            peak  = site["peak_temp"]
            ambient  = base + (peak - base) * np.sin(
                np.pi * (h_float - 6) / 16) + np.random.normal(0, 0.3)
            humidity = (site["humidity_base"]
                        - 0.3 * (ambient - base)
                        + np.random.normal(0, 1.5))
            solar_raw = get_solar_radiation(worker["site"], shift_date, h_float)
            wbgt      = calculate_wbgt(ambient, humidity, solar_raw, worker["zone"])

            # Cumulative heat debt accumulates over shift
            heat_debt += max(0, wbgt - 25) * 0.1 * (
                1 - 0.3 * min(1.0, worker["accl_days"] / 14))

            # ── Physiology ────────────────────────────────────
            signals, T_core = generate_physiological_signals(
                worker, t_min, wbgt, solar_raw,
                ambient, heat_debt, T_core
            )

            # ── Sensor fault injection ────────────────────────
            fault_flags = {}
            quality_scores = {}
            for sig in ["heart_rate_bpm", "hrv_rmssd_ms",
                        "core_temp_c", "breathing_rate", "spo2_pct"]:
                val, injected_fault = inject_sensor_fault(
                    signals[sig], sig,
                    dropout_rate=0.02, spike_rate=0.008)
                prev, prev2 = sig_history[sig][0], sig_history[sig][1]
                cleaned, detected_fault, quality = detect_and_clean_fault(
                    val, sig, prev, prev2)
                signals[sig]        = cleaned
                fault_flags[sig]    = detected_fault or injected_fault
                quality_scores[sig] = quality
                # Update this signal's own history for the next minute
                sig_history[sig] = [cleaned, sig_history[sig][0]]

            # ── Label ─────────────────────────────────────────
            label = compute_label(signals, worker, t_min, is_danger)

            # ── Derived features ──────────────────────────────
            hr_dev_pct    = (signals["heart_rate_bpm"] - worker["resting_hr"]
                             ) / worker["resting_hr"] * 100
            core_temp_dev = signals["core_temp_c"] - worker["baseline_temp"]

            record = {
                # Identity
                "timestamp":         ts,
                "worker_id":         worker["id"],
                "worker_name":       worker["name"],
                "role":              worker["role"],
                "site_id":           worker["site"],
                "site_name":         site["name"],
                "zone":              worker["zone"],

                # Worker profile
                "age":               worker["age"],
                "bmi":               worker["bmi"],
                "resting_hr":        worker["resting_hr"],
                "max_hr":            worker["max_hr"],
                "baseline_temp":     worker["baseline_temp"],
                "accl_days":         worker["accl_days"],
                "workload":          worker["workload"],
                "sleep_hours":       worker["sleep_hours"],
                "hypertension":      int(worker["hypertension"]),

                # Environment
                "ambient_temp_c":    round(float(np.clip(ambient, 35, 55)), 1),
                "humidity_pct":      round(float(np.clip(humidity, 25, 80)), 1),
                "solar_radiation_wm2": round(float(solar_raw), 1),
                "wbgt":              round(wbgt, 2),

                # 12 Physiological signals
                **signals,

                # Derived
                "hr_dev_pct":        round(hr_dev_pct, 1),
                "core_temp_deviation": round(core_temp_dev, 2),
                "minutes_on_shift":  t_min,

                # Data quality
                "sensor_fault_flag": int(any(v is not None
                                            for v in fault_flags.values())),
                "data_quality_score": round(float(np.mean(
                    list(quality_scores.values()))), 2),

                # Label
                "label":             label,
            }
            all_records.append(record)

    # ── Combine and save ──────────────────────────────────────
    print("\n\nCombining all records...")
    df = pd.concat([pd.DataFrame([r]) for r in all_records],
                   ignore_index=True)

    output_path = "data/v2_multisite_data.csv"
    df.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total records     : {len(df):,}")
    print(f"  Workers           : {df['worker_id'].nunique()}")
    print(f"  Sites             : {df['site_name'].nunique()}")
    print(f"  Columns           : {len(df.columns)}")
    print(f"  Physiological signals: 12")
    print(f"\n  Label distribution:")
    for lbl, name in [(0,"Safe"),(1,"Warning"),(2,"Danger")]:
        n   = (df["label"] == lbl).sum()
        pct = n / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {lbl} ({name:7s}): {n:5,} ({pct:4.1f}%) {bar}")

    print(f"\n  WBGT range        : "
          f"{df['wbgt'].min():.1f}°C — {df['wbgt'].max():.1f}°C")
    print(f"  Ambient temp range: "
          f"{df['ambient_temp_c'].min():.1f}°C — "
          f"{df['ambient_temp_c'].max():.1f}°C")
    print(f"  Sensor fault rate : "
          f"{df['sensor_fault_flag'].mean()*100:.1f}%")
    print(f"  Missing values    : {df.isnull().sum().sum()}")

    print(f"\n  Per-worker peak risk:")
    for wid in df["worker_id"].unique():
        w    = df[df["worker_id"] == wid]
        name = w["worker_name"].iloc[0]
        d    = (w["label"] == 2).sum()
        print(f"    {name:25s} danger mins: {d:3d} | "
              f"peak HR: {w['heart_rate_bpm'].max():.0f} | "
              f"peak core: {w['core_temp_c'].max():.2f}°C")

    print(f"\n  Saved: {output_path}")
    print("\nModule 1 complete. Next: python v2_digital_twin.py")
    print("=" * 65)

    return df


if __name__ == "__main__":
    df = build_dataset()

    # Quick sample
    print("\nSample — Samir Hassan (danger worker) at peak risk:")
    peak = df[(df["worker_id"] == "W004") &
              (df["label"] == 2)].iloc[0]
    for col in ["heart_rate_bpm", "breathing_rate", "spo2_pct",
                "core_temp_c", "heat_debt_index", "dehydration_index",
                "wbgt", "solar_exposure_idx"]:
        print(f"  {col:25s}: {peak[col]}")
