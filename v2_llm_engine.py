"""
ProactiveGuard v2.0 — Module 4: LLM Intelligence Engine
=========================================================
Generates audience-adaptive natural language explanations using:

Option C (Primary):  Groq API — free, fast, llama3-8b-8192
Option B (Fallback): Rule-based engine — works 100% offline

Three audiences per alert:
- Safety Manager: plain English, action-focused
- Site Doctor:    clinical language, vitals-focused
- Engineer:       technical, SHAP values, confidence

Also generates:
- Zone-level coordinated risk reports
- Shift summary narratives
- Acclimatization anomaly alerts

No competitor product has this capability.

Tested against: groq 1.2.0, python-dotenv

Run: python v2_llm_engine.py
Input:  data/v2_multisite_data.csv
        data/v2_twin_predictions.csv
        models/v2_model.json
        models/v2_scaler.pkl
        models/v2_features.pkl
Output: outputs/v2_llm_demo_output.txt
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════
# GROQ CLIENT SETUP
# ══════════════════════════════════════════════════════════════
def get_groq_client():
    """Returns Groq client if API key available, else None."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        print("      (No GROQ_API_KEY set — running in offline "
              "rule-based explanation mode.)")
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"      ⚠️  Groq client could not be initialized "
              f"({type(e).__name__}: {e}) — running in offline "
              f"rule-based explanation mode.")
        return None

GROQ_MODEL  = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_CLIENT = get_groq_client()


# ══════════════════════════════════════════════════════════════
# WORKER PROFILES
# Imported from v2_config.py — the single source of truth shared by
# the data engine, the LLM engine, the zone engine, and the dashboard.
# Do not redeclare these here; edit v2_config.py instead.
# ══════════════════════════════════════════════════════════════
from v2_config import WORKERS_BY_ID as WORKER_PROFILES


# ══════════════════════════════════════════════════════════════
# OPTION B — RULE-BASED EXPLANATION ENGINE
# Professional, audience-adaptive, zero API dependency.
# This runs when Groq is unavailable or as a fallback.
# ══════════════════════════════════════════════════════════════
def build_clinical_context(worker_id, signals):
    """Extracts key clinical parameters for explanation generation."""
    wp = WORKER_PROFILES.get(worker_id, {})
    hr       = float(signals.get("heart_rate_bpm", 0))
    ct       = float(signals.get("core_temp_c", 36.6))
    br       = float(signals.get("breathing_rate", 14))
    spo2     = float(signals.get("spo2_pct", 98))
    hd       = float(signals.get("heat_debt_index", 0))
    di       = float(signals.get("dehydration_index", 0))
    wbgt     = float(signals.get("wbgt", 28))
    dp       = float(signals.get("danger_prob_now", 0))
    dp_30    = float(signals.get("danger_prob_30min", 0))
    risk     = float(signals.get("risk_score", dp * 100))
    minutes  = int(signals.get("minutes_on_shift", 0))
    conf     = float(signals.get("model_confidence", 0.85))

    rhr      = float(wp.get("resting_hr", 68))
    mhr      = float(wp.get("max_hr", 189))
    base_ct  = float(wp.get("baseline_temp", 36.7))
    accl     = int(wp.get("accl_days", 14))

    hr_dev_pct  = (hr - rhr) / rhr * 100 if rhr > 0 else 0
    hr_reserve  = (hr - rhr) / (mhr - rhr) * 100 if (mhr - rhr) > 0 else 0
    temp_dev    = ct - base_ct
    hours_shift = minutes / 60

    return {
        "name":       wp.get("name", "Worker"),
        "role":       wp.get("role", "Field Worker"),
        "hr":         hr,  "rhr": rhr,  "mhr": mhr,
        "hr_dev_pct": hr_dev_pct,
        "hr_reserve": hr_reserve,
        "ct":         ct,  "base_ct": base_ct, "temp_dev": temp_dev,
        "br":         br,  "spo2": spo2,
        "hd":         hd,  "di": di,
        "wbgt":       wbgt,
        "dp":         dp,  "dp_30": dp_30,
        "risk":       risk,
        "accl":       accl,
        "hours_shift": hours_shift,
        "conf":       conf,
    }


def identify_top_drivers(ctx):
    """Identifies and ranks the top physiological risk drivers."""
    drivers = []

    if ctx["hr_dev_pct"] > 100:
        drivers.append((ctx["hr_dev_pct"] * 0.4,
            f"heart rate is {ctx['hr_dev_pct']:.0f}% above personal baseline "
            f"({ctx['hr']:.0f} vs {ctx['rhr']:.0f} bpm resting)"))
    elif ctx["hr_dev_pct"] > 50:
        drivers.append((ctx["hr_dev_pct"] * 0.3,
            f"heart rate {ctx['hr_dev_pct']:.0f}% above personal baseline"))

    if ctx["temp_dev"] > 1.0:
        drivers.append((ctx["temp_dev"] * 30,
            f"core temperature {ctx['temp_dev']:.1f}°C above personal baseline "
            f"({ctx['ct']:.1f}°C vs normal {ctx['base_ct']:.1f}°C)"))
    elif ctx["temp_dev"] > 0.5:
        drivers.append((ctx["temp_dev"] * 20,
            f"core temperature rising — {ctx['temp_dev']:.1f}°C above baseline"))

    if ctx["br"] > 26:
        drivers.append((ctx["br"] * 0.5,
            f"elevated breathing rate at {ctx['br']:.0f} breaths/min "
            f"(early physiological warning signal)"))

    if ctx["di"] > 0.45:
        drivers.append((ctx["di"] * 50,
            f"significant dehydration detected "
            f"(index {ctx['di']:.2f} — reduces heat tolerance)"))

    if ctx["hd"] > 250:
        drivers.append((ctx["hd"] * 0.1,
            f"high cumulative heat load "
            f"({ctx['hd']:.0f} units — body has not recovered between exertions)"))

    if ctx["accl"] < 7:
        drivers.append((80,
            f"only {ctx['accl']} days acclimatized "
            f"— new workers are 3-5x more susceptible to heat illness"))

    if ctx["wbgt"] > 32:
        drivers.append((ctx["wbgt"],
            f"extreme environmental heat (WBGT {ctx['wbgt']:.1f}°C — "
            f"above safe working threshold for heavy labor)"))

    drivers.sort(key=lambda x: x[0], reverse=True)
    return [d[1] for d in drivers[:3]] if drivers else [
        "multiple physiological signals trending above normal range"]


def explain_for_manager(ctx, drivers):
    """
    Plain English for safety supervisor.
    Action-focused. No jargon. One clear instruction.
    """
    name  = ctx["name"]
    risk  = ctx["risk"]
    dp_30 = ctx["dp_30"]

    if risk >= 85:
        urgency = "CRITICAL — EVACUATE IMMEDIATELY"
        action  = (f"Remove {name} from work area immediately. "
                   f"Move to air-conditioned area. Call site medic now.")
    elif risk >= 70:
        urgency = "DANGER — IMMEDIATE ACTION REQUIRED"
        action  = (f"Send {name} for immediate rest break in shade. "
                   f"Provide cold water. Reassess in 15 minutes.")
    elif risk >= 55:
        urgency = "WARNING — ACTION NEEDED SOON"
        action  = (f"Schedule rest break for {name} within the next 15 minutes. "
                   f"Reduce workload if possible.")
    else:
        urgency = "CAUTION — MONITOR CLOSELY"
        action  = f"Increase check-in frequency for {name}. Watch for deterioration."

    top3 = " | ".join(drivers[:3])

    explanation = (
        f"⚠️  {urgency}\n\n"
        f"Worker: {name} ({ctx['role']})\n"
        f"Risk Score: {risk:.0f}/100\n"
        f"Hours on shift: {ctx['hours_shift']:.1f}h\n\n"
        f"Why: {top3}\n\n"
        f"30-min forecast: danger probability {dp_30*100:.0f}%\n\n"
        f"ACTION: {action}"
    )
    return explanation


def explain_for_doctor(ctx, drivers):
    """
    Clinical language for site doctor or occupational health.
    Includes vitals, deviations, and clinical staging.
    """
    name    = ctx["name"]
    staging = ("Stage 2 Heat Exhaustion" if ctx["risk"] >= 70
               else "Stage 1 Heat Exhaustion" if ctx["risk"] >= 55
               else "Moderate Heat Stress" if ctx["risk"] >= 40
               else "Mild Heat Stress")

    explanation = (
        f"CLINICAL ASSESSMENT — {name}\n"
        f"{'─'*45}\n"
        f"Physiological strain index : {ctx['risk']:.0f}/100\n"
        f"Clinical staging           : {staging}\n\n"
        f"VITAL SIGNS:\n"
        f"  Heart rate               : {ctx['hr']:.0f} bpm "
        f"(+{ctx['hr_dev_pct']:.0f}% vs personal baseline {ctx['rhr']:.0f})\n"
        f"  HR reserve usage         : {ctx['hr_reserve']:.0f}%\n"
        f"  Core temperature         : {ctx['ct']:.2f}°C "
        f"(+{ctx['temp_dev']:.2f}°C vs baseline {ctx['base_ct']:.1f}°C)\n"
        f"  Respiratory rate         : {ctx['br']:.0f}/min\n"
        f"  SpO2                     : {ctx['spo2']:.1f}%\n\n"
        f"RISK FACTORS:\n"
        f"  Cumulative heat debt     : {ctx['hd']:.0f} units\n"
        f"  Dehydration index        : {ctx['di']:.3f}\n"
        f"  Acclimatization          : {ctx['accl']} days\n"
        f"  Environmental WBGT       : {ctx['wbgt']:.1f}°C\n\n"
        f"PRIMARY DRIVERS:\n"
        + "\n".join(f"  {i+1}. {d}" for i, d in enumerate(drivers)) +
        f"\n\n30-MIN PROGNOSIS: Danger probability {ctx['dp_30']*100:.0f}%\n"
        f"Model confidence: {ctx['conf']*100:.0f}%"
    )
    return explanation


def explain_for_engineer(ctx, drivers, shap_features=None):
    """
    Technical output for ML engineer or system integrator.
    Includes SHAP values and model internals.
    """
    name = ctx["name"]
    shap_str = ""
    if shap_features:
        shap_str = "\nSHAP TOP DRIVERS:\n"
        for s in shap_features:
            arrow = "↑" if s["shap_value"] > 0 else "↓"
            shap_str += (f"  {arrow} {s['feature']:35s} "
                         f"SHAP={s['shap_value']:+.3f}\n")

    explanation = (
        f"TECHNICAL REPORT — {name}\n"
        f"{'─'*45}\n"
        f"risk_score          : {ctx['risk']:.2f}\n"
        f"danger_prob_now     : {ctx['dp']:.4f}\n"
        f"danger_prob_30min   : {ctx['dp_30']:.4f}\n"
        f"model_confidence    : {ctx['conf']:.4f}\n\n"
        f"SIGNALS:\n"
        f"  hr_bpm={ctx['hr']:.1f} "
        f"hr_dev_pct={ctx['hr_dev_pct']:.1f} "
        f"hr_reserve={ctx['hr_reserve']:.1f}%\n"
        f"  core_temp={ctx['ct']:.3f} "
        f"temp_dev={ctx['temp_dev']:+.3f}C\n"
        f"  breathing_rate={ctx['br']:.1f} "
        f"spo2={ctx['spo2']:.1f}%\n"
        f"  heat_debt={ctx['hd']:.1f} "
        f"dehydration={ctx['di']:.3f}\n"
        f"  wbgt={ctx['wbgt']:.1f}C "
        f"accl_days={ctx['accl']}\n"
        f"{shap_str}"
    )
    return explanation


def generate_rule_based_explanation(worker_id, signals,
                                     audience="manager",
                                     shap_features=None):
    """
    Master rule-based explanation function.
    Called when Groq is unavailable or as primary Option B.
    """
    ctx     = build_clinical_context(worker_id, signals)
    drivers = identify_top_drivers(ctx)

    if audience == "manager":
        return explain_for_manager(ctx, drivers)
    elif audience == "doctor":
        return explain_for_doctor(ctx, drivers)
    elif audience == "engineer":
        return explain_for_engineer(ctx, drivers, shap_features)
    else:
        return explain_for_manager(ctx, drivers)


# ══════════════════════════════════════════════════════════════
# OPTION C — GROQ LLM ENGINE
# ══════════════════════════════════════════════════════════════
def generate_groq_explanation(worker_id, signals, audience="manager",
                               shap_features=None):
    """
    Generates LLM-powered explanation via Groq API.
    Falls back to rule-based if API fails.
    """
    if GROQ_CLIENT is None:
        return generate_rule_based_explanation(
            worker_id, signals, audience, shap_features), "rule_based"

    ctx     = build_clinical_context(worker_id, signals)
    drivers = identify_top_drivers(ctx)

    audience_instructions = {
        "manager": (
            "You are a safety AI system speaking to a construction site "
            "safety manager with no medical background. Be direct, use "
            "plain English, give ONE clear action. Maximum 4 sentences."
        ),
        "doctor": (
            "You are a safety AI system speaking to an occupational health "
            "physician. Use clinical terminology. Include vital sign "
            "deviations from personal baseline. Maximum 5 sentences."
        ),
        "engineer": (
            "You are a safety AI system speaking to an ML engineer. "
            "Reference the SHAP drivers by name and describe their "
            "direction and relative magnitude in your own words (e.g. "
            "'heart rate is the dominant driver, increasing risk "
            "substantially'). Do NOT restate the exact SHAP decimal "
            "values yourself — those are already shown precisely in a "
            "separate chart; transcribing specific numbers from memory "
            "risks stating one incorrectly. Include model confidence "
            "and signal names. Be precise and technical. Maximum 5 "
            "sentences."
        ),
    }

    shap_context = ""
    if shap_features:
        shap_context = "SHAP drivers: " + ", ".join(
            f"{s['feature']}({s['shap_value']:+.3f})"
            for s in shap_features[:3])

    prompt = (
        f"Worker: {ctx['name']} | Role: {ctx['role']}\n"
        f"Risk score: {ctx['risk']:.0f}/100 | "
        f"Danger probability: {ctx['dp']*100:.0f}%\n"
        f"Heart rate: {ctx['hr']:.0f} bpm "
        f"(+{ctx['hr_dev_pct']:.0f}% above personal baseline)\n"
        f"Core temp: {ctx['ct']:.2f}°C "
        f"(+{ctx['temp_dev']:.2f}°C above baseline)\n"
        f"Breathing: {ctx['br']:.0f}/min | SpO2: {ctx['spo2']:.1f}%\n"
        f"WBGT: {ctx['wbgt']:.1f}°C | "
        f"Acclimatization: {ctx['accl']} days\n"
        f"Hours on shift: {ctx['hours_shift']:.1f}h\n"
        f"Top risk drivers: {' | '.join(drivers[:2])}\n"
        f"{shap_context}\n\n"
        f"Generate a {audience} explanation."
    )

    try:
        response = GROQ_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": audience_instructions.get(
                     audience, audience_instructions["manager"])},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
            reasoning_effort="low",
        )
        text = response.choices[0].message.content.strip()
        if not text:
            # openai/gpt-oss-20b is a reasoning model: it can spend its
            # entire token budget on internal reasoning and emit an
            # empty final answer if that budget runs out first. Empty
            # content is a real failure for a safety system, not a
            # valid (if terse) explanation — treat it as one so the
            # honest rule-based fallback fires instead of showing the
            # person nothing.
            raise ValueError(
                "Groq returned empty content (reasoning model likely "
                "exhausted its token budget before producing a final "
                "answer)")
        return text, "groq"

    except Exception as e:
        # Previously this silently swallowed the real error, so if the
        # Groq API call failed for ANY reason (invalid/deprecated
        # model name, expired key, network issue, rate limit), the
        # system would silently and permanently run in fallback mode
        # with zero indication why — "Mode: Groq API" would print at
        # startup while every single explanation actually came from
        # the offline rule-based engine. Print the real reason so it's
        # diagnosable instead of a mystery.
        print(f"      ⚠️  Groq API call failed ({type(e).__name__}: {e}) "
              f"— falling back to rule-based explanation. If this "
              f"persists, check GROQ_MODEL in your .env — Groq "
              f"periodically deprecates older model names (e.g. "
              f"'llama3-8b-8192' has been superseded by newer Llama "
              f"3.1/3.3 models).")
        return generate_rule_based_explanation(
            worker_id, signals, audience, shap_features), "rule_based_fallback"


# ══════════════════════════════════════════════════════════════
# ZONE-LEVEL COORDINATED RISK DETECTION
# Novel feature — no competitor has this.
# If 3+ workers in same zone spike simultaneously:
# environmental event, not individual physiology.
#
# The actual detection logic lives in v2_zone_engine.py — this used
# to be a third independent reimplementation of the same "3+ workers
# elevated" rule (the dashboard had its own copy too). This wrapper
# just calls the real thing and reshapes the result into the field
# names generate_zone_report() below expects, so nothing downstream
# had to change.
# ══════════════════════════════════════════════════════════════
def detect_zone_event(worker_risk_scores, worker_zones,
                       threshold=65, min_workers=3):
    """
    Detects coordinated environmental risk events.

    Parameters:
    - worker_risk_scores: dict {worker_id: risk_score}
    - worker_zones: dict {worker_id: zone_name}
    - threshold: risk score to consider "elevated"
    - min_workers: minimum workers spiking to flag event

    Returns: list of zone events, or empty list
    """
    from v2_zone_engine import detect_coordinated_risk
    raw_events = detect_coordinated_risk(
        worker_risk_scores, worker_zones,
        threshold=threshold, min_workers=min_workers)

    events = []
    for ev in raw_events:
        events.append({
            "zone":          ev["zone"],
            "n_workers":     ev["n_workers"],
            "workers":       [w["worker_id"] for w in ev["workers"]],
            "avg_risk":      ev["avg_risk"],
            "event_type":    ev["event_type"],
            "severity":      ev["severity"],
            "recommendation": ev["action"],
        })
    return events


def generate_zone_report(events):
    """Generates plain English zone situation report."""
    if not events:
        return "All zones within normal parameters. No coordinated risk events detected."

    reports = []
    for event in events:
        reports.append(
            f"🚨 ZONE EVENT — {event['zone'].upper().replace('_',' ')}\n"
            f"   {event['n_workers']} workers elevated simultaneously "
            f"(avg risk: {event['avg_risk']:.0f})\n"
            f"   This pattern indicates an ENVIRONMENTAL event, "
            f"not individual physiology.\n"
            f"   {event['recommendation']}"
        )
    return "\n\n".join(reports)


# ══════════════════════════════════════════════════════════════
# SHIFT SUMMARY GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_shift_summary(shift_data):
    """
    Generates end-of-shift narrative report.
    Used for handover between safety supervisors.
    """
    n_workers     = shift_data.get("n_workers", 0)
    n_alerts      = shift_data.get("n_alerts", 0)
    max_risk      = shift_data.get("max_risk_score", 0)
    highest_risk  = shift_data.get("highest_risk_worker", "Unknown")
    avg_wbgt      = shift_data.get("avg_wbgt", 0)
    n_danger_mins = shift_data.get("total_danger_minutes", 0)
    accl_flags    = shift_data.get("accl_anomaly_workers", [])
    site_name     = shift_data.get("site_name", "Field Site")

    summary = (
        f"SHIFT SUMMARY REPORT — {site_name}\n"
        f"{'='*50}\n\n"
        f"OVERVIEW:\n"
        f"  Workers monitored     : {n_workers}\n"
        f"  Alerts fired          : {n_alerts}\n"
        f"  Peak risk score       : {max_risk:.0f}/100 ({highest_risk})\n"
        f"  Average WBGT          : {avg_wbgt:.1f}°C\n"
        f"  Total danger minutes  : {n_danger_mins}\n\n"
    )

    if n_alerts == 0:
        summary += "ASSESSMENT: Shift completed without critical incidents.\n"
    elif n_alerts <= 2:
        summary += (f"ASSESSMENT: {n_alerts} alert(s) fired. "
                    f"All workers returned to safe zone.\n")
    else:
        summary += (f"⚠️  ASSESSMENT: {n_alerts} alerts fired this shift. "
                    f"Review environmental conditions and workload assignment.\n")

    if accl_flags:
        summary += (f"\nACCLIMATIZATION FLAGS:\n"
                    f"  The following workers are not acclimatizing at "
                    f"expected rate: {', '.join(accl_flags)}.\n"
                    f"  Recommend occupational health review before "
                    f"next high-heat shift.\n")

    summary += f"\nReport generated by ProactiveGuard v2.0"
    return summary


# ══════════════════════════════════════════════════════════════
# DEMO FUNCTION — shows all LLM capabilities
# ══════════════════════════════════════════════════════════════
def _load_real_demo_data():
    """
    Loads real data from the actual pipeline output for the demo,
    instead of hand-typed numbers.

    IMPORTANT — this replaces a fully fabricated demo. The previous
    version of run_demo() used a hardcoded "Samir Hassan" vitals dict,
    a hardcoded worker_risks dict for zone detection, and a hardcoded
    shift_data dict — including the exact same fabricated
    accl_anomaly_workers=["Samir Hassan","Majed Al-Shehri"] list found
    and fixed elsewhere in this project (v2_safety_infrastructure.py).
    None of it was connected to data/v2_twin_predictions.csv despite
    this file's own docstring claiming that as an input. Now it is.

    Returns a dict with: demo_signals, demo_shap (or None), worker_id,
    worker_risks, worker_zones, shift_data — or None if the required
    pipeline files don't exist yet.
    """
    predictions_path = "data/v2_twin_predictions.csv"
    if not os.path.exists(predictions_path):
        return None

    twin_df = pd.read_csv(predictions_path)

    # Real signals: the worker/moment with the actual highest measured
    # danger probability across the whole simulated dataset.
    peak_idx = twin_df["danger_prob_now"].idxmax()
    peak_row = twin_df.loc[peak_idx]
    worker_id = peak_row["worker_id"]

    demo_signals = {
        "heart_rate_bpm":    float(peak_row["heart_rate_bpm"]),
        "core_temp_c":       float(peak_row["core_temp_c"]),
        "breathing_rate":    float(peak_row["breathing_rate"]),
        "spo2_pct":          float(peak_row["spo2_pct"]),
        "heat_debt_index":   float(peak_row["heat_debt_index"]),
        "dehydration_index": float(peak_row["dehydration_index"]),
        "wbgt":              float(peak_row["wbgt"]),
        "danger_prob_now":   float(peak_row["danger_prob_now"]),
        "danger_prob_30min": float(peak_row["danger_prob_30min"]),
        "risk_score":        round(float(peak_row["danger_prob_now"]) * 100, 1),
        "minutes_on_shift":  int(peak_row["minutes_on_shift"]),
        "model_confidence":  float(peak_row.get("twin_confidence", 0.85)),
    }

    # Real SHAP values for this exact moment, computed with the actual
    # trained model — not fabricated numbers. Best-effort: if the
    # trained model isn't available yet, proceed without SHAP rather
    # than fail the whole demo.
    demo_shap = None
    model_files = ["models/v2_model.json", "models/v2_scaler.pkl",
                    "models/v2_features.pkl"]
    if all(os.path.exists(p) for p in model_files):
        try:
            import xgboost as xgb
            from v2_model import (engineer_temporal_features,
                                   XGBoostNativeExplainer,
                                   get_shap_explanation_for_worker)
            df_main = pd.read_csv("data/v2_multisite_data.csv",
                                  parse_dates=["timestamp"])
            df_twin = pd.read_csv(predictions_path, parse_dates=["timestamp"])
            twin_cols = ["worker_id", "minutes_on_shift", "danger_prob_now",
                         "danger_prob_30min", "cv_strain_now"]
            available = [c for c in twin_cols if c in df_twin.columns]
            df = pd.merge(df_main, df_twin[available],
                          on=["worker_id", "minutes_on_shift"], how="left")
            df["accl_days"] = df["worker_id"].map(
                lambda x: WORKER_PROFILES.get(x, {}).get("accl_days", 30))
            df = engineer_temporal_features(df)

            with open("models/v2_features.pkl", "rb") as f:
                feature_names = pickle.load(f)
            with open("models/v2_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            model = xgb.XGBClassifier()
            model.load_model("models/v2_model.json")

            match = df[(df["worker_id"] == worker_id) &
                       (df["minutes_on_shift"] == peak_row["minutes_on_shift"])]
            if len(match):
                x_scaled = scaler.transform(match[feature_names].values)[0]
                explainer = XGBoostNativeExplainer(model, feature_names)
                demo_shap = get_shap_explanation_for_worker(
                    explainer, x_scaled, feature_names)
        except Exception as e:
            print(f"      ⚠️  Could not compute live SHAP for demo "
                  f"({type(e).__name__}: {e}) — proceeding without "
                  f"SHAP driver detail.")

    # Real per-worker peak risk for zone-level coordinated detection.
    worker_risks = {}
    worker_zones = {}
    for wid in twin_df["worker_id"].unique():
        w_data = twin_df[twin_df["worker_id"] == wid]
        worker_risks[wid] = round(float(w_data["danger_prob_now"].max()) * 100, 1)
        worker_zones[wid] = WORKER_PROFILES.get(wid, {}).get("zone", "direct_sun")

    # Real shift summary aggregates.
    danger_minutes_total = int((twin_df["danger_prob_now"] >= 0.70).sum())
    n_alerts = int((twin_df.groupby("worker_id")["danger_prob_now"]
                    .max() >= 0.50).sum())
    peak_worker_id = max(worker_risks, key=worker_risks.get)
    accl_flagged = [
        WORKER_PROFILES.get(wid, {}).get("name", wid)
        for wid in twin_df.loc[
            twin_df.get("strain_or_new_worker_flag", 0) == 1,
            "worker_id"].unique()
    ] if "strain_or_new_worker_flag" in twin_df.columns else []

    shift_data = {
        "n_workers":            twin_df["worker_id"].nunique(),
        "n_alerts":             n_alerts,
        "max_risk_score":       worker_risks[peak_worker_id],
        "highest_risk_worker":  WORKER_PROFILES.get(
                                    peak_worker_id, {}).get("name", peak_worker_id),
        "avg_wbgt":             round(float(twin_df["wbgt"].mean()), 1),
        "total_danger_minutes": danger_minutes_total,
        "accl_anomaly_workers": accl_flagged,
        "site_name":            "Dhahran Operations — Zone A",
    }

    return {
        "demo_signals": demo_signals, "demo_shap": demo_shap,
        "worker_id": worker_id, "worker_risks": worker_risks,
        "worker_zones": worker_zones, "shift_data": shift_data,
    }


def run_demo():
    print("=" * 65)
    print("ProactiveGuard v2.0 — Module 4: LLM Intelligence Engine")
    mode = "Groq API" if GROQ_CLIENT else "Rule-Based (Offline)"
    print(f"Mode: {mode}")
    print("=" * 65)

    real = _load_real_demo_data()
    if real is None:
        print("\n⚠️  data/v2_twin_predictions.csv not found — run "
              "v2_data_engine.py and v2_digital_twin.py first. "
              "Cannot run a real demo without real pipeline data.")
        return

    demo_signals = real["demo_signals"]
    demo_shap    = real["demo_shap"]
    worker_id    = real["worker_id"]

    output_lines = []

    # ── Three-audience explanations ───────────────────────────
    print("\n[1/4] Generating three-audience explanations...")
    for audience in ["manager", "doctor", "engineer"]:
        explanation, source = generate_groq_explanation(
            worker_id, demo_signals, audience=audience,
            shap_features=demo_shap if audience == "engineer" else None
        )
        print(f"\n{'─'*60}")
        print(f"AUDIENCE: {audience.upper()} | Source: {source}")
        print(f"{'─'*60}")
        print(explanation)
        output_lines.append(f"\n[{audience.upper()}] ({source})\n{explanation}")

    # ── Zone event detection ──────────────────────────────────
    print(f"\n[2/4] Zone-level coordinated risk detection...")
    events      = detect_zone_event(real["worker_risks"], real["worker_zones"])
    zone_report = generate_zone_report(events)
    print(f"\n{'─'*60}")
    print("ZONE SITUATION REPORT")
    print(f"{'─'*60}")
    print(zone_report)
    output_lines.append(f"\n[ZONE REPORT]\n{zone_report}")

    # ── Shift summary ─────────────────────────────────────────
    print(f"\n[3/4] Generating shift summary...")
    summary = generate_shift_summary(real["shift_data"])
    print(f"\n{'─'*60}")
    print(summary)
    output_lines.append(f"\n[SHIFT SUMMARY]\n{summary}")

    # ── Save output ───────────────────────────────────────────
    print(f"\n[4/4] Saving demo output...")
    output_path = "outputs/v2_llm_demo_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ProactiveGuard v2.0 — LLM Engine Demo Output\n")
        f.write(f"Mode: {mode}\n")
        f.write("=" * 65 + "\n")
        f.write("\n".join(output_lines))
    print(f"  Saved: {output_path}")

    print("\n" + "=" * 65)
    print("MODULE 4 COMPLETE")
    print("=" * 65)
    print(f"  LLM Mode            : {mode}")
    print(f"  Audiences supported : Manager | Doctor | Engineer")
    print(f"  Zone detection      : {len(events)} event(s) found")
    print(f"  Fallback available  : Rule-based engine (offline)")
    print("\nNext: streamlit run v2_dashboard.py")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
