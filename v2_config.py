"""
ProactiveGuard v2.0 — Shared Domain Configuration
===================================================
SINGLE SOURCE OF TRUTH for the worker roster and site definitions.

Before this file existed, the 10-worker roster and 3-site definitions
were hand-copied into THREE separate files (v2_data_engine.py,
v2_llm_engine.py, v2_dashboard.py). That meant changing one worker's
resting heart rate required editing three files and hoping you didn't
miss one — exactly the kind of drift that makes a demo diverge from
the "real" system silently. Every module now imports from here.

Import this as:
    from v2_config import WORKERS, WORKERS_BY_ID, SITES, \
        ZONE_SOLAR_FACTOR, ZONE_WBGT_OFFSET, ZONE_CAPACITY, DANGER_WORKERS
"""

# ══════════════════════════════════════════════════════════════
# SITE DEFINITIONS — 3 Saudi Aramco-style operational sites
# ══════════════════════════════════════════════════════════════
SITES = {
    "S001": {"name": "Dhahran", "lat": "26.4207", "lon": "50.0888",
             "base_temp": 42, "peak_temp": 52, "humidity_base": 55},
    "S002": {"name": "Jubail",  "lat": "27.0046", "lon": "49.6580",
             "base_temp": 41, "peak_temp": 51, "humidity_base": 58},
    "S003": {"name": "Yanbu",   "lat": "24.0895", "lon": "38.0618",
             "base_temp": 40, "peak_temp": 49, "humidity_base": 52},
}

# ── Zone shading effect: TWO deliberately different models ────
# ZONE_SOLAR_FACTOR (0-1 multiplier) feeds the physics-based WBGT
# calculation in v2_data_engine.py, where raw solar radiation (W/m2,
# from ephem's sun-altitude calculation) is scaled by shade before
# the globe-temperature term is computed.
#
# ZONE_WBGT_OFFSET (additive degrees C) is a fast approximation used
# by v2_zone_engine.py and the live dashboard, where there's no time
# to re-run the full radiation model every refresh tick — it applies
# a pre-computed average offset instead. This is an intentional
# accuracy/speed tradeoff, not accidental duplication: the offline
# data generator can afford the expensive calculation per worker per
# minute; the live dashboard needs a cheap approximation it can
# recompute every second for 10 workers at once.
ZONE_SOLAR_FACTOR = {
    "direct_sun":      1.00,
    "partial_shade":   0.50,
    "full_shade":       0.10,
    "indoor_covered":  0.00,
}

ZONE_WBGT_OFFSET = {
    "direct_sun":      +3.0,   # full solar radiation
    "partial_shade":   +1.0,   # intermittent shade
    "full_shade":      -1.0,   # no direct radiation
    "indoor_covered":  -4.0,   # enclosed, no solar load
}

ZONE_CAPACITY = {
    "direct_sun":      8,
    "partial_shade":   6,
    "full_shade":      4,
    "indoor_covered":  3,
}

# ══════════════════════════════════════════════════════════════
# WORKER PROFILES — 10 workers with full medical/context history
# This is the authoritative, richest record. Modules that only need
# a subset of fields (e.g. the LLM engine) simply read what they need
# from WORKERS_BY_ID rather than keeping their own copy.
# ══════════════════════════════════════════════════════════════
WORKERS = [
    {"id": "W001", "name": "Ahmed Al-Rashidi",  "site": "S001",
     "zone": "direct_sun",     "resting_hr": 62, "max_hr": 186,
     "baseline_temp": 36.6,    "accl_days": 45,  "workload": "heavy",
     "age": 34, "bmi": 26.2,   "sleep_hours": 7.5, "hypertension": False,
     "hydration_level": 0.85,  "role": "Pipefitter"},

    {"id": "W002", "name": "Tariq Mahmoud",     "site": "S001",
     "zone": "direct_sun",     "resting_hr": 71, "max_hr": 192,
     "baseline_temp": 36.8,    "accl_days": 12,  "workload": "heavy",
     "age": 28, "bmi": 24.1,   "sleep_hours": 6.0, "hypertension": False,
     "hydration_level": 0.75,  "role": "Welder"},

    {"id": "W003", "name": "Khalid Al-Otaibi",  "site": "S002",
     "zone": "partial_shade",  "resting_hr": 78, "max_hr": 178,
     "baseline_temp": 36.5,    "accl_days": 90,  "workload": "moderate",
     "age": 42, "bmi": 28.5,   "sleep_hours": 8.0, "hypertension": True,
     "hydration_level": 0.90,  "role": "Crane Operator"},

    {"id": "W004", "name": "Samir Hassan",      "site": "S001",
     "zone": "direct_sun",     "resting_hr": 68, "max_hr": 189,
     "baseline_temp": 36.7,    "accl_days": 3,   "workload": "heavy",
     "age": 31, "bmi": 23.8,   "sleep_hours": 5.5, "hypertension": False,
     "hydration_level": 0.70,  "role": "Scaffolder"},

    {"id": "W005", "name": "Faisal Al-Zahrani", "site": "S003",
     "zone": "full_shade",     "resting_hr": 65, "max_hr": 182,
     "baseline_temp": 36.6,    "accl_days": 60,  "workload": "light",
     "age": 38, "bmi": 25.0,   "sleep_hours": 7.0, "hypertension": False,
     "hydration_level": 0.95,  "role": "Safety Inspector"},

    {"id": "W006", "name": "Omar Al-Ghamdi",    "site": "S002",
     "zone": "direct_sun",     "resting_hr": 75, "max_hr": 185,
     "baseline_temp": 36.9,    "accl_days": 7,   "workload": "heavy",
     "age": 26, "bmi": 22.5,   "sleep_hours": 6.5, "hypertension": False,
     "hydration_level": 0.80,  "role": "Construction Worker"},

    {"id": "W007", "name": "Nasser Al-Qahtani", "site": "S001",
     "zone": "partial_shade",  "resting_hr": 70, "max_hr": 180,
     "baseline_temp": 36.6,    "accl_days": 30,  "workload": "moderate",
     "age": 45, "bmi": 30.1,   "sleep_hours": 7.0, "hypertension": True,
     "hydration_level": 0.85,  "role": "Maintenance Tech"},

    {"id": "W008", "name": "Yusuf Al-Harbi",    "site": "S003",
     "zone": "direct_sun",     "resting_hr": 66, "max_hr": 194,
     "baseline_temp": 36.5,    "accl_days": 21,  "workload": "heavy",
     "age": 24, "bmi": 21.9,   "sleep_hours": 8.0, "hypertension": False,
     "hydration_level": 0.88,  "role": "Rigger"},

    {"id": "W009", "name": "Ibrahim Al-Dosari", "site": "S002",
     "zone": "indoor_covered", "resting_hr": 80, "max_hr": 175,
     "baseline_temp": 36.7,    "accl_days": 180, "workload": "light",
     "age": 52, "bmi": 27.3,   "sleep_hours": 6.0, "hypertension": True,
     "hydration_level": 0.92,  "role": "Instrument Technician"},

    {"id": "W010", "name": "Majed Al-Shehri",   "site": "S003",
     "zone": "direct_sun",     "resting_hr": 63, "max_hr": 187,
     "baseline_temp": 36.6,    "accl_days": 5,   "workload": "heavy",
     "age": 29, "bmi": 23.2,   "sleep_hours": 5.0, "hypertension": False,
     "hydration_level": 0.65,  "role": "Scaffolder"},
]

# Dict-keyed view, for modules that look workers up by ID (dashboard,
# LLM engine, zone engine) rather than iterating a list.
WORKERS_BY_ID = {w["id"]: w for w in WORKERS}

# Workers who get injected danger events in the synthetic dataset
# (new to site + sleep-deprived + dehydrated — the profile most
# likely to actually spike in reality).
DANGER_WORKERS = {"W004", "W010"}
