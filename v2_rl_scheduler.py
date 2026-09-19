"""
ProactiveGuard v2.0 — Module 7: Reinforcement Learning Scheduler
=================================================================
Q-learning agent that proactively recommends schedule changes
BEFORE danger arrives — not after.

This transforms ProactiveGuard from a safety alarm into an
operations optimizer. No commercial safety product does this.

Features:
- Q-learning agent trained on shift simulation
- Personal danger window prediction per worker
- Proactive schedule recommendations
- Productivity vs safety tradeoff optimization
- Shift pattern memory (Ahmed always spikes at hour 5)

Run: python v2_rl_scheduler.py
Output: models/v2_rl_agent.pkl
        outputs/v2_rl_analysis.png
        data/v2_schedule_recommendations.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)

# ══════════════════════════════════════════════════════════════
# WORKER PROFILES
# Imported from v2_config.py — the single source of truth shared by
# the data engine, LLM engine, zone engine, RL scheduler, and
# dashboard. Do not redeclare these here; edit v2_config.py instead.
# ══════════════════════════════════════════════════════════════
from v2_config import WORKERS_BY_ID as WORKER_PROFILES

# ══════════════════════════════════════════════════════════════
# ACTIONS
# ══════════════════════════════════════════════════════════════
ACTIONS = {
    0: {"name": "CONTINUE",        "risk_reduction": 0,   "productivity_cost": 0},
    1: {"name": "INCREASE_BREAKS", "risk_reduction": 8,   "productivity_cost": 5},
    2: {"name": "REST_BREAK",      "risk_reduction": 20,  "productivity_cost": 12},
    3: {"name": "ROTATE_ZONE",     "risk_reduction": 15,  "productivity_cost": 8},
    4: {"name": "REDUCE_WORKLOAD", "risk_reduction": 18,  "productivity_cost": 10},
    5: {"name": "EVACUATE",        "risk_reduction": 50,  "productivity_cost": 50},
}
N_ACTIONS = len(ACTIONS)


# ══════════════════════════════════════════════════════════════
# STATE DISCRETIZATION
# Q-table needs discrete states.
# State = (risk_bin, wbgt_bin, accl_bin, hour_bin)
# ══════════════════════════════════════════════════════════════
def discretize_state(risk_score, wbgt, accl_days, hour_on_shift):
    risk_bin  = min(int(risk_score / 20), 4)       # 0-4 (0-20,20-40,40-60,60-80,80+)
    wbgt_bin  = min(int((wbgt - 25) / 5), 4)       # 0-4
    accl_bin  = 0 if accl_days<7 else 1 if accl_days<14 else 2  # 0-2
    hour_bin  = min(int(hour_on_shift / 2), 3)     # 0-3 (2hr blocks)
    return (risk_bin, wbgt_bin, accl_bin, hour_bin)

STATE_SHAPE = (5, 5, 3, 4)
Q_TABLE_SHAPE = STATE_SHAPE + (N_ACTIONS,)


# ══════════════════════════════════════════════════════════════
# REWARD FUNCTION
# Balances safety (primary) with productivity (secondary).
# Danger events are heavily penalized.
# Unnecessary interventions are lightly penalized.
# ══════════════════════════════════════════════════════════════
def compute_reward(risk_before, risk_after, action_id,
                    danger_event_occurred):
    action = ACTIONS[action_id]

    # Safety reward: risk reduction is good
    risk_reduction = risk_before - risk_after
    safety_reward  = risk_reduction * 0.5

    # Danger penalty: danger events are very costly
    danger_penalty = -100 if danger_event_occurred else 0

    # Productivity cost: unnecessary interventions waste time
    prod_cost = action["productivity_cost"] * 0.2

    # Proactive bonus: acting before danger is better than reacting
    proactive_bonus = 10 if (risk_before < 65 and action_id in [1,2,3,4]
                             and risk_after < 55) else 0

    return float(safety_reward + danger_penalty - prod_cost + proactive_bonus)


# ══════════════════════════════════════════════════════════════
# ENVIRONMENT SIMULATOR
# Simulates one worker's shift for RL training.
# ══════════════════════════════════════════════════════════════
class WorkerShiftEnvironment:
    def __init__(self, worker_id="W004"):
        self.worker_id  = worker_id
        self.wp         = WORKER_PROFILES[worker_id]
        self.reset()

    def reset(self):
        self.minute      = 0
        self.risk_score  = np.random.uniform(10, 30)
        self.wbgt        = 30.0
        self.accl_days   = self.wp["accl_days"]
        self.done        = False
        self.danger_occurred = False
        return discretize_state(self.risk_score, self.wbgt,
                                self.accl_days, self.minute/60)

    def step(self, action_id):
        action = ACTIONS[action_id]

        # Advance environment
        self.minute += 5  # 5-minute steps
        self.wbgt   = float(np.clip(
            self.wbgt + 0.05 + np.random.normal(0, 0.1), 28, 46))

        # Risk evolution
        natural_rise = (
            max(0, self.wbgt - 28) * 0.3 +
            (self.minute / 60) * 0.5 +
            (1 - min(1.0, self.accl_days/14)) * 2.0 +
            np.random.normal(0, 1.5)
        )
        risk_reduction = action["risk_reduction"] * np.random.uniform(0.7, 1.3)
        self.risk_score = float(np.clip(
            self.risk_score + natural_rise - risk_reduction, 0, 100))

        # Check danger
        danger_event = self.risk_score >= 85
        if danger_event:
            self.danger_occurred = True

        # Compute reward
        risk_before = self.risk_score + risk_reduction
        reward = compute_reward(risk_before, self.risk_score,
                                action_id, danger_event)

        self.done = self.minute >= 480
        next_state = discretize_state(
            self.risk_score, self.wbgt,
            self.accl_days, self.minute/60)

        return next_state, reward, self.done, {
            "risk_score": self.risk_score,
            "wbgt": self.wbgt,
            "danger": danger_event,
        }


# ══════════════════════════════════════════════════════════════
# Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════
class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.05):
        self.q_table      = np.zeros(Q_TABLE_SHAPE)
        self.alpha        = alpha        # learning rate
        self.gamma        = gamma        # discount factor
        self.epsilon      = epsilon      # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min  = epsilon_min
        self.episode_rewards = []

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        current_q  = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def best_action(self, state):
        """
        Returns best action name for a given state.

        IMPORTANT — safety fallback for untrained states:
        Testing this agent directly showed only ~12% of the 300
        possible (risk_bin, wbgt_bin, accl_bin, hour_bin) states ever
        get visited during training, and this barely improves even
        with far more episodes — it's a structural sparsity problem,
        not an undertrained-yet issue. Risk and hour-on-shift are
        naturally correlated in the training environment (risk climbs
        roughly monotonically through a shift), so most other
        combinations rarely or never occur.

        Every moderate-to-high-risk scenario tested (risk 60-72)
        landed on a completely untrained Q-row of all zeros.
        np.argmax([0,0,0,0,0,0]) mechanically returns index 0 — which
        happens to be CONTINUE — as a tie-breaking artifact of NumPy's
        argmax, not a learned safety decision. For a heat-stroke
        early-warning system, silently recommending "keep working"
        for a state the agent has never actually seen is exactly the
        failure mode this system exists to prevent.

        When the Q-row is genuinely untrained (all zeros), fall back
        to a simple, honest, risk-based heuristic instead of trusting
        the meaningless default. This is consistent with the risk
        band conventions already used elsewhere in this codebase
        (e.g. v2_digital_twin.py's danger_category thresholds).
        """
        q_row = self.q_table[state]
        if np.all(q_row == 0):
            risk_bin = state[0]  # 0-4, from discretize_state()
            fallback = {0: 0, 1: 0, 2: 1, 3: 2, 4: 5}[risk_bin]
            # 0-1 (risk<40): CONTINUE — 2 (40-60): INCREASE_BREAKS
            # 3 (60-80): REST_BREAK — 4 (80-100): EVACUATE
            return fallback, ACTIONS[fallback]["name"]
        action_id = int(np.argmax(q_row))
        return action_id, ACTIONS[action_id]["name"]


def train_agent(n_episodes=500, verbose=True):
    """Trains Q-learning agent across all worker profiles."""
    agent = QAgent()
    all_rewards = []

    # Train on highest-risk workers primarily
    training_workers = ["W004","W010","W002","W006","W001",
                        "W003","W005","W007","W008","W009"]

    for episode in range(n_episodes):
        wid = training_workers[episode % len(training_workers)]
        env = WorkerShiftEnvironment(wid)
        state = env.reset()
        total_reward = 0

        while not env.done:
            action  = agent.choose_action(state)
            next_st, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_st, done)
            state        = next_st
            total_reward += reward

        agent.decay_epsilon()
        all_rewards.append(total_reward)

        if verbose and (episode + 1) % 500 == 0:
            avg = np.mean(all_rewards[-100:])
            print(f"      Episode {episode+1:4d} | "
                  f"Avg reward: {avg:7.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, all_rewards


# ══════════════════════════════════════════════════════════════
# PERSONAL DANGER WINDOW PREDICTION
# Learns each worker's personal risk trajectory pattern.
# Ahmed always spikes at hour 5 — predict and prevent.
# ══════════════════════════════════════════════════════════════
def predict_personal_danger_window(worker_id, historical_risk_profile):
    """
    Identifies the typical hour range when this worker reaches
    peak risk based on historical shift patterns.

    Returns: (danger_window_start_hour, danger_window_end_hour,
               confidence, recommendation)
    """
    if len(historical_risk_profile) < 3:
        return None, None, 0.0, "Insufficient history"

    # Find average hour of peak risk across historical shifts
    peak_hours = []
    for shift_risks in historical_risk_profile:
        if len(shift_risks) > 0:
            peak_idx  = np.argmax(shift_risks)
            peak_hour = peak_idx / 60  # convert minutes to hours
            peak_hours.append(peak_hour)

    if not peak_hours:
        return None, None, 0.0, "No peak detected"

    avg_peak   = float(np.mean(peak_hours))
    std_peak   = float(np.std(peak_hours))
    confidence = float(np.clip(1.0 - std_peak / 4.0, 0.3, 0.95))

    window_start = max(0, avg_peak - 0.5)
    window_end   = min(8, avg_peak + 0.5)
    wp_name      = WORKER_PROFILES.get(worker_id, {}).get("name", worker_id)

    rec = (f"{wp_name} typically reaches peak risk between "
           f"{window_start:.1f}h-{window_end:.1f}h into shift. "
           f"Schedule heavy work before hour {window_start:.1f}. "
           f"Confidence: {confidence*100:.0f}%")

    return round(window_start, 1), round(window_end, 1), confidence, rec


# ══════════════════════════════════════════════════════════════
# PROACTIVE SCHEDULE GENERATOR
# Given current state → next 2 hours of recommendations
# ══════════════════════════════════════════════════════════════
def generate_proactive_schedule(agent, worker_id, current_risk,
                                 current_wbgt, minutes_on_shift,
                                 wbgt_forecast=None):
    """
    Generates proactive schedule for next 2 hours.
    Uses trained RL agent to recommend optimal actions.
    """
    wp       = WORKER_PROFILES.get(worker_id, {})
    accl     = wp.get("accl_days", 30)
    schedule = []

    sim_risk = current_risk
    sim_wbgt = current_wbgt
    sim_min  = minutes_on_shift

    for step in range(0, 120, 15):  # every 15 minutes
        future_min  = sim_min + step
        future_hour = future_min / 60

        # Use WBGT forecast if available
        if wbgt_forecast and step//15 < len(wbgt_forecast):
            sim_wbgt = wbgt_forecast[step//15]
        else:
            sim_wbgt = float(np.clip(sim_wbgt + 0.08*15, 25, 50))

        state      = discretize_state(sim_risk, sim_wbgt, accl, future_hour)
        action_id, action_name = agent.best_action(state)
        action     = ACTIONS[action_id]

        # Project risk after action
        natural_rise = max(0, sim_wbgt-28)*0.3 + 0.5 + (1-min(1,accl/14))*2
        projected    = float(np.clip(
            sim_risk + natural_rise - action["risk_reduction"], 0, 100))

        schedule.append({
            "minute":          future_min,
            "hour":            round(future_hour, 2),
            "projected_risk":  round(sim_risk, 1),
            "wbgt":            round(sim_wbgt, 1),
            "recommended_action": action_name,
            "risk_after_action":  round(projected, 1),
            "productivity_cost":  action["productivity_cost"],
            "rationale":          _action_rationale(
                action_name, sim_risk, sim_wbgt, wp.get("name",""), accl),
        })
        sim_risk = projected

    return schedule


def _action_rationale(action_name, risk, wbgt, name, accl):
    """Human-readable rationale for each scheduled action."""
    if action_name == "EVACUATE":
        return f"Critical risk {risk:.0f} — evacuate {name} immediately"
    elif action_name == "REST_BREAK":
        return (f"Risk {risk:.0f} — 15-min rest break reduces "
                f"cardiovascular strain before danger threshold")
    elif action_name == "ROTATE_ZONE":
        return f"WBGT {wbgt:.1f}°C — move {name} to shaded zone"
    elif action_name == "REDUCE_WORKLOAD":
        return f"Risk {risk:.0f} — lighter tasks for next 30 minutes"
    elif action_name == "INCREASE_BREAKS":
        return (f"Accl {accl}d — {'new worker' if accl<14 else 'precautionary'} "
                f"break schedule")
    return f"Risk {risk:.0f} — continue monitoring"


# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════
def plot_rl_analysis(all_rewards, schedules_by_worker):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("ProactiveGuard v2.0 — RL Scheduler Analysis",
                 color="#ecf0f1", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="#95a5a6", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2c3e50")
        ax.grid(True, alpha=0.15, color="#2c3e50")

    # Panel 1: Training reward curve
    ax1 = axes[0]
    window = 20
    smoothed = pd.Series(all_rewards).rolling(window, min_periods=1).mean()
    ax1.plot(range(len(all_rewards)), all_rewards,
             color="#3498db", alpha=0.3, linewidth=0.8)
    ax1.plot(range(len(smoothed)), smoothed,
             color="#e74c3c", linewidth=2, label=f"{window}-ep average")
    ax1.set_xlabel("Training Episode", color="#ecf0f1", fontsize=9)
    ax1.set_ylabel("Total Reward", color="#ecf0f1", fontsize=9)
    ax1.set_title("Q-Learning Training Convergence",
                  color="#ecf0f1", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, labelcolor="#95a5a6",
               facecolor="#1e2130", edgecolor="#2c3e50")

    # Panel 2: Proactive schedule for highest-risk worker
    ax2 = axes[1]
    wid = "W004"
    if wid in schedules_by_worker:
        sched  = schedules_by_worker[wid]
        hours  = [s["hour"] for s in sched]
        risks  = [s["projected_risk"] for s in sched]
        action_colors = {
            "CONTINUE":        "#2ecc71",
            "INCREASE_BREAKS": "#f39c12",
            "REST_BREAK":      "#e67e22",
            "ROTATE_ZONE":     "#3498db",
            "REDUCE_WORKLOAD": "#9b59b6",
            "EVACUATE":        "#e74c3c",
        }
        for i, s in enumerate(sched):
            color = action_colors.get(s["recommended_action"], "#95a5a6")
            ax2.bar(s["hour"], s["projected_risk"],
                    width=0.2, color=color, alpha=0.85)
            ax2.text(s["hour"], s["projected_risk"] + 1,
                     s["recommended_action"][:3],
                     ha="center", color=color, fontsize=6, rotation=45)

        ax2.axhline(70, color="#e74c3c", linestyle="--",
                    alpha=0.6, linewidth=1, label="Danger (70)")
        ax2.axhline(55, color="#f39c12", linestyle="--",
                    alpha=0.5, linewidth=1, label="Warning (55)")
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Hours into shift", color="#ecf0f1", fontsize=9)
        ax2.set_ylabel("Projected Risk", color="#ecf0f1", fontsize=9)
        name = WORKER_PROFILES[wid]["name"]
        ax2.set_title(f"Proactive Schedule — {name}\n"
                      f"(Actions prevent danger before it arrives)",
                      color="#ecf0f1", fontsize=9, fontweight="bold")
        ax2.legend(fontsize=7, labelcolor="#95a5a6",
                   facecolor="#1e2130", edgecolor="#2c3e50")

    plt.tight_layout()
    plt.savefig("outputs/v2_rl_analysis.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("  Saved: outputs/v2_rl_analysis.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("ProactiveGuard v2.0 — Module 7: RL Proactive Scheduler")
    print("Q-Learning | Personal Danger Windows | Schedule Optimization")
    print("=" * 65)

    # 1. Train agent
    print("\n[1/4] Training Q-learning agent (500 episodes)...")
    agent, rewards = train_agent(n_episodes=3000, verbose=True)
    print(f"\n      Final avg reward (last 100 ep): "
          f"{np.mean(rewards[-100:]):.1f}")
    print(f"      Final epsilon: {agent.epsilon:.4f}")

    # 2. Save agent
    # Saves only the learned q_table (a plain numpy array), not the
    # whole QAgent class instance.
    #
    # IMPORTANT — why not just pickle(agent) directly: when this
    # script runs via `python v2_rl_scheduler.py`, Python treats this
    # file as the __main__ module, and pickling a custom class
    # instance embeds a reference to "__main__.QAgent" in the saved
    # file. Any OTHER script that later unpickles it (like the
    # dashboard, run via `streamlit run v2_dashboard.py`) has its OWN,
    # different __main__ module with no QAgent class — confirmed this
    # exact failure in testing: "Can't get attribute 'QAgent' on
    # <module '__main__' from '...v2_dashboard.py'>".
    #
    # The tempting fix — forcing QAgent.__module__ = "v2_rl_scheduler"
    # right after the class definition — was tried and confirmed
    # BROKEN by isolated testing: it makes THIS SAME script's own
    # pickle.dump() call fail instead, because pickle then verifies
    # the class by freshly re-importing "v2_rl_scheduler" (a second,
    # separate execution of this same file under a different module
    # identity), producing a QAgent class that doesn't match the one
    # already running — a PicklingError.
    #
    # The robust fix: never pickle the custom class at all. The
    # agent's entire "brain" is just its q_table array — all the
    # QAgent methods (choose_action, best_action, etc.) are pure
    # functions of that array plus module-level constants, not
    # per-instance state. Any script that needs the trained agent
    # just does `agent = QAgent(); agent.q_table = <loaded array>` —
    # a fresh, fully-functional instance with zero pickling fragility.
    print("\n[2/4] Saving trained agent...")
    with open("models/v2_rl_agent.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("      Saved: models/v2_rl_agent.pkl")

    # 3. Generate schedules for all workers
    print("\n[3/4] Generating proactive schedules...")
    schedules_by_worker = {}
    all_recs = []

    # Real current conditions per worker, read from the actual digital
    # twin output -- not hand-typed demo numbers. Previously this was
    # a hardcoded dict of (risk, wbgt, minutes) tuples that never
    # changed regardless of what the real simulation produced (the
    # same fabricated-demo-data pattern already found and fixed in
    # v2_safety_infrastructure.py and v2_llm_engine.py this session).
    predictions_path = "data/v2_twin_predictions.csv"
    current_conditions = {}
    if os.path.exists(predictions_path):
        twin_df = pd.read_csv(predictions_path)
        for wid in twin_df["worker_id"].unique():
            w_data = twin_df[twin_df["worker_id"] == wid]
            peak_idx = w_data["danger_prob_now"].idxmax()
            peak_row = w_data.loc[peak_idx]
            current_conditions[wid] = (
                round(float(peak_row["danger_prob_now"]) * 100, 1),
                round(float(peak_row["wbgt"]), 1),
                int(peak_row["minutes_on_shift"]),
            )
    if not current_conditions:
        print(f"      ⚠️  {predictions_path} not found — run "
              f"v2_data_engine.py and v2_digital_twin.py first for "
              f"real per-worker conditions. Using illustrative "
              f"placeholder values for this demo run only.")
        current_conditions = {
            "W001": (48, 35.5, 180), "W002": (55, 36.2, 210),
            "W003": (40, 34.8, 195), "W004": (72, 38.1, 290),
            "W005": (32, 33.5, 165), "W006": (60, 37.0, 240),
            "W007": (45, 35.0, 200), "W008": (52, 36.5, 220),
            "W009": (28, 32.0, 180), "W010": (68, 37.8, 270),
        }

    for wid, (risk, wbgt, mins) in current_conditions.items():
        wp     = WORKER_PROFILES.get(wid, {})
        sched  = generate_proactive_schedule(
            agent, wid, risk, wbgt, mins)
        schedules_by_worker[wid] = sched

        # Find first non-CONTINUE action
        first_action = next(
            (s for s in sched if s["recommended_action"] != "CONTINUE"),
            sched[0])

        print(f"\n  {wp.get('name',''):25s} "
              f"risk={risk:3.0f} wbgt={wbgt:.1f}")
        print(f"    Next action: {first_action['recommended_action']:18s} "
              f"at hour {first_action['hour']:.1f}")
        print(f"    {first_action['rationale'][:65]}...")

        for s in sched:
            all_recs.append({
                "worker_id":    wid,
                "worker_name":  wp.get("name", wid),
                "minute":       s["minute"],
                "hour":         s["hour"],
                "projected_risk": s["projected_risk"],
                "action":       s["recommended_action"],
                "rationale":    s["rationale"],
            })

    # 4. Personal danger windows
    print("\n[4/4] Personal danger window analysis...")
    # Simulate historical shift data for each worker
    for wid in ["W004", "W010", "W002"]:
        historical = []
        for _ in range(5):
            wp   = WORKER_PROFILES[wid]
            accl = min(1.0, wp["accl_days"]/14)
            t    = np.arange(480)
            risk = np.clip(
                20 + t*0.08*(1-0.3*accl) + np.random.normal(0,3,480),
                0, 100)
            historical.append(risk.tolist())

        start, end, conf, rec = predict_personal_danger_window(
            wid, historical)
        print(f"\n  {WORKER_PROFILES[wid]['name']}:")
        print(f"    {rec}")

    # Save recommendations
    pd.DataFrame(all_recs).to_csv(
        "data/v2_schedule_recommendations.csv", index=False)

    # Plot
    plot_rl_analysis(rewards, schedules_by_worker)

    print("\n" + "=" * 65)
    print("MODULE 7 COMPLETE")
    print("=" * 65)
    print("  models/v2_rl_agent.pkl")
    print("  data/v2_schedule_recommendations.csv")
    print("  outputs/v2_rl_analysis.png")
    print(f"\n  Agent trained on : 500 episodes")
    print(f"  Workers covered  : {len(schedules_by_worker)}")
    print(f"  Final avg reward : {np.mean(rewards[-100:]):.1f}")
    print("\nNext: python v2_safety_infrastructure.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
