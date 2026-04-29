"""
AI Change Radar - Scenario Simulation Engine
=============================================

Runs a modified Friedkin-Johnsen opinion dynamics model on a synthetic
banking workforce to compare three AI rollout strategies.

Author:  Edoardo Miozzi - M.Sc. Management, University of Mannheim
Project: AI Change Radar - Commerzbank IDDP Assessment Center 2026
License: MIT
"""

import json
import random
import math
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX required: pip install networkx")

SEED = 42
N_STEPS = 50
ADOPTION_THRESHOLD = 0.70
AGENTS_FILE = "agents.json"
OUTPUT_FILE = "precomputed_results.json"

random.seed(SEED)

SCENARIOS = {
    "top_down": {
        "name": "Top-down Mandate",
        "description": "Management imposes the AI tool with a firm deadline. High initial pressure creates rapid compliance, but without training or champions the effect fades as the announcement loses momentum.",
        "color": "#E74C3C",
        "params": {
            "mgmt_signal_peak": 0.85, "mgmt_decay_rate": 0.05,
            "training_intensity": 0.05, "training_ramp_steps": 3,
            "champion_pct": 0.00, "champion_multiplier": 1.0,
            "innate_shift_rate": 0.001,
        },
    },
    "grassroots": {
        "name": "Grassroots Champions",
        "description": "Identify the top 10% most influential and tech-ready employees, train them intensively, and let adoption spread through peer networks. Slow start, sustained growth through organic diffusion.",
        "color": "#F39C12",
        "params": {
            "mgmt_signal_peak": 0.15, "mgmt_decay_rate": 0.01,
            "training_intensity": 0.85, "training_ramp_steps": 12,
            "champion_pct": 0.10, "champion_multiplier": 2.5,
            "innate_shift_rate": 0.006,
        },
    },
    "full_alignment": {
        "name": "Full Alignment",
        "description": "Management communicates actively and consistently, training is broadly available, and 15% are identified as champions. All three levers produce fast adoption that sustains over time.",
        "color": "#27AE60",
        "params": {
            "mgmt_signal_peak": 0.50, "mgmt_decay_rate": 0.012,
            "training_intensity": 0.75, "training_ramp_steps": 8,
            "champion_pct": 0.15, "champion_multiplier": 2.5,
            "innate_shift_rate": 0.005,
        },
    },
}

def build_network(agents, k=6, p=0.15):
    n = len(agents)
    G = nx.watts_strogatz_graph(n, k, p, seed=SEED)
    for i, a in enumerate(agents):
        G.nodes[i]["agent_id"] = a["id"]
        G.nodes[i]["role"] = a["role"]
        G.nodes[i]["department"] = a["department"]
        G.nodes[i]["network_influence"] = a["network_influence"]
    return G

def compute_layout(G):
    pos = nx.spring_layout(G, seed=SEED, k=1.5/math.sqrt(len(G)), iterations=60)
    return {n: {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4)} for n, p in pos.items()}

def select_champions(agents, pct):
    if pct <= 0: return set()
    n = max(1, int(len(agents) * pct))
    ranked = sorted(range(len(agents)), key=lambda i: agents[i]["champion_score"], reverse=True)
    return set(ranked[:n])

def run_scenario(agents, G, scenario_key):
    sc = SCENARIOS[scenario_key]
    p = sc["params"]
    n = len(agents)
    champion_set = select_champions(agents, p["champion_pct"])
    champion_ids = [agents[i]["id"] for i in champion_set]

    opinions = [a["initial_opinion"] for a in agents]
    innate = [a["initial_opinion"] for a in agents]
    suscept = [a["susceptibility"] for a in agents]
    mgmt_r = [a["mgmt_receptivity"] for a in agents]
    train_r = [a["training_responsiveness"] for a in agents]
    net_infl = [a["network_influence"] for a in agents]
    fatigue = [a["traits"]["change_fatigue"] for a in agents]

    for i in champion_set:
        opinions[i] = min(1.0, opinions[i] + 0.10 * train_r[i])

    adoption_curve = []
    node_opinions = {}

    def record(step):
        rate = sum(1 for o in opinions if o > ADOPTION_THRESHOLD) / n * 100
        adoption_curve.append(round(rate, 1))
        if step % 5 == 0 or step == N_STEPS:
            node_opinions[str(step)] = [round(o, 3) for o in opinions]

    record(0)

    for step in range(1, N_STEPS + 1):
        mgmt_signal = p["mgmt_signal_peak"] * math.exp(-p["mgmt_decay_rate"] * step)
        train_level = p["training_intensity"] * min(1.0, step / p["training_ramp_steps"])
        innate_shift = p["innate_shift_rate"]
        new_opinions = list(opinions)

        for i in range(n):
            neighbours = list(G.neighbors(i))
            if not neighbours: continue

            # 1. Peer influence
            total_weight, weighted_sum = 0.0, 0.0
            for j in neighbours:
                w = net_infl[j]
                if j in champion_set: w *= p["champion_multiplier"]
                weighted_sum += w * opinions[j]
                total_weight += w
            peer_mean = weighted_sum / total_weight if total_weight > 0 else opinions[i]

            # 2. Management nudge
            mgmt_nudge = mgmt_signal * mgmt_r[i] * 0.18

            # 3. Training nudge + innate shift
            if i in champion_set:
                train_nudge = train_level * train_r[i] * 0.15
                innate[i] = min(1.0, innate[i] + innate_shift * train_r[i])
            else:
                train_nudge = train_level * 0.12 * train_r[i] * 0.15
                innate[i] = min(1.0, innate[i] + innate_shift * 0.3 * train_r[i])

            # 4. Fatigue dampening
            damper = 1.0 - 0.20 * fatigue[i]
            mgmt_nudge *= damper
            train_nudge *= damper

            # 5. Composite external signal
            external = max(0.0, min(1.0, peer_mean + mgmt_nudge + train_nudge))

            # 6. Friedkin-Johnsen update
            s = suscept[i]
            new_opinions[i] = max(0.0, min(1.0, (1 - s) * innate[i] + s * external))

        opinions = new_opinions
        record(step)

    # KPIs
    final_adoption = adoption_curve[-1]
    peak_adoption = max(adoption_curve)
    peak_step = adoption_curve.index(peak_adoption)
    steps_to_50 = next((s for s, a in enumerate(adoption_curve) if a >= 50.0), N_STEPS)
    resistant = sum(1 for o in opinions if o <= 0.40)
    enthusiasts = sum(1 for o in opinions if o >= 0.80)
    avg_opinion = round(sum(opinions) / n, 3)

    dept_data = defaultdict(list)
    for i, a in enumerate(agents): dept_data[a["department"]].append(opinions[i])
    dept_rates = {d: round(sum(1 for o in ops if o > ADOPTION_THRESHOLD)/len(ops)*100, 1)
                  for d, ops in dept_data.items()}

    return {
        "scenario_key": scenario_key, "scenario_name": sc["name"],
        "description": sc["description"], "color": sc["color"], "params": p,
        "adoption_curve": adoption_curve,
        "final_opinions": [round(o, 3) for o in opinions],
        "node_opinions": node_opinions, "champions": champion_ids,
        "kpis": {
            "final_adoption_pct": final_adoption, "peak_adoption_pct": peak_adoption,
            "peak_step": peak_step, "steps_to_50_pct": steps_to_50,
            "resistant_count": resistant, "enthusiast_count": enthusiasts,
            "avg_final_opinion": avg_opinion, "dept_adoption_rates": dept_rates,
        },
    }

def load_agents(path=AGENTS_FILE):
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    return data["agents"] if isinstance(data, dict) and "agents" in data else data

def run_all_scenarios(agents):
    G = build_network(agents)
    layout = compute_layout(G)
    edges = [{"source": int(u), "target": int(v)} for u, v in G.edges()]
    results = {}
    for key in SCENARIOS:
        print(f"  Running: {SCENARIOS[key]['name']}...")
        results[key] = run_scenario(agents, G, key)
    return results, layout, edges

def build_output(agents, results, layout, edges):
    agent_summaries = [
        {"id": a["id"], "idx": i, "role": a["role"], "department": a["department"],
         "level": a["level"], "age": a["age"], "gender": a["gender"],
         "x": layout[i]["x"], "y": layout[i]["y"]}
        for i, a in enumerate(agents)
    ]
    comparison = {
        key: {"name": res["scenario_name"], "color": res["color"],
              "final_adoption": res["kpis"]["final_adoption_pct"],
              "peak_adoption": res["kpis"]["peak_adoption_pct"],
              "peak_step": res["kpis"]["peak_step"],
              "steps_to_50": res["kpis"]["steps_to_50_pct"],
              "resistant": res["kpis"]["resistant_count"],
              "enthusiasts": res["kpis"]["enthusiast_count"],
              "avg_opinion": res["kpis"]["avg_final_opinion"]}
        for key, res in results.items()
    }
    return {
        "metadata": {
            "project": "AI Change Radar", "subtitle": "Internal AI Rollout Simulator",
            "description": "Pre-computed simulation results comparing three AI adoption strategies.",
            "n_agents": len(agents), "n_steps": N_STEPS, "threshold": ADOPTION_THRESHOLD,
            "network": "Watts-Strogatz small-world (k=6, p=0.15)",
            "model": "Modified Friedkin-Johnsen with decaying mgmt signal, ramping training, innate shift, fatigue dampening.",
            "disclaimer": "Illustrative simulation for decision support. Not a prediction of actual outcomes.",
            "author": "Edoardo Miozzi", "institution": "University of Mannheim, M.Sc. Management 2025",
        },
        "agents": agent_summaries, "edges": edges,
        "scenarios": results, "comparison": comparison,
    }

def print_summary(results):
    print(f"\n{'='*60}")
    print(f"  AI CHANGE RADAR - Scenario Comparison")
    print(f"{'='*60}")
    for key in ["top_down", "grassroots", "full_alignment"]:
        r = results[key]; k = r["kpis"]; c = r["adoption_curve"]
        print(f"\n  {r['scenario_name']}  ({r['color']})")
        print(f"  {'-'*50}")
        print(f"    Final adoption:    {k['final_adoption_pct']:5.1f}%")
        print(f"    Peak adoption:     {k['peak_adoption_pct']:5.1f}%  (step {k['peak_step']})")
        print(f"    Steps to 50%:      {k['steps_to_50_pct']:3d}")
        print(f"    Resistant agents:  {k['resistant_count']:3d}")
        print(f"    Enthusiasts:       {k['enthusiast_count']:3d}")
        print(f"    Avg final opinion: {k['avg_final_opinion']:.3f}")
        if r["champions"]: print(f"    Champions:         {len(r['champions'])}")
        steps = [0, 5, 10, 15, 20, 30, 40, 50]
        vals = [f"{c[s]:5.1f}" for s in steps]
        print(f"    Curve: {', '.join(vals)}")
        print(f"    Dept adoption:")
        for dept, rate in sorted(k["dept_adoption_rates"].items(), key=lambda x: -x[1]):
            bar = chr(9608) * int(rate / 5)
            print(f"      {dept:<25s} {rate:5.1f}%  {bar}")
    td = results["top_down"]["kpis"]["final_adoption_pct"]
    gr = results["grassroots"]["kpis"]["final_adoption_pct"]
    fa = results["full_alignment"]["kpis"]["final_adoption_pct"]
    print(f"\n  {'='*50}")
    print(f"  Full Alignment ({fa:.0f}%) = {fa/max(td,0.1):.1f}x Top-down ({td:.0f}%)")
    print(f"  Grassroots ({gr:.0f}%) = {gr/max(td,0.1):.1f}x Top-down")
    print(f"  {'='*50}\n")

if __name__ == "__main__":
    print("\n  AI Change Radar - Simulation Engine")
    print("  " + "-" * 40)
    print("\n  Loading agents...")
    agents = load_agents()
    print(f"  Loaded {len(agents)} agents")
    init = sum(1 for a in agents if a["initial_opinion"] > ADOPTION_THRESHOLD)
    print(f"  Initial adoption: {init}% (threshold={ADOPTION_THRESHOLD})")
    print("\n  Building network & running simulations...")
    results, layout, edges = run_all_scenarios(agents)
    print_summary(results)
    print("  Assembling output...")
    output = build_output(agents, results, layout, edges)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {OUTPUT_FILE}")
    print("  Done. Ready for frontend.\n")
