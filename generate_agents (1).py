"""
AI Change Radar - Synthetic Employee Generator
================================================

Generates a synthetic workforce of 100 banking employees for an
agent-based simulation comparing AI rollout strategies.

Each agent is calibrated on real German labor market data
(Bundesagentur fuer Arbeit, September 2025, Insurance & Finance sector)
and carries latent behavioural traits, survey responses, and model
parameters used by the Friedkin-Johnsen opinion dynamics engine.

Disclaimer
----------
Synthetic data for illustrative simulation purposes only.
Not representative of any real organisation's workforce.

Author:  Edoardo Miozzi - M.Sc. Management, University of Mannheim
Project: AI Change Radar - Commerzbank IDDP Assessment Center 2026
License: MIT
"""

import json
import random
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_AGENTS = 100
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Labour-market distribution (Bundesagentur fuer Arbeit, Sep 2025)
# ---------------------------------------------------------------------------

DISTRIBUTION = [
    ("25-35", "Skilled Worker", "M", True,  0.95),
    ("25-35", "Skilled Worker", "M", False, 8.16),
    ("25-35", "Skilled Worker", "F", True,  1.18),
    ("25-35", "Skilled Worker", "F", False, 7.07),
    ("25-35", "Specialist",    "M", True,  0.24),
    ("25-35", "Specialist",    "M", False, 1.77),
    ("25-35", "Specialist",    "F", True,  0.23),
    ("25-35", "Specialist",    "F", False, 1.08),
    ("25-35", "Expert",        "M", True,  1.32),
    ("25-35", "Expert",        "M", False, 4.90),
    ("25-35", "Expert",        "F", True,  1.18),
    ("25-35", "Expert",        "F", False, 2.50),
    ("35-45", "Skilled Worker", "M", True,  0.78),
    ("35-45", "Skilled Worker", "M", False, 8.90),
    ("35-45", "Skilled Worker", "F", True,  0.87),
    ("35-45", "Skilled Worker", "F", False, 4.53),
    ("35-45", "Specialist",    "M", True,  0.25),
    ("35-45", "Specialist",    "M", False, 2.37),
    ("35-45", "Specialist",    "F", True,  0.23),
    ("35-45", "Specialist",    "F", False, 1.04),
    ("35-45", "Expert",        "M", True,  1.13),
    ("35-45", "Expert",        "M", False, 5.32),
    ("35-45", "Expert",        "F", True,  0.89),
    ("35-45", "Expert",        "F", False, 2.10),
    ("45-55", "Skilled Worker", "M", True,  0.37),
    ("45-55", "Skilled Worker", "M", False, 8.79),
    ("45-55", "Skilled Worker", "F", True,  0.36),
    ("45-55", "Skilled Worker", "F", False, 3.44),
    ("45-55", "Specialist",    "M", True,  0.14),
    ("45-55", "Specialist",    "M", False, 2.44),
    ("45-55", "Specialist",    "F", True,  0.12),
    ("45-55", "Specialist",    "F", False, 0.91),
    ("45-55", "Expert",        "M", True,  0.39),
    ("45-55", "Expert",        "M", False, 4.24),
    ("45-55", "Expert",        "F", True,  0.26),
    ("45-55", "Expert",        "F", False, 1.26),
    ("55-65", "Skilled Worker", "M", True,  0.14),
    ("55-65", "Skilled Worker", "M", False, 7.44),
    ("55-65", "Skilled Worker", "F", True,  0.09),
    ("55-65", "Skilled Worker", "F", False, 3.02),
    ("55-65", "Specialist",    "M", True,  0.05),
    ("55-65", "Specialist",    "M", False, 2.28),
    ("55-65", "Specialist",    "F", True,  0.04),
    ("55-65", "Specialist",    "F", False, 0.83),
    ("55-65", "Expert",        "M", True,  0.11),
    ("55-65", "Expert",        "M", False, 3.36),
    ("55-65", "Expert",        "F", True,  0.06),
    ("55-65", "Expert",        "F", False, 0.87),
]

# ---------------------------------------------------------------------------
# 2. Role catalogue: (role, department, weight) per (age_group, level)
# ---------------------------------------------------------------------------

ROLE_MAP = {
    ("25-35", "Skilled Worker"): [
        ("Customer Advisor",      "Retail Banking",    30),
        ("Teller",                "Retail Banking",    20),
        ("Back Office Analyst",   "Operations",        20),
        ("KYC/AML Analyst",       "Compliance",        15),
        ("Operations Specialist", "Operations",        15),
    ],
    ("25-35", "Specialist"): [
        ("Credit Analyst",        "Risk Management",   25),
        ("Compliance Officer",    "Compliance",        25),
        ("Risk Analyst",          "Risk Management",   20),
        ("IT Support Specialist", "IT / Digital",      15),
        ("Junior Product Owner",  "IT / Digital",      15),
    ],
    ("25-35", "Expert"): [
        ("Data Analyst",          "Data & Analytics",  25),
        ("Data Scientist",        "Data & Analytics",  20),
        ("IT Project Manager",    "IT / Digital",      20),
        ("Product Owner Digital", "IT / Digital",      20),
        ("Comms Specialist",      "HR / Change Mgmt",  15),
    ],
    ("35-45", "Skilled Worker"): [
        ("Senior Customer Adv.",  "Retail Banking",    30),
        ("Relationship Manager",  "Corporate Banking", 25),
        ("Operations Specialist", "Operations",        20),
        ("Back Office Analyst",   "Operations",        15),
        ("Branch Service Coord.", "Retail Banking",    10),
    ],
    ("35-45", "Specialist"): [
        ("Sr. Compliance Officer","Compliance",        25),
        ("Senior Credit Analyst", "Risk Management",   25),
        ("IT Project Manager",    "IT / Digital",      20),
        ("Change Manager",        "HR / Change Mgmt",  15),
        ("Regulatory Affairs",    "Compliance",        15),
    ],
    ("35-45", "Expert"): [
        ("Senior Data Scientist", "Data & Analytics",  20),
        ("IT Architect",          "IT / Digital",      20),
        ("Senior Risk Manager",   "Risk Management",   20),
        ("Product Owner Digital", "IT / Digital",      20),
        ("Sr. Relationship Mgr.", "Corporate Banking", 20),
    ],
    ("45-55", "Skilled Worker"): [
        ("Sr. Relationship Mgr.", "Corporate Banking", 30),
        ("Branch Manager",        "Retail Banking",    25),
        ("Sr. Operations Spec.",  "Operations",        20),
        ("Client Service Manager","Retail Banking",    15),
        ("Back Office Team Lead", "Operations",        10),
    ],
    ("45-55", "Specialist"): [
        ("Head of Compliance",    "Compliance",        25),
        ("Sr. IT Project Manager","IT / Digital",      25),
        ("Head of Credit Analysis","Risk Management",  25),
        ("Senior Change Manager", "HR / Change Mgmt",  15),
        ("Process Excellence Mgr.","Operations",       10),
    ],
    ("45-55", "Expert"): [
        ("Department Head IT",    "IT / Digital",      25),
        ("Head of Risk Analytics","Risk Management",   25),
        ("Head of Data & Analyt.","Data & Analytics",  20),
        ("Head of Digital Banking","IT / Digital",     15),
        ("Senior Branch Director","Retail Banking",    15),
    ],
    ("55-65", "Skilled Worker"): [
        ("Senior Branch Manager", "Retail Banking",    35),
        ("Principal Client Adv.", "Corporate Banking", 25),
        ("Regional Ops Manager",  "Operations",        20),
        ("Senior Service Director","Retail Banking",   20),
    ],
    ("55-65", "Specialist"): [
        ("Head of Reg. Compliance","Compliance",       30),
        ("Senior IT Director",    "IT / Digital",      25),
        ("Head of Internal Audit","Risk Management",   25),
        ("Senior Change Director","HR / Change Mgmt",  20),
    ],
    ("55-65", "Expert"): [
        ("Divisional Director",   "Corporate Banking", 30),
        ("Head of Transformation","HR / Change Mgmt",  25),
        ("Chief Risk Analyst",    "Risk Management",   25),
        ("Senior Strategy Advisor","Corporate Banking",20),
    ],
}

FOREIGN_NATIONALITIES = [
    "Italian", "Turkish", "Polish", "Greek", "Spanish",
    "Romanian", "Croatian", "Indian", "French", "British",
]

# ---------------------------------------------------------------------------
# 3. Latent traits
# ---------------------------------------------------------------------------

def _clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))

def _likert(val):
    return round(max(1.0, min(5.0, val)), 1)

def generate_latent_traits(age, level, department, role):
    # --- Digital exposure ---
    base_de = 0.5
    if department in ("IT / Digital", "Data & Analytics"):
        base_de += 0.25
    if level == "Expert":
        base_de += 0.10
    elif level == "Skilled Worker":
        base_de -= 0.10
    base_de += (45 - age) / 80
    digital_exposure = _clamp(base_de + random.gauss(0, 0.15))

    # --- Change fatigue ---
    base_cf = 0.35
    if age > 50: base_cf += 0.15
    elif age > 40: base_cf += 0.05
    if department == "Operations": base_cf += 0.10
    if department == "HR / Change Mgmt": base_cf -= 0.10
    change_fatigue = _clamp(base_cf + random.gauss(0, 0.15))

    # --- Organizational trust ---
    base_ot = 0.55
    if level == "Expert": base_ot += 0.10
    if age > 45: base_ot += 0.10
    if department == "Compliance": base_ot += 0.05
    organizational_trust = _clamp(base_ot + random.gauss(0, 0.15))

    # --- Learning orientation ---
    base_lo = 0.50
    if department in ("IT / Digital", "Data & Analytics"): base_lo += 0.15
    if department == "HR / Change Mgmt": base_lo += 0.10
    if level == "Expert": base_lo += 0.10
    base_lo += (40 - age) / 120
    learning_orientation = _clamp(base_lo + random.gauss(0, 0.15))

    # --- Social influence ---
    base_si = 0.40
    if level == "Expert": base_si += 0.20
    elif level == "Specialist": base_si += 0.10
    if any(kw in role for kw in ("Manager", "Head", "Director", "Lead")):
        base_si += 0.15
    if "Relationship" in role: base_si += 0.10
    if department == "HR / Change Mgmt": base_si += 0.10
    social_influence = _clamp(base_si + random.gauss(0, 0.12))

    return {
        "digital_exposure":     round(digital_exposure, 3),
        "change_fatigue":       round(change_fatigue, 3),
        "organizational_trust": round(organizational_trust, 3),
        "learning_orientation": round(learning_orientation, 3),
        "social_influence":     round(social_influence, 3),
    }

# ---------------------------------------------------------------------------
# 4. Survey responses
# ---------------------------------------------------------------------------

def generate_survey(traits):
    de = traits["digital_exposure"]
    cf = traits["change_fatigue"]
    ot = traits["organizational_trust"]
    lo = traits["learning_orientation"]
    si = traits["social_influence"]

    return {
        "tech_comfort":         _likert(1 + 4*(0.6*de + 0.4*lo) + random.gauss(0, 0.3)),
        "ai_belief":            _likert(1 + 4*(0.4*de + 0.3*ot + 0.3*(1-cf)) + random.gauss(0, 0.3)),
        "mgmt_trust":           _likert(1 + 4*ot + random.gauss(0, 0.3)),
        "learning_willingness": _likert(1 + 4*(0.6*lo + 0.4*(1-cf)) + random.gauss(0, 0.3)),
        "peer_influence":       _likert(1 + 4*(0.5*si + 0.5*(1-de)) + random.gauss(0, 0.3)),
    }

# ---------------------------------------------------------------------------
# 5. Model parameters
# ---------------------------------------------------------------------------

def calculate_model_params(survey, traits, level, role):
    tc = survey["tech_comfort"]
    ab = survey["ai_belief"]
    mt = survey["mgmt_trust"]
    lw = survey["learning_willingness"]
    pi = survey["peer_influence"]
    de = traits["digital_exposure"]
    cf = traits["change_fatigue"]
    ot = traits["organizational_trust"]
    lo = traits["learning_orientation"]
    si = traits["social_influence"]

    initial_opinion = _clamp((0.35*tc + 0.40*ab + 0.25*lw) / 5.0)
    susceptibility = _clamp(pi / 5.0)
    mgmt_receptivity = _clamp((0.6*mt + 0.4*(ot*5)) / 5.0)

    level_b = {"Skilled Worker": 0.0, "Specialist": 0.15, "Expert": 0.30}[level]
    role_b = 0.0
    if any(kw in role for kw in ("Manager", "Head", "Director", "Lead")):
        role_b = 0.10
    if "Relationship" in role:
        role_b = max(role_b, 0.08)
    network_influence = _clamp(0.5*si + 0.3*level_b + 0.2*role_b + random.gauss(0, 0.05))

    training_responsiveness = _clamp((0.5*lw + 0.5*(de*5)) / 5.0)

    resistance_risk = _clamp(
        0.35*cf + 0.25*(1-ot) + 0.25*(1-ab/5.0) + 0.15*(1-lo)
    )

    champion_score = _clamp(
        0.30*initial_opinion + 0.30*network_influence
        + 0.25*training_responsiveness + 0.15*(1-resistance_risk)
    )

    return {
        "initial_opinion":         round(initial_opinion, 3),
        "susceptibility":          round(susceptibility, 3),
        "mgmt_receptivity":        round(mgmt_receptivity, 3),
        "network_influence":       round(network_influence, 3),
        "training_responsiveness": round(training_responsiveness, 3),
        "resistance_risk":         round(resistance_risk, 3),
        "champion_score":          round(champion_score, 3),
    }

# ---------------------------------------------------------------------------
# 6. Agent generation
# ---------------------------------------------------------------------------

def allocate_agents(n):
    total_pct = sum(row[4] for row in DISTRIBUTION)
    cells = []
    remainders = []
    for row in DISTRIBUTION:
        exact = (row[4] / total_pct) * n
        base = int(exact)
        cells.append((row, base))
        remainders.append((exact - base, len(cells) - 1))
    assigned = sum(c[1] for c in cells)
    remainders.sort(reverse=True)
    for i in range(n - assigned):
        idx = remainders[i][1]
        row, count = cells[idx]
        cells[idx] = (row, count + 1)
    return cells

def generate_agents(n=N_AGENTS):
    cells = allocate_agents(n)
    agents = []
    agent_id = 0
    age_bounds = {
        "25-35": (25, 34), "35-45": (35, 44),
        "45-55": (45, 54), "55-65": (55, 64),
    }

    for (age_group, level, gender_code, is_foreigner, _), count in cells:
        for _ in range(count):
            lo_age, hi_age = age_bounds[age_group]
            age = random.randint(lo_age, hi_age)
            gender = "Male" if gender_code == "M" else "Female"
            edu_end = (random.choice([25,26,27,28]) if level == "Expert"
                       else random.choice([22,23,24,25]))
            years_exp = max(1, age - edu_end)
            nationality = (random.choice(FOREIGN_NATIONALITIES)
                           if is_foreigner else "German")

            options = ROLE_MAP[(age_group, level)]
            names   = [r[0] for r in options]
            weights = [r[2] for r in options]
            idx = random.choices(range(len(options)), weights=weights, k=1)[0]
            role, department, _ = options[idx]

            traits = generate_latent_traits(age, level, department, role)
            survey = generate_survey(traits)
            params = calculate_model_params(survey, traits, level, role)

            agent = {
                "id": f"EMP_{agent_id:03d}",
                "age": age, "age_group": age_group,
                "gender": gender, "nationality": nationality,
                "is_foreigner": is_foreigner,
                "level": level, "department": department,
                "role": role, "years_experience": years_exp,
                "traits": traits, "survey": survey,
                **params,
                "is_champion": False,
            }
            agents.append(agent)
            agent_id += 1
    return agents

# ---------------------------------------------------------------------------
# 7. Summary & output
# ---------------------------------------------------------------------------

def print_summary(agents):
    n = len(agents)
    sep = "-" * 60
    print(f"\n{'='*60}")
    print(f"  AI CHANGE RADAR - Synthetic Workforce Summary")
    print(f"{'='*60}")
    print(f"  Agents: {n}  |  Seed: {SEED}")
    print(sep)

    print("\n  Age Distribution")
    for ag in ["25-35", "35-45", "45-55", "55-65"]:
        c = sum(1 for a in agents if a["age_group"] == ag)
        print(f"    {ag}:  {c:3d}  ({c/n*100:5.1f}%)")

    print(f"\n  Department Distribution")
    for dept, c in Counter(a["department"] for a in agents).most_common():
        print(f"    {dept:<25s} {c:3d}  ({c/n*100:5.1f}%)")

    print(f"\n  Level Distribution")
    for lvl in ["Skilled Worker", "Specialist", "Expert"]:
        c = sum(1 for a in agents if a["level"] == lvl)
        print(f"    {lvl:<16s} {c:3d}  ({c/n*100:5.1f}%)")

    print(f"\n  Gender Distribution")
    for g in ["Male", "Female"]:
        c = sum(1 for a in agents if a["gender"] == g)
        print(f"    {g:<8s} {c:3d}  ({c/n*100:5.1f}%)")

    foreign = sum(1 for a in agents if a["is_foreigner"])
    print(f"\n  Nationality")
    print(f"    German:   {n-foreign:3d}  ({(n-foreign)/n*100:5.1f}%)")
    print(f"    Foreign:  {foreign:3d}  ({foreign/n*100:5.1f}%)")

    print(f"\n  Top 12 Roles")
    for role, c in Counter(a["role"] for a in agents).most_common(12):
        print(f"    {role:<28s} {c:3d}")

    print(f"\n  Model Parameter Averages")
    for key in ["initial_opinion", "susceptibility", "mgmt_receptivity",
                "network_influence", "training_responsiveness", "resistance_risk"]:
        vals = [a[key] for a in agents]
        avg = sum(vals) / len(vals)
        print(f"    {key:<28s} {avg:.3f}")

    print(f"\n  Top 10 Champion Candidates")
    ranked = sorted(agents, key=lambda a: a["champion_score"], reverse=True)
    for a in ranked[:10]:
        print(f"    {a['id']}  {a['role']:<28s}  age={a['age']}  "
              f"dept={a['department']:<20s}  score={a['champion_score']:.3f}")
    print(f"\n{'='*60}\n")

def save_output(agents, path="agents.json"):
    output = {
        "metadata": {
            "project": "AI Change Radar",
            "description": "Synthetic employee population for AI adoption scenario simulation in a banking context.",
            "n_agents": len(agents),
            "source": "Demographic distribution calibrated on Bundesagentur fuer Arbeit data (Sep 2025, Insurance & Finance).",
            "disclaimer": "Synthetic data for illustrative simulation purposes only. Not representative of any real organisation's workforce.",
            "seed": SEED,
        },
        "agents": agents,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {path}")

# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agents = generate_agents(N_AGENTS)
    print_summary(agents)
    save_output(agents)
