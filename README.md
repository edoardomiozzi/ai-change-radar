 AI Change Radar

Internal AI Rollout Simulator — How should a bank roll out an internal AI tool?

An agent-based simulation comparing three adoption strategies for internal AI tools in a banking organisation. Built as a decision-support prototype for the Commerzbank IDDP Assessment Center 2026.

What It Does: 100 synthetic bank employees — from Junior Tellers to Department Heads — are connected in a social network. Their opinions about a new AI tool evolve over 50 time steps based on peer influence, management pressure, and training. Three rollout strategies produce dramatically different outcomes:

| Strategy | Final Adoption | Peak | Shape |
|---|---|---|---|
| **Top-down Mandate** | 41% | 99% (step 6) | 📈📉 Spike then crash |
| **Grassroots Champions** | 82% | 82% (step 48) | 📈 Slow, sustained growth |
| **Full Alignment** | 93% | 96% (step 9) | 📈 Fast and sustained |

Key finding: Full Alignment delivers significantly higher adoption compared to a Top-down Mandate. 

## How It Works

### Model
Modified **Friedkin-Johnsen opinion dynamics** with three influence channels:

opinion(t+1) = (1-s) × innate(t) + s × [peer_mean + mgmt_nudge + train_nudge]

- **Management signal** decays exponentially (announcement fatigue)
- **Training effect** ramps up over time and permanently shifts innate opinions
- **Champions** amplify peer influence by 2.5× their network weight
- **Change fatigue** dampens management and training effects

### Data
Synthetic workforce calibrated on real German labor market data from **Bundesagentur für Arbeit** (September 2025, Insurance & Finance sector). Demographics, seniority levels, and department distribution reflect actual workforce composition.

### Agent Generation
Each agent carries:
- 5 latent behavioural traits (digital exposure, change fatigue, org trust, learning orientation, social influence)
- 5 survey responses derived from traits (tech comfort, AI belief, mgmt trust, learning willingness, peer influence)
- 7 model parameters (initial opinion, susceptibility, mgmt receptivity, network influence, training responsiveness, resistance risk, champion score)

## Project Structure

```
ai-change-radar/
├── index.html                  # Frontend (single-page app)
├── precomputed_results.json    # Pre-computed simulation data
├── generate_agents.py          # Synthetic employee generator
├── simulate.py                 # Scenario simulation engine
├── agents.json                 # Generated agent population
└── README.md
```

## Run Locally

```bash
# 1. Generate agents
python generate_agents.py

# 2. Run simulations
pip install networkx
python simulate.py

# 3. Open frontend
# Serve locally (needed for fetch to work):
python -m http.server 8000
# Open http://localhost:8000
```

## Deploy to GitHub Pages

```bash
git init
git add .
git commit -m "AI Change Radar v1.0"
git remote add origin https://github.com/YOUR_USERNAME/ai-change-radar.git
git push -u origin main
# Then enable GitHub Pages in repo Settings → Pages → Source: main branch
```

## Disclaimer

Synthetic data for illustrative simulation purposes only. Not representative of any real organisation's workforce. Not a prediction of actual outcomes.

## Author

**Edoardo Miozzi** 

Built for the Commerzbank IDDP Assessment Center, May 2026.
