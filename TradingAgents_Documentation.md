### 📄 c:\Users\ELaunch\OneDrive\elaunch_projects\TradingAgents\.env
*Saved at: 6/23/2026, 4:00:04 PM*

**[REMOVED]**
```
(from line ~91)
TRADINGAGENTS_KRONOS_ENABLED=0

```
**[ADDED]**
```
91    TRADINGAGENTS_KRONOS_ENABLED=1
```

---

### 📄 c:\Users\ELaunch\OneDrive\elaunch_projects\TradingAgents\.env
*Saved at: 6/23/2026, 4:00:02 PM*

**[REMOVED]**
```
(from line ~91)
#TRADINGAGENTS_KRONOS_ENABLED=0

```
**[ADDED]**
```
91    TRADINGAGENTS_KRONOS_ENABLED=0
```

---

### 📄 c:\Users\ELaunch\OneDrive\elaunch_projects\TradingAgents\.env
*Saved at: 6/23/2026, 3:59:55 PM*

**[ADDED]**
```
86    
87    # Kronos (opt-in): a zero-shot generative foundation model used as a SECOND
88    # directional prior, fused like the quant brain. Heavy — install once:
89    # pip install ".[kronos]" (torch etc.; downloads ~100MB from HuggingFace on first
90    # use). 1 turns it on.
91    #TRADINGAGENTS_KRONOS_ENABLED=0
92    # Weight on Kronos when fusing with the agents (1.0 = Kronos-only, 0.0 = agents).
93    #TRADINGAGENTS_KRONOS_FUSION_WEIGHT=0.6
94    # Price paths sampled per forecast (more = smoother P(up) but slower on CPU).
95    #TRADINGAGENTS_KRONOS_SAMPLES=30
96    # Recent 5m bars fed to Kronos (<= the model's 512-bar context).
97    #TRADINGAGENTS_KRONOS_LOOKBACK=512
98    # Model id + device (cpu / cuda:0). Kronos-small ~100MB, CPU-friendly.
99    #TRADINGAGENTS_KRONOS_MODEL=NeoQuasar/Kronos-small
100   #TRADINGAGENTS_KRONOS_DEVICE=cpu
```

---

