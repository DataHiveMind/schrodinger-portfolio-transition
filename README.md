# Schrödinger-Portfolio-Transition: Score-Based Diffusion Control for Institutional Portfolio Rebalancing

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![kdb+/q 4.1](https://img.shields.io/badge/kdb%2B%2Fq-4.1-0052CC.svg)
![PyKX](https://img.shields.io/badge/PyKX-Zero--Copy_IPC-00B8D9.svg)
![PyTorch / JAX](https://img.shields.io/badge/Diffusion_SDE-PyTorch_%26_JAX-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-OMS_Execution_API-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-Pre%2FPost_TCA-0194E2.svg)

An institutional **Multi-Asset Portfolio Transition and Execution Platform** that solves large-scale regime rebalances as a **Dynamic Schrödinger Bridge / Stochastic Optimal Transport** problem. Unlike classical linear Almgren-Chriss trajectories that ignore cross-asset cannibalization and transient correlation spikes, this platform combines a **kdb+/q v4.1 Order-Flow Imbalance (OFI) Cross-Impact Engine** with **Score-Based Diffusion Stochastic Control (Iterative Proportional Fitting / Dynamic Sinkhorn)** to transport an initial institutional mandate distribution $\mu_0$ at $t = 0$ to a target mandate $\mu_T$ at $t = T$ along a minimum-impact, minimum-drawdown stochastic manifold.

---

## 1. Enterprise Usage Across Internal Teams

| Internal Team | Interface | Operational Workflow |
| :--- | :--- | :--- |
| **Central Execution Desk & OMS** | `services/api/` (`/v1/transition/compute-bridge` & `/v1/transition/update-mid-flight`) | Submits parent transition mandates $(\mathbf{q}_0, \mathbf{q}_T, T)$, receives time-sliced multi-asset child order trajectories persisted directly to the kdb+ `bridge_child_schedule` table, and re-solves mid-flight based on live fills. |
| **Multi-Asset Portfolio Managers** | `services/desk_ui/` (Transition Cockpit & Pre-Trade TCA) | Compares expected Implementation Shortfall (bps) and interim factor exposure of the Schrödinger Bridge path against multi-asset Almgren-Chriss and convex MPC baselines before approving a rebalance. |
| **Quantitative Execution Researchers** | `src/sb_transition/` (Internal SDK) | Imports `sb_transition` to benchmark custom cross-impact decay kernels, score SDE architectures, and Sinkhorn bridge solvers against historical L2 book partitions in kdb+. |
| **Transaction Cost Analysis (TCA) Team** | `mlops/post_trade_tca.py` & `cep_live_tca.q` | Audits realized child fills (`tca_fill` table in kdb+) against predicted pre-trade Schrödinger Bridge shortfall distributions. |

---

## 2. Mathematical Formulation

### 2.1 Multi-Asset Cross-Impact & Transient Price Dynamics
Consider an institutional desk transitioning an $N$-asset inventory vector $\mathbf{q}_t \in \mathbb{R}^N$ over horizon $t \in [0, T]$ with trading velocity $\mathbf{u}_t = d\mathbf{q}_t / dt$. Execution flow in Asset $i$ exerts both permanent and transient cross-impact on Asset $j$ via a non-diagonal **Kyle-Obizhaeva-Wang Cross-Impact Tensor** $\boldsymbol{\Lambda}_t \in \mathbb{R}^{N \times N}$ estimated in real time by `cep_cross_impact.q`:

$$d\mathbf{S}_t = \left( \boldsymbol{\mu}_t + \boldsymbol{\Gamma} \mathbf{u}_t \right) dt + \mathbf{G}_t \, d\mathbf{I}_t + \boldsymbol{\Sigma}_t^{1/2} \, d\mathbf{W}_t, \qquad d\mathbf{I}_t = -\boldsymbol{\Beta} \mathbf{I}_t \, dt + \boldsymbol{\Lambda}_t \mathbf{u}_t \, dt$$

where $\mathbf{I}_t \in \mathbb{R}^N$ denotes transient book resilience distortion, $\boldsymbol{\Lambda}_t$ captures cross-asset order-flow cannibalization, and $\boldsymbol{\Sigma}_t$ is the regime-dependent asset covariance matrix.

### 2.2 Dynamic Schrödinger Bridge Formulation
Rather than optimizing a rigid deterministic path that breaks under intraday liquidity shocks, we seek a controlled path measure $\mathbb{P}^{\mathbf{u}} \in \mathcal{P}(\Omega)$ over joint inventory-price state $\mathbf{x}_t = (\mathbf{q}_t, \mathbf{S}_t, \mathbf{I}_t)$ that transports the initial mandate marginal $\mu_0$ at $t=0$ to the exact target portfolio marginal $\mu_T$ at $t=T$ while minimizing relative entropy (Kullback-Leibler divergence) with respect to a liquidity-weighted reference diffusion $\mathbb{Q}$ plus cumulative execution and risk costs $V(\mathbf{x}_t, \mathbf{u}_t)$:

$$\min_{\mathbb{P}^{\mathbf{u}} \in \mathcal{D}(\mu_0, \mu_T)} \; D_{\text{KL}}\left(\mathbb{P}^{\mathbf{u}} \,\Vert{}\, \mathbb{Q}\right) + \mathbb{E}_{\mathbb{P}^{\mathbf{u}}} \left[ \int_0^T \left( \frac{1}{2} \mathbf{u}_t^\top \boldsymbol{\Lambda}_t \mathbf{u}_t + \frac{\eta}{2} (\mathbf{q}_t - \mathbf{q}_T)^\top \boldsymbol{\Sigma}_t (\mathbf{q}_t - \mathbf{q}_T) \right) dt \right]$$

### 2.3 Forward-Backward Score SDEs & Iterative Proportional Fitting (IPF)
By Girsanov's Theorem and the Hopf-Cole transformation, the optimal transition trajectory is governed by a coupled pair of **Forward-Backward Stochastic Differential Equations (FBSDEs)** parameterized by forward and backward neural score networks $\mathbf{s}_\theta^+(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}} \log \Psi(\mathbf{x}_t, t)$ and $\mathbf{s}_\phi^-(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}} \log \widehat{\Psi}(\mathbf{x}_t, t)$:

$$\text{Forward Controlled SDE:} \quad d\mathbf{x}_t = \left[ \mathbf{f}(\mathbf{x}_t, t) + \mathbf{g}(t)\mathbf{g}(t)^\top \mathbf{s}_\theta^+(\mathbf{x}_t, t) \right] dt + \mathbf{g}(t) \, d\mathbf{W}_t^+$$

$$\text{Backward Adjoint SDE:} \quad d\mathbf{x}_t = \left[ \mathbf{f}(\mathbf{x}_t, t) - \mathbf{g}(t)\mathbf{g}(t)^\top \mathbf{s}_\phi^-(\mathbf{x}_t, t) \right] dt + \mathbf{g}(t) \, d\mathbf{W}_t^-$$

Alternating Half-Bridge projections (**Iterative Proportional Fitting / Dynamic Sinkhorn**) guarantee exact boundary satisfaction $\mathbf{x}_0 \sim \mu_0$ and $\mathbf{x}_T \sim \mu_T$, routing trades through high-liquidity correlation corridors before unwinding unhedged residual legs.

---

## 3. kdb+/q v4.1 & PyKX Tick-Plant Architecture

* **`cep_cross_impact.q (:5013)`:** Ingests L2 limit order book ticks (`l2_book_tick`) across equities, futures, and FX to compute 1-second bucketed **Order Flow Imbalance ($\text{OFI}_{i,t}$)** vectors and updates rolling $N \times N$ cross-impact matrices $\boldsymbol{\Lambda}_t$ and covariance tensors $\boldsymbol{\Sigma}_t$ in native q vector math.
* **`cep_live_tca.q (:5013)`:** Performs real-time as-of joins (`aj`) comparing incoming OMS child fills (`tca_fill`) against parent arrival prices (`parent_mandate`) and the planned Schrödinger trajectory (`bridge_child_schedule`).
* **`oms_kdb_connector.py`:** Uses `pykx.AsyncQConnection` to stream live cross-impact tensors from `gw.q (:5015)` into the Score SDE controller and write updated child execution schedules back to kdb+ in under 5 milliseconds.

---

## 4. Repository Structure

```text
schrodinger-portfolio-transition/
├── .devcontainer/                 # VS Code Dev Container (Python 3.12, PyKX, PyTorch/JAX, cvxpy, kdb+ 4.1)
├── .vscode/                       # Debug configs for IPF Bridge Solver, Score SDE, and OMS API
├── configs/                       # Hydra configs (impact kernels, IPF steps, asset universes)
├── data/                          # Schemas for covariance tensors & sample transition mandates
├── kdb_database/                  # Multi-Asset Execution & TCA kdb+/q Tick Plant
│   ├── schema.q                   # Tables: l2_book_tick, parent_mandate, bridge_child_schedule, tca_fill
│   └── scripts/                   # cep_cross_impact.q, cep_volume_surface.q, cep_live_tca.q
├── src/sb_transition/             # Installable internal Python SDK
│   ├── kdb_bridge/                # PyKX OMS connector, zero-copy tensor loader, and HDB replay stream
│   ├── market_dynamics/           # Cross-asset Kyle/Obizhaeva-Wang impact & regime covariance
│   ├── bridge_solvers/            # Score SDE networks, IPF Sinkhorn solver, and drift controller
│   ├── benchmarks/                # Multi-asset Almgren-Chriss & rolling Convex MPC (cvxpy) baselines
│   └── analytics/                 # Implementation Shortfall (IS) & interim factor exposure diagnostics
├── services/
│   ├── api/                       # FastAPI OMS server (/v1/transition/compute-bridge, /v1/transition/update-mid-flight)
│   └── desk_ui/                   # Transition Cockpit & Pre/Post-Trade TCA web portal
├── mlops/                         # MLflow registry & automated EOD Post-Trade TCA reconciliation
├── tests/                         # Boundary marginal verification (mu_0 -> mu_T) & q OFI tensor tests
├── docker/                        # Multi-stage Dockerfiles & docker-compose stack
├── Makefile                       # Standardized operational entrypoints
├── environment.yml                # Conda environment specification
└── pyproject.toml                 # Package metadata & CLI entrypoint (`sb-transit`)
```

---

## 5. Quickstart & Production Operations

### 5.1 Setup & Initialization
```bash
conda env create -f environment.yml
conda activate sb_transition_env
pip install -e ".[dev]"
export QLIC=/opt/kx/lic
```

### 5.2 Launching the Execution Tick Plant, Bridge Solver & Transition Cockpit
```bash
# 1. Start kdb+ Tickerplant, RDB, HDB, Cross-Impact CEP Engine, and Gateway
make kdb-up

# 2. Train the Forward/Backward Score SDE networks via IPF on historical L2 partitions
sb-transit train asset_universes=sp500_sector_etfs diffusion_bridge.ipf_iterations=20

# 3. Start the OMS Execution API (:8002) and PM Transition Cockpit (:8503)
make serve-all
```

### 5.3 Example FastAPI Request (OMS Mandate Submission)
```bash
curl -X POST "http://localhost:8002/v1/transition/compute-bridge" \
  -H "Content-Type: application/json" \
  -d '{
    "mandate_id": "MANDATE-2026-1108-A",
    "symbols": ["SPY", "QQQ", "TLT", "GLD", "HYG"],
    "initial_weights": [0.45, 0.35, 0.05, 0.05, 0.10],
    "target_weights":  [0.15, 0.10, 0.45, 0.25, 0.05],
    "notional_usd": 150000000.0,
    "horizon_hours": 13.0,
    "risk_aversion_eta": 2.5
  }'
```