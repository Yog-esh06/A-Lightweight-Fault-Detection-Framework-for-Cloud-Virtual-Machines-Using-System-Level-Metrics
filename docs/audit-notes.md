# ThermoTwin Codebase Audit Notes

Internal technical audit of `src/thermotwin/` packages, classes, functions, and inter-package wiring.

---

## 1. Package: `physics/`

### Role
Implements the core physical plant simulation, thermal ODE mechanics, cooling actuator dynamics, sensor observation noise/missingness, and counterfactual rollout models.

### Key Modules & Functions
- **`thermal_model.py`**:
  - `chain_coupling(n_racks, rack_zone, coefficient_w_per_k)`: Builds a symmetric nearest-neighbour inter-rack thermal conductance matrix strictly within zones to conserve energy.
  - `thermal_derivative(temperatures_c, it_power_w, disturbance_w, inlet_temperatures_c, conductance_w_per_k, coupling_w_per_k, capacitance_j_per_k)`: Evaluates the lumped ODE:
    $$C_i \frac{dT_i}{dt} = P_{\text{IT},i} + Q_{\text{dist},i} - H_i(m_i)(T_i - T_{\text{in},i}) + \sum_j K_{ij}(T_j - T_i)$$
  - `rk4_step(state, dt_s, derivative)`: Stable 4th-order Runge-Kutta numerical integrator with boundary stability checks ($<= 200 ^\circ\text{C}$).
- **`cooling_model.py`**:
  - `advance_actuator(actual, command, tau_s, slew_per_s, dt_s)`: Models first-order lag with hard slew rate limits for supply temperature and airflow.
  - `inlet_temperatures(rack_temperatures_c, supply_c, rack_zone, recirculation_fraction)`: Computes mixed inlet temperature combining zone supply air and bounded recirculated exhaust.
  - `capacity_limited_conductance(...)`: Computes airflow-driven conductance ($H_i \propto m_i^{0.72}$) scaled down if requested zone extraction exceeds chiller capacity.
  - `cooling_electrical_power(...)`: Calculates compressor power via empirical coefficient of performance (COP) and fan power via cubic airflow scaling ($P_{\text{fan}} \propto m^3$).
- **`workload.py`**:
  - `scenario_inputs(step, n_racks, scenario, rng)`: Generates diurnal sinusoidal workload profiles with bursts (`burst`), spatial skew (`skewed`), elevated ambient temperature (`hot_ambient`), and cooling degradation (`cooling_degradation`).
  - `allocate(offered, placement_offset)`: Applies zero-sum placement offsets with saturation clipping and tracks unserved capacity.
- **`observation.py`**:
  - `observe(temperatures_c, rng, noise_std_c, missing_probability)`: Injects zero-mean Gaussian noise and masks missing readings with `NaN`.
  - `correct_estimate(predicted_c, observed_c, valid, gain)`: Blends physics prediction with valid noisy sensor innovations using a bounded Kalman-style gain.
- **`plant.py`**:
  - `HiddenPlantState`: Stores evaluation-only true state (temperatures, degradation factor, unmodelled process disturbance) that is strictly hidden from control policies.
  - `DataCenterPlant`: Stateful simulation runtime. Manages actuators, executes RK4 substeps, applies validated actions, generates policy-visible defensive `Snapshot` instances, and records hidden truth history.
- **`twin.py`**:
  - `DigitalTwin`: Generates isolated, nominal plant clones from snapshots to run forward counterfactual rollouts for candidate evaluation without mutating live plant state.
  - `LearnedOnlyTwin`: Derivative twin substituting physics integration with a learned action transition model for ablation studies.

### System Wiring
- **Driven by**: Configuration dictionaries (`configs/default.yaml`, `configs/smoke.yaml`, `configs/research.yaml`).
- **Called by**: `data.generate`, `control.optimizer`, `experiments.runner`, `experiments.scenarios`, `services.simulation_service`.

---

## 2. Package: `data/`

### Role
Generates synthetic trajectories, enforces data schemas, and builds leak-free supervised tabular and sequence samples separated by chronological purge gaps.

### Key Modules & Functions
- **`schemas.py`**:
  - `SCHEMA_VERSION = "1.0"`, `validate_frame(frame)`: Enforces column presence, schema version, step uniqueness, and non-null constraints on controller-visible fields.
- **`splitting.py`**:
  - `chronological_ranges(n_steps, window_steps, max_horizon_steps)`: Divides trajectories into train (50%), validation (17%), calibration (16%), and test (17%) partitions separated by purge gaps of size $\text{window\_steps} + \text{max\_horizon\_steps}$ (e.g., $12 + 60 = 72$ minutes).
  - `assert_partition_local(input_start, origin, target_end, split)`: Ensures no sample lag window or future prediction target crosses split boundaries.
- **`features.py`**:
  - `build_supervised(frame, config)`: Constructs tabular features ($x \in \mathbb{R}^{15}$), recurrent sequence windows ($X \in \mathbb{R}^{12 \times 7}$), multi-horizon targets ($y \in \mathbb{R}^5$ for 5, 10, 15, 30, 60 min), and nominal ODE forecasts.
  - `_nominal_forecast(...)`: Computes closed-form analytic thermal equilibrium and exponential decay trajectories.
- **`generate.py`**:
  - `generate_dataset(config, output_path, scenarios, steps, progress)`: Spawns independent `SeedSequence` streams per scenario and generates standardized long-form CSV trajectories.

### System Wiring
- **Driven by**: `config["forecasting"]`, `config["topology"]`, `config["physics"]`.
- **Called by**: `workflows.prepare_project`, `cli.py`, `forecasting.train`.

---

## 3. Package: `forecasting/`

### Role
Implements multi-horizon temperature forecasting and uncertainty calibration across 7 distinct algorithmic paths.

### Key Modules & Functions
- **`baselines.py`**: `persistence_predict` (last known value repeat), `physics_predict` (nominal ODE).
- **`classical.py`**: `make_classical_models` (constructs scikit-learn pipelines for `ridge`, `random_forest`, and multi-output `gradient_boosting`).
- **`recurrent.py`**: `GRUForecaster` (PyTorch GRU with direct multi-horizon linear head), `TrainedGRU`, `train_gru` (Adam optimizer, MSE loss, validation early stopping).
- **`hybrid.py`**: `ResidualCorrector` (Ridge regression predicting ground-truth minus nominal physics prior $y - y_{\text{nominal}}$), `ActionTransitionModel` (pure ML step transition for ablation), `fit_operational_hybrid`, `fit_action_models`.
- **`uncertainty.py`**:
  - `IntervalCalibrator`: Conformal-style empirical quantile scaling on calibration residuals to guarantee nominal empirical coverage.
  - `HotspotCalibrator`: Calibrated logistic regression predicting the probability of exceeding $35\ ^\circ\text{C}$ within future horizon windows.
- **`train.py`**: `train_all` (coordinates training of all models, validation model selection, and calibration).
- **`evaluate.py`**: `timed_predict`, `regression_rows` (computes MAE, RMSE, max error, $R^2$), `uncertainty_rows` (coverage and interval width), `hotspot_metrics` (Brier score, ROC-AUC, precision, recall, F1).
- **`registry.py`**: `save_bundle`, `load_bundle` (atomic disk persistence with SHA-256 digests and environment manifest validation).

### System Wiring
- **Driven by**: `config["forecasting"]`.
- **Called by**: `workflows.py`, `control.policies`, `experiments.runner`, `services.simulation_service`.

---

## 4. Package: `control/`

### Role
Generates candidate control actions, enforces operational constraints, calculates multi-objective costs, and conducts MPC-style counterfactual optimization.

### Key Modules & Functions
- **`actions.py`**:
  - `generate_candidates(snapshot, config, allow_migration)`: Generates bounded discrete candidate actions: no-op, $\pm 0.5\ ^\circ\text{C}$ supply temp changes, $\pm 0.08$ airflow changes, rack-to-rack workload migration, combined cooling+migration, and maximum safe cooling.
- **`constraints.py`**:
  - `validate_action(snapshot, action, config)`: Validates actuator limits, slew bounds, rack workload capacities ($\le 1.0$), and migration cooldowns.
- **`objectives.py`**:
  - `score_rollout(result, action, config)`: Evaluates dimensionless normalized objective combining:
    1. Chiller and fan energy
    2. Hotspot threshold squared exceedance
    3. Hard-limit ($40\ ^\circ\text{C}$) squared penalty
    4. Inter-rack temperature imbalance
    5. Actuator movement penalty
    6. Workload migration cost
- **`optimizer.py`**:
  - `optimize(snapshot, candidates, twin, config, controller_name, use_residual)`: Runs defensive snapshot rollouts for all candidates inside `DigitalTwin`. Selects the minimum objective action among feasible rollouts; falls back to minimum hard-limit violation if all candidates breach safety.
- **`policies.py`**:
  - Implements `FixedPolicy`, `ReactivePolicy` (hysteresis), `PredictivePolicy` (`physics_mpc`, `hybrid_mpc`, `no_migration`, `learned_only`), and `ForecastRulePolicy`.
  - `make_policy(name, config, twin)`: Factory constructor.

### System Wiring
- **Driven by**: `config["control"]`, `config["thresholds"]`, `config["cooling"]`.
- **Called by**: `physics.plant`, `experiments.scenarios`, `services.simulation_service`.

---

## 5. Package: `experiments/`

### Role
Coordinates execution of 6 research experiments, ablation studies, statistical paired bootstrap testing, figure generation, and HTML report rendering.

### Key Modules & Functions
- **`metrics.py`**: Computes integrated energy (kWh), cooling-only PUE, peak temperatures, hotspot rack-minutes, hard-limit violations, and service fractions.
- **`statistics.py`**: `paired_bootstrap_interval(control_a, control_b, seed)` computes 95% bootstrap confidence intervals for paired differences.
- **`plots.py`**: Headless matplotlib routines for line and bar charts saved to disk as PNG/SVG.
- **`scenarios.py`**: `run_closed_loop(...)` executes simulation runs under specified scenarios and policies.
- **`ablation.py`**: Maps 5 ablation arms (`full_hybrid`, `no_residual_learning`, `no_physics`, `no_optimization`, `no_migration`).
- **`runner.py`**:
  - `_experiment_one`: 8 analytic, conservation, lag, and timestep verification checks.
  - `_experiment_two`: 7-model forecast benchmark across horizons on held-out test data.
  - `_experiment_three`: Multi-horizon accuracy, interval coverage, and hotspot event metrics.
  - `_experiment_four`: 4-controller closed loop across 5 scenarios with paired bootstrap intervals.
  - `_experiment_five`: Component ablation under burst scenario.
  - `_experiment_six`: Robustness to plant mismatch, sensor faults, degradation, and 6/12/24 rack scaling.
- **`report.py`**: `generate_report(...)` compiles self-contained standalone HTML report embedding all figures as base64 and data tables.

### System Wiring
- **Driven by**: `config["experiments"]`.
- **Called by**: `workflows.py`, `services.job_manager`, `cli.py`.

---

## 6. Package: `services/`

### Role
Background service layer managing live simulation sessions, background experiment jobs, and disk artifact management.

### Key Modules & Functions
- **`artifacts.py`**: Computes SHA-256 source fingerprints, saves atomic JSON, manages experiment manifests and run catalogs.
- **`simulation_service.py`**:
  - `SimulationSession`: Manages individual worker loops with thread synchronization.
  - `SimulationService`: Manages session lifecycles, real-time stepping, playback speeds, and provides state/forecast/decision snapshot endpoints.
- **`job_manager.py`**:
  - `JobManager`: Manages bounded `ThreadPoolExecutor` workers for long-running experiment jobs, persists job states, and supports cooperative cancellation.

### System Wiring
- **Called by**: `web.routes`, `web.__init__`.

---

## 7. Package: `web/`

### Role
Local Flask server, REST API endpoints, input validation, and vanilla HTML5/Canvas/CSS control room dashboard.

### Key Modules & Functions
- **`routes.py`**: REST endpoints:
  - `GET /`: Dashboard UI.
  - `GET /api/health`: Server & model status.
  - `GET /api/config`: Topology, thresholds, scenario & controller metadata.
  - `POST /api/simulations`: Create simulation session.
  - `GET /api/simulations/<id>`: Get live session snapshot and metrics.
  - `POST /api/simulations/<id>/control`: Play, pause, step, reset, speed controls.
  - `GET /api/simulations/<id>/forecast`: Causal multi-horizon forecast with prediction intervals.
  - `GET /api/simulations/<id>/decisions`: Candidate action ledger and objective breakdown.
  - `POST /api/jobs`, `GET /api/jobs`, `POST /api/jobs/<id>/cancel`: Job management.
  - `GET /api/runs`, `GET /api/runs/<id>`, `GET /api/runs/<id>/download/<file>`: Run history and artifact downloads.
- **`validation.py`**: Type and range validation for API parameters.
- **`static/js/`**:
  - `app.js`: Main UI orchestration, tab management, polling loops, simulation control.
  - `charts.js`: High-DPI canvas charting for temperature traces and forecast curves.
  - `components.js`: DOM rendering for rack maps, state indicators, and candidate tables.
  - `api.js`: Fetch wrapper.

### System Wiring
- **Entry Points**: `app.py`, `python -m thermotwin serve`.
