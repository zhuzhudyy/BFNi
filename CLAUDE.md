# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Virtual environment**: Always activate before running Python:
```bash
source .venv/Scripts/activate
```

**Data & training pipeline** (run sequentially):
```bash
python data_builder.py           # Build Optimized_Training_Data.csv from EBSD data
python check_data.py             # Quick data quality check
python contextual_bo_model.py    # Train GPR model (interactive scheme selection)
python bo_optimization/honest_visualization.py  # Full visualization suite (9 plots, parity + PDP + ICE + ARD)
python predict_new_sample.py     # Predict optimal process for a new sample
python bo_optimization/space_filling_plan.py  # Batch experiment allocation for global model improvement
python bo_optimization/model_health_dashboard.py  # 2x2 model health dashboard (RMSE, pairplot, variance, coverage)
```

**Literature mining** (package: `lit_mining`):
```bash
python -m lit_mining.pipeline --mode search   # Search → export download_list.txt
python -m lit_mining.pipeline --mode local    # Extract local PDFs in lit_mining/local_pdfs/
python -m lit_mining.pipeline --mode fuse     # Fuse extractions → training CSV
python -m lit_mining.pipeline --mode full     # Full auto pipeline (OA papers only)
```

**Environment**: Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (used as DeepSeek key for Paratera API).

## Architecture

### Bayesian Optimization Core

`contextual_bo_model.py` contains the central class `ContextualBayesianOptimizer`:

- **Feature space** (order matters everywhere): `Pre_*` (EBSD, ~21 cols) + `Target_{hkl}` (multi-hot, 9 cols) + `Process_*` (4 cols: Temp, Time, H2, Ar) → target `TARGET_Yield`
- **Process constraint**: Ar >= 2 * H2 (氩气流量 ≥ 2倍氢气流量). Enforced in `suggest_next()` via random sample filtering + penalty in L-BFGS-B.
- **Surrogate model**: `GaussianProcessRegressor` with `ConstantKernel * Matern(nu=2.5, ARD=True) + WhiteKernel`. Length scales bounded to (1.0, 100.0) to prevent "length scale collapse" on small datasets.
- **ARD**: Each feature gets independent length scale — smaller = more important. Lower bound of 1.0 forces spatial smoothness.
- **Acquisition**: Expected Improvement (EI) using **local y_best** (within same target orientation scheme, not global). y_best is winsorized to P95 + k-nearest-neighbor context-aware in Pre_ feature space.
- **Optimization**: Two-step — 100K Latin hypercube random samples → Top-5 refine with L-BFGS-B.
- **Multi-task**: Each experiment is expanded into 4 rows (one per target orientation scheme). 4 schemes map to specific crystal orientation combinations.
- **Persistence**: `save_model()` / `load_model()` via pickle, stored in `trained_models/`.

### Data Builder

`data_builder.py` reads raw experimental data from `D:\毕业设计\织构数据\数据总结\`:

- Each experiment folder contains: `condition.{xls,csv}`, `pre*.csv`, `done*.csv` (supports multiple numbered files like `pre1.csv`, `pre2.csv`).
- `extract_macro_rgb_features()`: Parses AZtecCrystal CSV exports (IPF RGB colors + GND/Half quadratic data) → 21 Pre_ features. Uses text-cutting to skip multi-line AZtec headers.
- `hkl_to_aztec_rgb()`: Converts Miller indices directly to IPF-Z RGB colors via stereographic projection.
- Yield = fraction of pixels whose IPF color matches any target orientation within `tolerance=80` (RGB Euclidean distance).
- `build_training_dataset_multi_target()`: Iterates all folders, for each computes yields for all 4 schemes → 1 experiment × 4 rows.

### Visualization & Analysis

- `predict_new_sample.py`: Interactive pipeline for recommending process parameters. Supports loading pre-trained model or retraining.
- `bo_optimization/honest_visualization.py`: Physically honest visualization suite (9 plots in 4 tiers). Tier 0: LOOCV parity plot. Tier 1: PDP (marginalizing over Pre_ distribution), yield bar charts, raw scatter. Tier 2: ICE curves, Pre_ feature space, ARD feature importance (dual-panel: ranked bar + log-scale violin). Tier 3: uncertainty heatmap with Ar≥2H₂ constraint mask, process→Pre_ mediation analysis.

### Literature Mining (`lit_mining/`)

Package with 4 modules:

1. `searcher.py` — `LiteratureSearcher`: Queries Semantic Scholar API with predefined English + Chinese search queries. Caches to `lit_mining/cache/papers_metadata.json`. Exports `download_list.txt` with DOI links.
2. `extractor.py` — `PaperDataExtractor`: Downloads PDF → extracts text via PyMuPDF (`fitz`) → invokes LLM (Anthropic or OpenAI-compatible, configured in `config.py`) with a JSON schema prompt to get structured `{material, experiments: [{process, outcome}]}`.
3. `feature_fusion.py` — `FeatureFusion`: Maps LLM-extracted JSON fields to the training feature schema (unit conversions, GND normalization). Imputes missing Pre_ features using real data distribution statistics. Quality scores each row.
4. `pipeline.py` — `LiteratureMiningPipeline`: CLI orchestrator with modes `search/extract/local/fuse/full/stepwise`. Recommended workflow: search → manual PDF download via university library → local extraction → fuse with existing training data.

Configuration is centralized in `lit_mining/config.py` (search queries, API endpoints, model names, field mappings).

### Key Design Decisions

- **GPR length scale bounds (0.3, 5.0)**: Calibrated to StandardScaler-normalized ranges (~[-3,+3]). Lower bound 0.3 prevents overfitting; upper bound 5.0 ensures process parameters remain visible (原 100.0 上界导致 exp(-(Δx/100)²)≈1，所有工艺参数被模型忽略).
- **LOOCV for evaluation**: Used because N is small (~44 samples). GPR's built-in `.predict()` on training data is exact interpolation (useless for evaluation).
- **Winsorized y_best**: Uses P95 instead of max to prevent a single outlier yield from dominating EI.
- **Multi-hot target encoding**: A scheme activates multiple orientations simultaneously, enabling multi-objective optimization.
- **Color-tolerance-based yield**: Yield = |RGB_pixel - RGB_target| < 80, aggregating across all orientations in a scheme (any match counts).
