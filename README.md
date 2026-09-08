# Urban Layer Agent

Master's thesis prototype for AI-assisted preparation of geospatial inputs for
urban land-surface-temperature what-if simulations.

The system combines:

- a locally deployable open-source LLM for language understanding;
- a controlled conversational workflow with clarification and confirmation;
- LCZ-based reference predictor values;
- deterministic GIS validation and vector editing;
- Python implementations of Layer Alterator C1, C2, and C3 workflows.

The LLM does not invent predictor values and does not directly edit rasters.
Numerical values come from the reference table and GIS operations are performed
by deterministic Python functions.

## Current thesis-Agent workflow

1. Upload a polygon vector.
2. Select a stable zone ID (`polygon_id`, `fid`, `id`, or `name`).
3. Choose C1 LCZ replacement or C2/C3 direct percentage adjustment.
4. For C1, describe an optional overall goal or directly assign an LCZ type.
5. For C2/C3, directly specify the Polygon, predictor, direction, and percentage.
6. Let the Agent ask one important clarification question at a time until the
   requirement is sufficiently complete.
7. Review scored LCZ scenarios for C1, or the extracted percentage plan for C2/C3.
8. Confirm or revise each proposal.
9. Generate the workflow-specific vector, rules, and compact configuration.
10. Produce 12 modified raster layers.

C1 (`mask`) and a unified percentage-adjustment workflow are exposed to the
user. Internally, the latter is classified as C2 when all 12 predictors use
`pct`, or C3 when selected predictors use `pct` and the remainder use `none`.
Percentage execution follows the professor-provided
`layers_simulator.ipynb`: UCP layers are changed only inside each polygon and
the seven land-cover fractions are changed jointly and normalized per pixel.
The downstream LST model remains a future extension.

The percentage vector contains numeric attributes `TCH`, `IMD`, `BH`, `BSF`, `SVF`
(the professor sample's `SFV` spelling is also accepted), `F_AC`, `F_S`,
`F_M`, `F_BS`, `F_G`, `F_TV`, and `F_W`. Select **Percentage adjustment** in
the **Run Layer Alterator** panel. A task can contain confirmed percentage
plans for multiple polygons; polygons explicitly left unchanged keep their
original values. The Agent determines whether the generated rule is C2 or C3
from the combined confirmed predictor scope.

Example conversational percentage workflow:

```text
Increase F_TV in Polygon 7 by 20% and reduce F_G by 10%.
Confirm
Generate
```

The percentage workflow accepts direct expressions such as `F_TV +30%` and
natural-language equivalents. Explicitly applying PCT to all 12 predictors
creates C2; selecting a subset creates C3 and assigns NONE to the remainder.
The Agent extracts and validates user-provided percentages; it does not derive
them from an LCZ target or invent numeric percentages.

The fixed Milan input rasters are loaded automatically from
`data/simulation_inputs/ucps` and `data/simulation_inputs/lc_fractions`.
Normal users do not need to enter folder paths. The web interface retains an
**Advanced data paths** section for maintenance or deployment to another study
area. Because the GeoTIFF inputs total about 440 MB, they are excluded from Git
and must be copied into those folders once when deploying the project.

## Run the web application

```powershell
.\.venv\Scripts\Activate.ps1
python start_web.py
```

Default address: <http://127.0.0.1:8501>

## Run automated tests

```powershell
python -m pytest tests -q
```

The tests cover ID preservation, intent routing, professor-table retrieval,
multi-turn requirement elicitation, multi-polygon instructions, constraints,
undo/confirmation, scenario scoring, partial unchanged polygons, reference-value
validation, output validation, complete 12-raster C1/C2/C3 runs, per-polygon
percentage behavior, unchanged pixels outside polygons, NoData preservation,
and land-cover fraction normalization.

## Run all engineering verification

```powershell
python verify_project.py
```

This runs the automated tests and all small development-set evaluations. Their
scores are engineering regression indicators, not the final controlled thesis
experiment.

## Thesis documentation

- `docs/ARCHITECTURE.md`: controlled Agent design and authority boundaries.
- `docs/THESIS_AGENT_DEMO.md`: step-by-step professor demonstration.

## Run the baseline evaluation

```powershell
python evaluation\run_baseline.py
```

The report is written to `evaluation/baseline_report.json`. The included cases
are a small engineering baseline, not the final thesis experiment.

## Important directories

```text
src/agent/                    Conversation schemas, intent routing, state machine
src/layer_alterator_agent/   LCZ matching and input-vector generation
src/layer_alterator/         C1 masking and polygon-aware C2/C3 execution
src/web/                     Streamlit interface
data/reference/              LCZ predictor reference data
tests/                       Automated tests
evaluation/                  Baseline evaluation cases and runner
```

## Validate a C2 run

The validator prints one compact JSON report and checks the main C2 invariants:
all 12 outputs exist, pixels outside polygons are unchanged, raster grids are
preserved, and the seven fractions sum to one inside polygons.

```powershell
python evaluation\validate_c2_outputs.py `
  path\to\vector_pct.geojson `
  path\to\ucps `
  path\to\lc_fractions `
  path\to\c2_outputs
```

## Local files excluded from Git

`.env`, `.venv`, model files, logs, Chroma data, conversations, uploads, and
generated Layer Alterator outputs are intentionally not committed.
