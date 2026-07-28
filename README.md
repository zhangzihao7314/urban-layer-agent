# Urban Layer Agent

Master's thesis prototype for AI-assisted preparation of geospatial inputs for
urban land-surface-temperature what-if simulations.

The system combines:

- a locally deployable open-source LLM for language understanding;
- a controlled conversational workflow with clarification and confirmation;
- LCZ-based reference predictor values;
- deterministic GIS validation and vector editing;
- a Python implementation of the Layer Alterator C1 masking workflow.

The LLM does not invent predictor values and does not directly edit rasters.
Numerical values come from the reference table and GIS operations are performed
by deterministic Python functions.

## Current MVP workflow

1. Upload a polygon vector.
2. Select a stable zone ID (`polygon_id`, `fid`, `id`, or `name`).
3. Describe the overall simulation goal.
4. Describe a target transformation for each polygon.
5. Clarify ambiguous descriptions.
6. Review the proposed LCZ urban type and reference values.
7. Confirm or revise each proposal.
8. Generate `updated_vector.geojson` and `rules.json`.
9. Run the C1 Layer Alterator against the UCP and fraction raster folders.
10. Produce 12 modified raster layers.

Only C1 (`mask`) is implemented in the thesis MVP. C2/C3 percentage workflows
and the downstream LST model are future extensions.

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

The tests cover ID preservation, intent routing, typology matching, clarification,
confirmation, reference-value validation, and a complete 12-raster C1 run.

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
src/layer_alterator/         C1 validation and raster masking
src/web/                     Streamlit interface
data/reference/              LCZ predictor reference data
tests/                       Automated tests
evaluation/                  Baseline evaluation cases and runner
```

## Local files excluded from Git

`.env`, `.venv`, model files, logs, Chroma data, conversations, uploads, and
generated Layer Alterator outputs are intentionally not committed.
