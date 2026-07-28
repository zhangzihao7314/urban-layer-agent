# Professor notebook traceability

Source: `layers_simulator.ipynb`, supplied by the supervisor.

| Notebook responsibility | Project implementation | Status |
|---|---|---|
| C0-C5 rule classification | `src/layer_alterator/router.py` | Ported and unit-tested |
| C1 polygon mask replacement | `src/layer_alterator/masking.py`, `service.py` | Ported and end-to-end tested |
| C2 percentage modification | `src/layer_alterator/percentage.py` | Ported and unit-tested |
| C3 percentage + unchanged layers | `router.py`, `percentage.py` | Classification and numeric primitive tested |
| C4/C5 invalid combinations | `router.py` | Explicitly rejected |

The notebook is treated as the algorithmic reference, not as an imported runtime
dependency. Production code is extracted into testable Python modules. A future
experiment must compare both implementations using the same supervisor raster
fixtures; current tests verify the documented formulas and invariants.
