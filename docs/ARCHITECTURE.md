# Urban Layer Agent architecture

```text
User message
  -> intent and flexible-language parsing
  -> requirement elicitation loop
  -> completeness and conflict checks
  -> professor-table retrieval
  -> scored scenarios and explicit limitations
  -> user confirmation
  -> deterministic GIS execution
  -> input and raster-output validation
  -> result evaluation or replanning advice
```

## Authority boundaries

- The language model may interpret wording and generate conversational text.
- `LCZ_urban_types_match.xlsx` is authoritative for planning-type/LCZ correspondence.
- `ref_predictor_values_mean.csv` is authoritative for predictor values.
- Python validates IDs, geometry, fractions, files, and raster alignment.
- C1/C2/C3 generation and execution require explicitly confirmed plans. C1
  values come from the LCZ reference table. C2/C3 percentages must be supplied
  explicitly by the user and are never inferred from an LCZ target.
- LCZ suitability is not evidence of legal, construction, pedestrian, or thermal
  performance feasibility. Those claims require additional datasets or models.

## Main modules

| Module | Responsibility |
|---|---|
| `requirements.py` | Persistent structured requirements |
| `elicitation.py` | Completeness checks and one-question-at-a-time clarification |
| `proposal_engine.py` | Candidate scoring, confidence, alternatives |
| `conversation_controller.py` | Multi-polygon and contextual commands |
| `knowledge_base.py` | Professor workbook access |
| `execution_manager.py` | Plans, generated-input checks, raster checks |
| `failure_guard.py` | Safe degradation and truthful failure reporting |
| `c2_dialogue.py` | Direct C2/C3 Polygon, predictor, direction, and percentage dialogue |
| `c2_vector_writer.py` | Confirmed C2/C3 decisions to GeoJSON and rules |

## Current research boundary

C1 deterministic masking and a unified direct-percentage workflow (internally
C2 for all-PCT instructions or C3 for partial-PCT/NONE instructions) are
implemented and tested. Actual urban-temperature improvement
requires the downstream LST model and suitable meteorological validation.
