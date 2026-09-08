# Professor demonstration script

1. Run `python start_web.py`.
2. Upload `test.geojson`.
3. State: `I want to reduce urban heat while keeping recreational space.`
4. Ask: `Polygon 7 should become a pleasant park.`
5. Answer the Agent:
   - `Cooling and recreation.`
   - `Scattered trees with some shade, and keep it open.`
6. Review the requirement summary, scored scenarios, confidence, evidence, and
   external-data limitations.
7. Ask `Why not Dense trees?`.
8. Confirm the preferred option.
9. Ask: `Make Polygon 8 a grass park and leave Polygon 9 unchanged.`
10. Review and confirm the plan.
11. Generate Layer Alterator inputs and inspect the validation report.
12. Run C1, or select **C2/C3 — direct percentage adjustment** and enter, for
    example, `Increase F_TV in Polygon 7 by 20% and reduce F_G by 10%`.
    Explicit PCT rules for all predictors mean C2; a selected subset means C3.
    The fixed Milan UCP/fraction rasters are
    loaded automatically from the project; manual paths are only an advanced option.
13. Show that “simulation completed” appears only after raster validation.

The Agent retrieves facts from the professor's tables. It does not claim legal
or physical feasibility, and it does not claim an actual temperature reduction
without a downstream model and suitable validation data.
