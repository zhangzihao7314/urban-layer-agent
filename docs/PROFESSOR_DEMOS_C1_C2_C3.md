# Urban Layer Agent — Professor Demo Guide

This guide contains two complete demonstrations for the current prototype:

1. **Demo 1 — C1: LCZ-based value replacement**
2. **Demo 2 — C2/C3: direct percentage adjustment**

The demonstrations follow the current system design:

- C1 target values come from the professor-provided LCZ reference data.
- C2/C3 percentages must be explicitly entered by the user.
- The Agent never derives a percentage from an LCZ target.
- All 12 predictors using PCT are classified as C2.
- Selected predictors using PCT, with the others set to NONE, are classified as C3.
- The user confirms every proposal before GIS execution.

---

# Demo 1 — C1: LCZ-Based Replacement

## Purpose

This demo shows:

- Vector upload and real Polygon ID detection
- Natural-language goal understanding
- Agent clarification questions
- LCZ scenario recommendation
- Reference-data retrieval
- Explanation and alternative comparison
- Confirm, Revise, and Undo
- Multi-polygon planning
- C1 input generation
- Twelve raster outputs and validation

## Step 1 — Create a new task

Create a task named:

```text
Demo C1 - Urban Cooling
```

Choose:

```text
C1 — LCZ type replacement
```

Upload:

```text
test.geojson
```

### Check

- The system detects three polygons.
- The IDs are Polygon 7, Polygon 8, and Polygon 9.
- The map preview appears.
- The interface explains that C1 uses LCZ reference values.

## Step 2 — Enter an overall goal

Enter:

```text
I want to reduce urban heat while keeping recreational space.
```

### Expected result

The Agent should recognize an urban cooling goal and recommend relevant types,
such as Dense trees, Low Plants, Scattered trees, or Water.

The goal helps rank possible LCZ types. It does not directly change a raster.

## Step 3 — Enter an incomplete planning request

Enter:

```text
Polygon 7 should become a pleasant park.
```

Because “pleasant park” is not precise, the Agent should ask a clarification
question. For example:

```text
What is the main purpose: cooling, recreation, biodiversity, or stormwater management?
```

Answer:

```text
Cooling and recreation.
```

If the Agent asks about vegetation form, answer:

```text
Scattered trees with some shade, and keep it open.
```

### Expected result

The Agent should prepare a Polygon 7 proposal with:

- A recommended LCZ type
- Alternative LCZ types
- Reference evidence
- A confidence value
- A limitation statement

The exact ranking can depend on the current reference table and scoring logic.

### Simple explanation for the professor

> The agent asks questions because “pleasant park” is not precise.  
> It does not execute the first possible interpretation.

## Step 4 — Test explanations

Enter:

```text
Why did you recommend Scattered trees?
```

Then enter:

```text
Why not Dense trees?
```

### Check

The Agent should explain:

- Why the recommendation matches the user's requirements
- How the reference-table correspondence supports it
- The difference between the preferred and alternative type
- That the result does not prove real temperature reduction
- That construction and legal feasibility are not evaluated

Asking for information must not modify the current plan.

## Step 5 — Confirm Polygon 7

Enter:

```text
Confirm
```

### Expected status

```text
Polygon 7: Scattered trees
Polygon 8: not defined yet
Polygon 9: not defined yet
```

## Step 6 — Plan the remaining polygons

Enter:

```text
Make Polygon 8 a grass park and leave Polygon 9 unchanged.
```

The system should interpret this as:

```text
Polygon 8 → Low Plants
Polygon 9 → unchanged
```

Confirm the proposal with:

```text
Confirm all
```

If the interface requests an ordinary confirmation, use:

```text
Confirm
```

### Expected final plan

```text
Polygon 7: Scattered trees
Polygon 8: Low Plants
Polygon 9: unchanged
```

## Step 7 — Test state recall

Enter:

```text
What have I decided so far?
```

The Agent should report the overall goal and current Polygon decisions.

## Step 8 — Test Undo

Enter:

```text
Undo
```

Check that the last confirmed change is removed. Then restore the final plan:

```text
Set Polygon 8 to Low Plants and leave Polygon 9 unchanged.
```

Confirm again.

### Simple explanation for the professor

> The task state stores confirmed decisions.  
> Undo restores the previous confirmed state.

## Step 9 — Generate C1 inputs

Enter:

```text
Generate
```

### Expected outputs

- Updated vector input
- C1 operation rules
- Simulation configuration

Check that:

- Polygon 7 is linked to Scattered trees.
- Polygon 8 is linked to Low Plants.
- Polygon 9 remains unchanged.
- Predictor values come from the reference data.

## Step 10 — Run Layer Alterator

Open **Run Layer Alterator** and click:

```text
Run Layer Alterator
```

### Expected result

- Twelve GeoTIFF raster files are generated.
- The interface reports successful validation.
- C1 assigns the LCZ reference values inside each selected polygon.
- Raster cells outside the polygons remain unchanged.
- NoData cells remain NoData.
- Land-cover fraction checks pass.

### Key explanation for the professor

> Reference data provides the predictor values.  
> The LLM does not create these values.  
> Python and Layer Alterator modify the rasters.

---

# Demo 2 — C2/C3: Direct Percentage Adjustment

This demonstration has two parts:

- **Part A — C3:** selected predictors use PCT.
- **Part B — C2:** all twelve predictors use PCT.

Use separate tasks for the two parts. This makes the automatic classification
clear and avoids mixing their state.

---

## Part A — C3 Demonstration

### Step 1 — Create a C3 task

Create a task named:

```text
Demo C3 - Selected Predictors
```

Choose:

```text
C2/C3 — direct percentage adjustment
```

Upload:

```text
test.geojson
```

### Check

The interface should explain that:

- The overall goal is optional.
- The user must provide the Polygon, predictor, direction, and percentage.
- The system automatically classifies C2 or C3.

### Step 2 — Define Polygon 8

Enter:

```text
For Polygon 8, increase F_W by 33%, decrease IMD by 11%, and increase TCH by 22%.
```

### Expected proposal

```text
C3 direct percentage proposal for Polygon 8

TCH: +22%
IMD: -11%
F_W: +33%
```

The remaining predictors should be shown as NONE.

Enter:

```text
Confirm
```

### Expected status

```text
Polygon 7: not defined
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: not defined
```

### Step 3 — Define Polygon 7

Enter:

```text
For Polygon 7, increase F_W by 20%, decrease IMD by 10%, and increase TCH by 15%.
```

Enter:

```text
Confirm
```

### Expected status

```text
Polygon 7: TCH +15%, IMD -10%, F_W +20%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: not defined
```

This is an important multi-polygon test. Confirming Polygon 7 must not remove
the confirmed Polygon 8 plan.

### Step 4 — Mark the remaining polygon unchanged

Enter:

```text
Leave the remaining polygons unchanged.
```

### Expected final plan

```text
Polygon 7: TCH +15%, IMD -10%, F_W +20%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: unchanged
```

### Step 5 — Replace one Polygon decision

Enter:

```text
For Polygon 7, increase F_W by 25%, decrease IMD by 8%, and increase TCH by 12%.
```

Confirm it.

### Expected result

Only Polygon 7 should change:

```text
Polygon 7: TCH +12%, IMD -8%, F_W +25%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: unchanged
```

### Step 6 — Test Undo

Enter:

```text
Undo
```

### Expected result

Polygon 7 should return to its earlier confirmed plan:

```text
Polygon 7: TCH +15%, IMD -10%, F_W +20%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: unchanged
```

### Step 7 — Generate and run C3

Enter:

```text
Generate
```

The system should classify the workflow as C3 because only TCH, IMD, and F_W
use PCT rules. Other predictors use NONE.

Click **Run Layer Alterator**.

### Check

- Twelve output rasters are generated.
- TCH, IMD, and F_W receive direct percentage adjustments.
- Other UCP predictors remain unchanged.
- The seven land-cover fractions are normalized jointly per pixel.
- Cells outside the intervention polygons remain unchanged.
- The structural validation passes.

### Key explanation for the professor

> F_W uses the percentage entered by the user.  
> The agent does not calculate this percentage from an LCZ type.

---

## Part B — C2 Demonstration

### Step 1 — Create a C2 task

Create a task named:

```text
Demo C2 - All Predictors
```

Choose **C2/C3 — direct percentage adjustment** and upload `test.geojson`.

### Step 2 — Apply PCT to all predictors

For a short technical demonstration, enter:

```text
Increase all predictors in Polygon 7 by 5%.
```

### Expected result

The Agent should detect all twelve predictors:

- TCH
- IMD
- BH
- BSF
- SVF
- F_AC
- F_S
- F_M
- F_BS
- F_G
- F_TV
- F_W

The proposal should be classified as:

```text
C2 direct percentage proposal for Polygon 7
```

Enter:

```text
Confirm
```

Then enter:

```text
Leave the remaining polygons unchanged.
```

### Step 3 — Generate and run C2

Enter:

```text
Generate
```

The system should classify the workflow as C2 because all twelve predictors
use PCT rules.

Click **Run Layer Alterator**.

### Check

- Twelve output rasters are generated.
- All predictors in Polygon 7 use PCT rules.
- Polygon 8 and Polygon 9 remain unchanged.
- Fraction layers are normalized after adjustment.
- Cells outside Polygon 7 remain unchanged.
- Output validation passes.

---

# Main Difference Between the Workflows

| Item | C1 | C2/C3 |
|---|---|---|
| User input | Urban type or planning description | Explicit percentages |
| Numeric source | Professor-provided LCZ reference data | User input |
| Agent role | Understand requirements and match LCZ | Extract percentage instructions |
| Classification | C1 | Automatic C2 or C3 |
| Raster operation | Replace values inside polygons | Apply percentage adjustments |
| Confirmation | Required | Required |

---

# Simple English Summary for the Meeting

## After the C1 demo

> In C1, the user describes an urban transformation.  
> The agent matches the request with an LCZ type.  
> Predictor values come from the professor-provided reference data.  
> The LLM does not create the values.  
> Layer Alterator replaces raster values after user confirmation.

## After the C2/C3 demo

> C2 and C3 use direct percentage instructions.  
> Every percentage comes from the user.  
> The agent only extracts and validates the instruction.  
> All twelve PCT predictors mean C2.  
> Selected PCT predictors mean C3.  
> The system makes this classification automatically.

## Final summary

> Both workflows are human-in-the-loop.  
> The agent prepares the inputs, but the user confirms the plan.  
> Python controls the real GIS execution.  
> The current system does not yet predict real LST change.
