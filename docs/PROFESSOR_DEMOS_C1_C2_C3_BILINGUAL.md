# Urban Layer Agent — Professor Demo Guide
# Urban Layer Agent——导师演示指南

This guide contains two complete demonstrations for the current prototype.  
本指南包含当前原型系统的两套完整演示。

1. **Demo 1 — C1: LCZ-based value replacement**  
   **演示一——C1：基于 LCZ 的数值替换**
2. **Demo 2 — C2/C3: direct percentage adjustment**  
   **演示二——C2/C3：直接百分比调整**

The demonstrations follow the current system design.  
这些演示严格遵循当前系统设计。

- C1 target values come from the professor-provided LCZ reference data.  
  C1 的目标值来自教授提供的 LCZ 参考数据。
- C2/C3 percentages must be explicitly entered by the user.  
  C2/C3 的百分比必须由用户明确输入。
- The Agent never derives a percentage from an LCZ target.  
  Agent 不会根据 LCZ 目标推导百分比。
- All 12 predictors using PCT are classified as C2.  
  全部 12 个 predictors 使用 PCT 时，系统分类为 C2。
- Selected predictors using PCT are classified as C3.  
  只有选定 predictors 使用 PCT 时，系统分类为 C3。
- The user confirms every proposal before GIS execution.  
  在 GIS 执行前，用户必须确认每个方案。

---

# Demo 1 — C1: LCZ-Based Replacement
# 演示一——C1：基于 LCZ 的数值替换

## Purpose
## 演示目的

This demo shows the complete C1 workflow.  
本演示展示完整的 C1 工作流。

- Vector upload and real Polygon ID detection  
  上传 Vector 并识别真实 Polygon ID
- Natural-language goal understanding  
  理解自然语言目标
- Agent clarification questions  
  Agent 主动追问
- LCZ recommendation and reference-data retrieval  
  LCZ 推荐和参考数据检索
- Explanation and alternative comparison  
  推荐解释和备选方案比较
- Confirm, Revise, and Undo  
  确认、修改和撤销
- Multi-polygon planning  
  多 Polygon 规划
- C1 generation, execution, and validation  
  C1 文件生成、执行和验证

## Step 1 — Create a new task
## 第一步——新建任务

Create a task named:  
创建以下名称的任务：

```text
Demo C1 - Urban Cooling
```

Choose:  
选择：

```text
C1 — LCZ type replacement
```

Upload:  
上传：

```text
test.geojson
```

### Check
### 检查

- The system detects three polygons.  
  系统应识别出三个 Polygon。
- The IDs are Polygon 7, Polygon 8, and Polygon 9.  
  ID 应为 Polygon 7、Polygon 8 和 Polygon 9。
- The map preview appears.  
  右侧应显示地图预览。
- The interface explains that C1 uses LCZ reference values.  
  页面应说明 C1 使用 LCZ 参考值。

## Step 2 — Enter an overall goal
## 第二步——输入总体目标

Enter:  
输入：

```text
I want to reduce urban heat while keeping recreational space.
```

Meaning: I want to reduce urban heat and keep space for recreation.  
含义：我希望降低城市热，同时保留休闲空间。

### Expected result
### 预期结果

The Agent should recognize an urban cooling goal.  
Agent 应识别出城市降温目标。

It should recommend relevant types such as Dense trees, Low Plants, Scattered trees, or Water.  
它应推荐 Dense trees、Low Plants、Scattered trees 或 Water 等相关类型。

The goal helps rank possible LCZ types.  
总体目标帮助系统对可能的 LCZ 类型进行排序。

It does not directly change a raster.  
总体目标不会直接修改 Raster。

## Step 3 — Enter an incomplete planning request
## 第三步——输入不完整的规划要求

Enter:  
输入：

```text
Polygon 7 should become a pleasant park.
```

Meaning: Polygon 7 should become a comfortable park.  
含义：Polygon 7 应改造成一个舒适宜人的公园。

“Pleasant park” is not precise.  
“Pleasant park” 的描述并不具体。

The Agent should ask a clarification question.  
Agent 应提出澄清问题。

For example:  
例如：

```text
What is the main purpose: cooling, recreation, biodiversity, or stormwater management?
```

Meaning: What is the main function of this park?  
含义：这个公园的主要功能是什么？

Answer:  
回答：

```text
Cooling and recreation.
```

Meaning: The main functions are cooling and recreation.  
含义：主要功能是降温和休闲。

If the Agent asks about vegetation form, answer:  
如果 Agent 继续询问植被形式，回答：

```text
Scattered trees with some shade, and keep it open.
```

Meaning: Use scattered trees and shade, but keep the space open.  
含义：使用分散树木并提供一定遮阴，同时保持空间开阔。

### Expected result
### 预期结果

The Agent should prepare a proposal for Polygon 7.  
Agent 应为 Polygon 7 准备一个方案。

The proposal should contain:  
方案应包含：

- A recommended LCZ type  
  一个推荐的 LCZ 类型
- Alternative LCZ types  
  其他备选 LCZ 类型
- Reference evidence  
  参考数据证据
- A confidence value  
  置信度数值
- A limitation statement  
  局限性说明

The exact ranking can depend on the current reference table and scoring logic.  
准确排序可能取决于当前参考表和评分逻辑。

### Simple explanation for the professor
### 向教授进行的简单说明

> The agent asks questions because “pleasant park” is not precise.  
> Agent 会进行追问，因为“pleasant park”的描述不够具体。
>
> It does not execute the first possible interpretation.  
> 它不会直接执行第一个可能的理解结果。

## Step 4 — Test explanations
## 第四步——测试解释功能

Enter:  
输入：

```text
Why did you recommend Scattered trees?
```

Meaning: Why is Scattered trees the recommended option?  
含义：为什么推荐 Scattered trees？

Then enter:  
然后输入：

```text
Why not Dense trees?
```

Meaning: Why is Dense trees not the preferred option?  
含义：为什么 Dense trees 不是首选方案？

### Check
### 检查

The Agent should explain why the recommendation matches the user's requirements.  
Agent 应解释该推荐为什么符合用户要求。

It should show how the reference-table correspondence supports the result.  
它应说明参考表中的对应关系如何支持该结果。

It should compare the preferred and alternative types.  
它应比较推荐类型和备选类型。

It should state that the result does not prove real temperature reduction.  
它应说明该结果不能证明真实的温度下降。

It should state that construction and legal feasibility are not evaluated.  
它应说明当前系统没有评价建设和法律可行性。

Asking for information must not modify the plan.  
询问信息不能修改当前方案。

## Step 5 — Confirm Polygon 7
## 第五步——确认 Polygon 7

Enter:  
输入：

```text
Confirm
```

Meaning: Accept the current proposal.  
含义：接受当前方案。

### Expected status
### 预期状态

```text
Polygon 7: Scattered trees
Polygon 8: not defined yet
Polygon 9: not defined yet
```

Polygon 7 is confirmed, while Polygon 8 and 9 are still undefined.  
Polygon 7 已确认，而 Polygon 8 和 9 尚未定义。

## Step 6 — Plan the remaining polygons
## 第六步——规划其余 Polygon

Enter:  
输入：

```text
Make Polygon 8 a grass park and leave Polygon 9 unchanged.
```

Meaning: Change Polygon 8 into a grass park and do not modify Polygon 9.  
含义：将 Polygon 8 改造成草地公园，并保持 Polygon 9 不变。

The system should interpret the instruction as:  
系统应将指令理解为：

```text
Polygon 8 → Low Plants
Polygon 9 → unchanged
```

Confirm with:  
使用以下指令确认：

```text
Confirm all
```

If the interface asks for an ordinary confirmation, enter:  
如果页面要求普通确认，则输入：

```text
Confirm
```

### Expected final plan
### 预期最终方案

```text
Polygon 7: Scattered trees
Polygon 8: Low Plants
Polygon 9: unchanged
```

## Step 7 — Test state recall
## 第七步——测试状态记忆

Enter:  
输入：

```text
What have I decided so far?
```

Meaning: Show all decisions made in the current task.  
含义：显示当前任务中已经作出的全部决定。

The Agent should report the goal and current Polygon decisions.  
Agent 应报告总体目标和当前 Polygon 决策。

## Step 8 — Test Undo
## 第八步——测试撤销功能

Enter:  
输入：

```text
Undo
```

Meaning: Remove the last confirmed change.  
含义：撤销上一次确认的修改。

Check that the last confirmed change is removed.  
检查最后一次确认的修改是否被撤销。

Then restore the final plan:  
然后恢复最终方案：

```text
Set Polygon 8 to Low Plants and leave Polygon 9 unchanged.
```

Confirm again.  
再次进行确认。

### Simple explanation for the professor
### 向教授进行的简单说明

> The task state stores confirmed decisions.  
> 任务状态会保存已确认的决策。
>
> Undo restores the previous confirmed state.  
> Undo 会恢复到上一次确认后的状态。

## Step 9 — Generate C1 inputs
## 第九步——生成 C1 输入

Enter:  
输入：

```text
Generate
```

Meaning: Generate the structured files required by Layer Alterator.  
含义：生成 Layer Alterator 所需的结构化文件。

### Expected outputs
### 预期输出

- Updated vector input  
  更新后的 Vector 输入文件
- C1 operation rules  
  C1 操作规则文件
- Simulation configuration  
  模拟配置文件

Check that Polygon 7 is linked to Scattered trees.  
检查 Polygon 7 是否对应 Scattered trees。

Check that Polygon 8 is linked to Low Plants.  
检查 Polygon 8 是否对应 Low Plants。

Check that Polygon 9 remains unchanged.  
检查 Polygon 9 是否保持不变。

Check that predictor values come from the reference data.  
检查 predictor 数值是否来自参考数据。

## Step 10 — Run Layer Alterator
## 第十步——运行 Layer Alterator

Open **Run Layer Alterator** and click:  
展开 **Run Layer Alterator** 并点击：

```text
Run Layer Alterator
```

Meaning: Execute the deterministic raster modification.  
含义：执行确定性的 Raster 修改程序。

### Expected result
### 预期结果

- Twelve GeoTIFF raster files are generated.  
  系统生成 12 个 GeoTIFF Raster 文件。
- The interface reports successful validation.  
  页面显示验证成功。
- C1 assigns LCZ reference values inside selected polygons.  
  C1 在选定 Polygon 内写入 LCZ 参考值。
- Raster cells outside the polygons remain unchanged.  
  Polygon 外部的 Raster 像元保持不变。
- NoData cells remain NoData.  
  NoData 像元继续保持为 NoData。
- Land-cover fraction checks pass.  
  土地覆盖比例检查应通过。

### Key explanation for the professor
### 向教授强调的内容

> Reference data provides the predictor values.  
> Predictor 数值由参考数据提供。
>
> The LLM does not create these values.  
> LLM 不会创造这些数值。
>
> Python and Layer Alterator modify the rasters.  
> Python 和 Layer Alterator 负责修改 Raster。

---

# Demo 2 — C2/C3: Direct Percentage Adjustment
# 演示二——C2/C3：直接百分比调整

This demonstration has two parts.  
该演示分成两个部分。

- **Part A — C3:** selected predictors use PCT.  
  **A 部分——C3：**只有选定的 predictors 使用 PCT。
- **Part B — C2:** all twelve predictors use PCT.  
  **B 部分——C2：**全部 12 个 predictors 使用 PCT。

Use separate tasks for the two parts.  
请为两个部分分别建立独立任务。

This makes the automatic classification clear and avoids mixed state.  
这样可以清楚展示自动分类，并避免任务状态混合。

---

## Part A — C3 Demonstration
## A 部分——C3 演示

### Step 1 — Create a C3 task
### 第一步——建立 C3 任务

Create a task named:  
创建以下名称的任务：

```text
Demo C3 - Selected Predictors
```

Choose:  
选择：

```text
C2/C3 — direct percentage adjustment
```

Upload:  
上传：

```text
test.geojson
```

### Check
### 检查

The overall goal should be optional.  
总体目标应是可选的。

The user must provide the Polygon, predictor, direction, and percentage.  
用户必须提供 Polygon、predictor、方向和百分比。

The system should automatically classify C2 or C3.  
系统应自动判断 C2 或 C3。

### Step 2 — Define Polygon 8
### 第二步——定义 Polygon 8

Enter:  
输入：

```text
For Polygon 8, increase F_W by 33%, decrease IMD by 11%, and increase TCH by 22%.
```

Meaning: Increase water fraction by 33%, reduce imperviousness by 11%, and increase tree height by 22% in Polygon 8.  
含义：在 Polygon 8 中，将水体比例增加 33%，不透水程度降低 11%，树冠高度增加 22%。

### Expected proposal
### 预期方案

```text
C3 direct percentage proposal for Polygon 8

TCH: +22%
IMD: -11%
F_W: +33%
```

The remaining predictors should be shown as NONE.  
其余 predictors 应显示为 NONE。

Enter:  
输入：

```text
Confirm
```

### Expected status
### 预期状态

```text
Polygon 7: not defined
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: not defined
```

### Step 3 — Define Polygon 7
### 第三步——定义 Polygon 7

Enter:  
输入：

```text
For Polygon 7, increase F_W by 20%, decrease IMD by 10%, and increase TCH by 15%.
```

Meaning: Apply another direct percentage plan to Polygon 7.  
含义：为 Polygon 7 设置另一个直接百分比方案。

Enter:  
输入：

```text
Confirm
```

### Expected status
### 预期状态

```text
Polygon 7: TCH +15%, IMD -10%, F_W +20%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: not defined
```

This is an important multi-polygon test.  
这是一个重要的多 Polygon 测试。

Confirming Polygon 7 must not remove the confirmed Polygon 8 plan.  
确认 Polygon 7 时不能删除已经确认的 Polygon 8 方案。

### Step 4 — Mark the remaining polygon unchanged
### 第四步——将剩余 Polygon 标记为不变

Enter:  
输入：

```text
Leave the remaining polygons unchanged.
```

Meaning: Do not modify any polygon without a confirmed percentage plan.  
含义：不修改尚未设置百分比方案的其他 Polygon。

### Expected final plan
### 预期最终方案

```text
Polygon 7: TCH +15%, IMD -10%, F_W +20%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: unchanged
```

### Step 5 — Replace one Polygon decision
### 第五步——修改一个 Polygon 的方案

Enter:  
输入：

```text
For Polygon 7, increase F_W by 25%, decrease IMD by 8%, and increase TCH by 12%.
```

Meaning: Replace only the existing plan for Polygon 7.  
含义：只替换 Polygon 7 现有的方案。

Confirm it.  
确认该方案。

### Expected result
### 预期结果

Only Polygon 7 should change.  
只有 Polygon 7 应发生变化。

```text
Polygon 7: TCH +12%, IMD -8%, F_W +25%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: unchanged
```

### Step 6 — Test Undo
### 第六步——测试 Undo

Enter:  
输入：

```text
Undo
```

Meaning: Restore the plan before the last confirmation.  
含义：恢复到上一次确认之前的方案。

### Expected result
### 预期结果

Polygon 7 should return to its earlier confirmed plan.  
Polygon 7 应恢复为之前确认的方案。

```text
Polygon 7: TCH +15%, IMD -10%, F_W +20%
Polygon 8: TCH +22%, IMD -11%, F_W +33%
Polygon 9: unchanged
```

### Step 7 — Generate and run C3
### 第七步——生成并运行 C3

Enter:  
输入：

```text
Generate
```

Meaning: Generate the direct-percentage Layer Alterator inputs.  
含义：生成直接百分比 Layer Alterator 输入文件。

The system should classify the workflow as C3.  
系统应将该工作流分类为 C3。

Only TCH, IMD, and F_W use PCT rules.  
只有 TCH、IMD 和 F_W 使用 PCT 规则。

Other predictors use NONE.  
其他 predictors 使用 NONE。

Click **Run Layer Alterator**.  
点击 **Run Layer Alterator**。

### Check
### 检查

- Twelve output rasters are generated.  
  系统生成 12 个输出 Raster。
- TCH, IMD, and F_W receive direct percentage adjustments.  
  TCH、IMD 和 F_W 执行直接百分比调整。
- Other UCP predictors remain unchanged.  
  其他 UCP predictors 保持不变。
- The seven land-cover fractions are normalized jointly per pixel.  
  七个土地覆盖比例图层会按像元联合归一化。
- Cells outside the intervention polygons remain unchanged.  
  干预 Polygon 外的像元保持不变。
- Structural validation passes.  
  结构验证应通过。

### Key explanation for the professor
### 向教授强调的内容

> F_W uses the percentage entered by the user.  
> F_W 使用用户输入的百分比。
>
> The agent does not calculate this percentage from an LCZ type.  
> Agent 不会根据 LCZ 类型计算该百分比。

---

## Part B — C2 Demonstration
## B 部分——C2 演示

### Step 1 — Create a C2 task
### 第一步——建立 C2 任务

Create a task named:  
创建以下名称的任务：

```text
Demo C2 - All Predictors
```

Choose **C2/C3 — direct percentage adjustment**.  
选择 **C2/C3 — direct percentage adjustment**。

Upload `test.geojson`.  
上传 `test.geojson`。

### Step 2 — Apply PCT to all predictors
### 第二步——对全部 predictors 使用 PCT

For a short technical demonstration, enter:  
为了进行简短的技术演示，输入：

```text
Increase all predictors in Polygon 7 by 5%.
```

Meaning: Apply a positive 5% PCT rule to all twelve predictors in Polygon 7.  
含义：对 Polygon 7 中的全部 12 个 predictors 应用增加 5% 的 PCT 规则。

### Expected result
### 预期结果

The Agent should detect all twelve predictors.  
Agent 应识别全部 12 个 predictors。

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

The proposal should be classified as C2.  
该方案应被分类为 C2。

```text
C2 direct percentage proposal for Polygon 7
```

Enter:  
输入：

```text
Confirm
```

Then enter:  
然后输入：

```text
Leave the remaining polygons unchanged.
```

Meaning: Keep Polygon 8 and Polygon 9 unchanged.  
含义：保持 Polygon 8 和 Polygon 9 不变。

### Step 3 — Generate and run C2
### 第三步——生成并运行 C2

Enter:  
输入：

```text
Generate
```

The system should classify the workflow as C2.  
系统应将该工作流分类为 C2。

All twelve predictors use PCT rules.  
全部 12 个 predictors 使用 PCT 规则。

Click **Run Layer Alterator**.  
点击 **Run Layer Alterator**。

### Check
### 检查

- Twelve output rasters are generated.  
  系统生成 12 个输出 Raster。
- All predictors in Polygon 7 use PCT rules.  
  Polygon 7 中的全部 predictors 使用 PCT 规则。
- Polygon 8 and Polygon 9 remain unchanged.  
  Polygon 8 和 Polygon 9 保持不变。
- Fraction layers are normalized after adjustment.  
  Fraction layers 在调整后进行归一化。
- Cells outside Polygon 7 remain unchanged.  
  Polygon 7 外部的像元保持不变。
- Output validation passes.  
  输出验证应通过。

---

# Main Difference Between the Workflows
# 工作流之间的主要区别

| Item 项目 | C1 | C2/C3 |
|---|---|---|
| User input 用户输入 | Urban type or planning description 城市类型或规划描述 | Explicit percentages 明确百分比 |
| Numeric source 数值来源 | Professor-provided LCZ reference data 教授提供的 LCZ 参考数据 | User input 用户输入 |
| Agent role Agent 作用 | Understand requirements and match LCZ 理解需求并匹配 LCZ | Extract percentage instructions 提取百分比指令 |
| Classification 分类 | C1 | Automatic C2 or C3 自动判断 C2 或 C3 |
| Raster operation Raster 操作 | Replace values inside polygons 替换 Polygon 内数值 | Apply percentage adjustments 执行百分比调整 |
| Confirmation 确认 | Required 必须 | Required 必须 |

---

# Simple English Summary for the Meeting
# 面谈时使用的简单英文总结

## After the C1 demo
## 完成 C1 演示后

> In C1, the user describes an urban transformation.  
> 在 C1 中，用户描述一种城市空间改造。
>
> The agent matches the request with an LCZ type.  
> Agent 将用户要求与 LCZ 类型进行匹配。
>
> Predictor values come from the professor-provided reference data.  
> Predictor 数值来自教授提供的参考数据。
>
> The LLM does not create the values.  
> LLM 不会创造这些数值。
>
> Layer Alterator replaces raster values after user confirmation.  
> 用户确认后，Layer Alterator 才会替换 Raster 数值。

## After the C2/C3 demo
## 完成 C2/C3 演示后

> C2 and C3 use direct percentage instructions.  
> C2 和 C3 使用直接百分比指令。
>
> Every percentage comes from the user.  
> 每一个百分比都来自用户。
>
> The agent only extracts and validates the instruction.  
> Agent 只负责提取和验证指令。
>
> All twelve PCT predictors mean C2.  
> 全部 12 个 predictors 使用 PCT 时表示 C2。
>
> Selected PCT predictors mean C3.  
> 只有选定 predictors 使用 PCT 时表示 C3。
>
> The system makes this classification automatically.  
> 系统会自动完成该分类。

## Final summary
## 最后总结

> Both workflows are human-in-the-loop.  
> 两种工作流都采用人在回路机制。
>
> The agent prepares the inputs, but the user confirms the plan.  
> Agent 准备输入，但用户负责确认方案。
>
> Python controls the real GIS execution.  
> Python 控制实际的 GIS 执行过程。
>
> The current system does not yet predict real LST change.  
> 当前系统尚不能预测真实的 LST 变化。
