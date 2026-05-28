
## 0. 总体判断

论文核心 idea 是：

1. 固定同一个 root scene；
2. 对不同 ego interventions 做 reactive rollout；
3. 对同一 agent 学一个 intervention → response primitive 的 surface；
4. 用 ceding / non-ceding 分支下的安全性差异 + ceding burden + priority 判断 coercive false-safe；
5. 部署时只能用 scene-only token，不能用 support probes 或未来标签。

代码现在存在三层断裂：

第一，**数据构建层断裂**：`examples/mfrp_waymax_adapter.py` 与当前 `mfrp/data/schema.py` 根本不匹配，会直接 import / constructor / attribute error。

第二，**算法实现被大幅简化**：scene encoder、support encoder、geometry loss、trajectory distribution、coercion ranking、boundary sensitivity 都远低于论文定义。

第三，**实验闭环缺失**：`scripts/eval_mfrp.py` 只做简单离线 branch/risk 指标，没有论文 Experiments 里要求的 same-root held-out response、surface geometry、false-safe selection、closed-loop planning、ablation stress tests。

---

# 1. 论文理论上有问题，并导致代码部署也有问题的地方

## 1.1 “scene-only token 能继承 support-adapted surface”缺少可识别性保证

**论文位置**：Method 里 `Ω^S` 是 support-adapted teacher，`Ω^0` 是部署 scene-only token；通过 distillation 让 `Ω^0` 继承 `Ω^S` 的 response surface。

**理论问题**：训练时 `Ω^S` 看到了 support probes，本质上含有 counterfactual response evidence；部署时 `Ω^0` 只看 scene。除非 scene 中有足够信息能预测 agent response mechanism，否则 distillation 只能让模型平均化，不能保证同一个 scene-only token 可恢复具体 surface。论文需要明确这是 amortized approximation，而不是 causal identification。

**代码定位**：

* `mfrp/models/mfrp_model.py:61-70`：`_tokens()` 中 support token 直接由 `support_probe_features` 生成，scene token 只由 `scene_features` 生成。
* `mfrp/training/losses.py:47-54`：`distillation_loss()` 只匹配 branch + burden + margin，**没有匹配 trajectory distribution，也没有匹配 witness / safety dependence / uncertainty**。
* `mfrp/data/tensors.py:66-76`：`scene_features` 只是很粗的 history summary，远远不包含论文里 map、traffic control、route、agent role token 的 scene evidence。

**建议修改**：

* 在论文中把 claim 改成：scene-only token is an amortized prior over mechanisms, not identifiable without response probes。
* 在代码中加强 `scene_features`，至少包含：map vector tokens、traffic-control state、route alignment、ego-agent relative history、target agent ID role embedding。
* `distillation_loss()` 应匹配完整 response primitive：branch KL、trajectory distribution distance、burden/margin NLL 或 KL、`S_C/S_notC/B_C/D_C/κ` 的 consistency。
* 报实验时必须区分 `support_adapted upper bound` 和 `scene_only deployable result`。

---

## 1.2 Coercion witness 理论上需要 ceding 和 non-ceding 双侧证据，但代码经常只有单侧或伪双侧

**论文位置**：Counterfactual Coercion Witness，`S_C(u)`、`S_notC(u)`、`B_C(u)`；Appendix 里也说 forced-dependence supervision requires evidence under both ceding and non-ceding responses。

**理论问题**：如果某 candidate 的 rollout variants 里只有 ceding，或只有 non-ceding，那么 `S_C - S_notC` 不可估计。论文说要降低 confidence，但这个机制在代码中不完整。

**代码定位**：

* `mfrp/models/mfrp_model.py:91-98`：直接用预测 branch probability 计算 `S_C`、`S_notC`、`B_C`。当 `p_c` 很小或 non-ceding probability 很小时，只用 `clamp_min(1e-6)` 避免数值炸掉，但没有把低证据/低概率条件传给 witness confidence。
* `mfrp/training/losses.py:76-86`：`cw_loss()` 只有 BCE soft label，没有 pairwise ranking，也没有显式检查 ceding/non-ceding evidence diversity。
* `mfrp/data/tensors.py:106-107`：`cw_soft_label` 和 `cw_confidence` 是对 observations 取 `max`，这会把某个 variant 的高 label 扩散到 candidate 层，理论上不等价于 appendix 里按 variants 聚合计算 `S_C/S_notC/B_C`。

**建议修改**：

* 增加数据层函数：按 `(scene, agent, candidate)` 聚合所有 variants，显式计算：

  * ceding_count；
  * nonceding_count；
  * branch diversity；
  * `S_C`、`S_notC`、`B_C`；
  * confidence。
* `cw_loss()` 加 pairwise ranking loss，对应论文 Eq. ranking 设计。
* witness 输入增加 evidence mask / confidence，不要只靠 `p_c.clamp_min()`。

---

## 1.3 Scene-level risk 使用 noisy-OR，隐含 agent 独立性，论文只轻描淡写

**论文位置**：Appendix selector calibration，scene risk `1 - ∏(1 - p_i^viol)`。

**理论问题**：多 agent response 在 merge、unprotected turn、lane change 中强相关。noisy-OR 会错误估计风险，尤其是多个 agent 同时对 ego 行为响应时。

**代码定位**：

* `mfrp/planning/estimators.py:14-15`：直接 noisy-OR 聚合。
* `mfrp/planning/selector.py:17-22`：selector 直接使用这个 `rho_mech` 过滤 candidate。

**建议修改**：

* 论文中明确 noisy-OR 是 approximation。
* 代码中至少支持两种模式：

  * conservative upper bound：`max_i p_viol_i` 或 clipped sum；
  * learned correlated aggregation：用 scene-agent graph encoder 输出 joint violation。
* 实验中报告 noisy-OR vs max/sum/joint aggregator ablation。

---

## 1.4 论文中的 boundary sensitivity / surface sensitivity 在 selector 中没有实现

**论文说**：candidate 会因为 uncertainty 或 boundary-sensitive 被拒绝。

**代码问题**：

* `mfrp/planning/selector.py:14` 有 `gamma_bar` 参数。
* 但 `gamma_bar` 在 `select_mechanism_feasible()` 中完全没有被使用。
* `mfrp/training/losses.py:57-73` 的 `geometry_loss()` 也没有输出可供部署使用的 local sensitivity `γ_i(u)`。

**建议修改**：

* 在 model output 中增加 local sensitivity estimate，例如基于：

  * neighbor finite difference；
  * gradient norm `||∂p(O|u)/∂a||`；
  * boundary probability。
* `selector.py` 中加入：

  ```python
  feasible = feasible & (gamma <= gamma_bar)
  ```
* `scripts/eval_mfrp.py` 里增加 boundary-sensitive rejection 指标。

---

## 1.5 Calibration 理论上不能保证 planner selection 后的风险控制

**论文位置**：Appendix calibration 用 residual quantile `V - rho_hat`。

**理论问题**：这是 candidate-level risk calibration，不自动保证“选出来的 candidate”的 selective risk。选择器会偏向低估风险的 candidate，产生 selection bias。

**代码定位**：

* `mfrp/planning/calibration.py:27-35`：只拟合 residual quantile。
* `scripts/calibrate_mfrp.py:17-22`：truth 由 unsafe 或 cw label 粗略合成。
* `mfrp/planning/selector.py:18-22`：直接 `rho + q_beta <= alpha`。

**建议修改**：

* 论文里把当前 calibration 称为 conservative score correction，不要暗示 formal guarantee。
* 如需理论保证，新增 selector-level calibration：在 calibration set 上完整跑 selector，调 `alpha/nu/gamma` 使 selected violation rate 受控。
* `scripts/calibrate_mfrp.py` 应输出 candidate-level 和 selected-action-level 两套 calibration。

---

# 2. 代码中不符合现实或会直接失败的 bug

## 2.1 Waymax adapter 当前基本不能运行：schema / import / attribute 全部不匹配

**文件**：`examples/mfrp_waymax_adapter.py`

严重问题：

* `examples/mfrp_waymax_adapter.py:26` 从 `mfrp.data.scene_schema` import `AgentTrackTensor, RootScene, RouteContext`，但 `mfrp/data/scene_schema.py:1` 只 re-export 了 `RootScene, EgoCandidate, ResponseObservation, SameRootGroup`，没有 `AgentTrackTensor` 和 `RouteContext`。
* `examples/mfrp_waymax_adapter.py:28-32` import `BoundaryPair, CandidateValidity, root_scene_hash`，但 `mfrp/data/schema.py` 没有这些定义。
* `examples/mfrp_waymax_adapter.py:34-36` import `build_interaction_region, constant_velocity_extrapolate, make_response_observation, coercion_witness_label, compute_priority_score`，当前对应文件中不存在这些函数。
* `examples/mfrp_waymax_adapter.py:167-182` 构造 `RootScene` 使用 `split/source/womd_version/current_time_index/dt/...` 等字段，但 `mfrp/data/schema.py:22-33` 的 `RootScene` 只有 `scene_id, t0, history, history_mask, ego_index, map_features, traffic_controls, route_features, metadata`。
* `examples/mfrp_waymax_adapter.py:207` 使用 `root.root_state()`，当前 `RootScene` 无该方法。
* `examples/mfrp_waymax_adapter.py:215` 使用 `base.future_states_ego_frame`，当前 `EgoCandidate` 字段叫 `trajectory`。
* `examples/mfrp_waymax_adapter.py:225` 构造 `EgoCandidate("id", "family", states, ...)`，但当前 `EgoCandidate` 构造参数是 `(candidate_id, trajectory, features, nominal_cost, valid, metadata)`。

**后果**：真实数据集无法通过这个 adapter materialize。论文最核心的 same-root reactive rollout 数据根本不能生成。

**建议修改**：

* 先统一 schema，只保留一版。
* 给 `RootScene` 增加或恢复：

  * `dt`
  * `current_time_index`
  * `agent_tracks`
  * `route_context`
  * `traffic_controls`
  * `root_state()`
* 或者把 adapter 全部改成当前 minimal schema。
* 补齐 `label_extraction.py` 中缺失函数，或修改 adapter 使用已有 `classify_response_branch / compute_burden / safety_margin`。

---

## 2.2 测试与当前 API 不一致，说明代码处在半重构状态

我用 `PYTHONPATH=.` 跑测试，收集阶段就失败，关键错误包括：

* `tests/test_mfrp_planning.py` 期望 `apply_calibration`，但 `mfrp/planning/calibration.py` 没有。
* `tests/test_mfrp_support_model_loss.py` 期望 `MonotoneCoercionWitnessHead`，但实际是 `MonotoneCoercionWitness`。
* `tests/test_mfrp_schema_labels.py` 期望 `baseline_relative_burden, signed_oriented_box_separation, coercion_witness_label`，但 `label_extraction.py` 没有这些函数。

**文件定位**：

* `mfrp/planning/calibration.py:9-35`
* `mfrp/models/coercion_witness.py:8-32`
* `mfrp/data/label_extraction.py:1-81`
* `tests/test_mfrp_planning.py:3-4`
* `tests/test_mfrp_support_model_loss.py:6-7`
* `tests/test_mfrp_schema_labels.py:4-5`

**建议修改**：

* 要么更新 tests 适配当前 API；
* 要么恢复旧 API wrapper，例如：

  * `apply_calibration(rho, cal) -> cal.apply(rho)`
  * `MonotoneCoercionWitnessHead = MonotoneCoercionWitness`
  * 实现 `baseline_relative_burden()`、`signed_oriented_box_separation()`、`coercion_witness_label()`。

---

## 2.3 `scene_only_inference()` 可能返回 key 与旧测试/调用不兼容

**文件**：`mfrp/planning/deploy.py:14-16`

当前：

```python
return model(clean, mode="scene_only")
```

`MFRPModel` 返回 key 是：

* `scene_branch_logits`
* `scene_branch_prob`
* `scene_kappa`
* 等。

但测试里期待：

* `branch_logits`
* `kappa`

**建议修改**：

* 要么统一所有 downstream 使用 `scene_*` key；
* 要么 `scene_only_inference()` 做一次 key normalization，把 `scene_branch_logits` 映射成 `branch_logits`。

---

## 2.4 `collate_same_root_groups()` 的 agent feature 映射可能错位

**文件**：`mfrp/data/tensors.py:68-76`

```python
agent_num = a_idx + 1 if a_idx + 1 < hist.shape[0] else min(a_idx, hist.shape[0] - 1)
```

这是假设 relevant agents 顺序和 history tensor 索引一致，而且 ego 是第 0 个。现实 WOMD/Waymax 中 agent id 可能是字符串或 track id，不保证 `a_idx+1` 就是该 agent。

**后果**：scene_features 可能拿错 agent 历史，训练 surface 会错配 agent。

**建议修改**：

* 在 `SameRootGroup` 里明确保存 `agent_id -> track_index`。
* collate 时用真实 track index，而不是 `a_idx+1`。
* 如果 agent id 是字符串，要在 schema 里保存 mapping。

---

## 2.5 `BRANCHES` 类型与 tests / 旧逻辑不一致

**文件**：`mfrp/data/schema.py:9`

```python
BRANCHES = ("keep", "cede", "brake", "accelerate", "pass", "nonconflict")
```

测试期待 list：

```python
assert BRANCHES == ["keep", ...]
```

这不是理论问题，但说明 API 不稳定。

**建议**：统一成 tuple 或 list，并更新测试。更建议保持 tuple，用 `list(BRANCHES)` 给需要 mutable/list 的地方。

---

# 3. 代码对论文算法的简化或偷懒实现

## 3.1 Scene encoder 被简化成 history summary MLP，缺少论文要求的 map / route / traffic control tokens

**论文要求**：map polylines、lane boundaries、route paths、crosswalks、traffic controls、agent histories、target-agent role token。

**代码定位**：

* `mfrp/data/tensors.py:66-76`：只取 agent history 最后状态和 velocity diff。
* `mfrp/models/mfrp_model.py:47`：`scene_encoder` 是简单 MLP。
* `mfrp/models/mfrp_model.py:61-70`：token 由 `scene_features` 投影得到，没有 vectorized token attention。

**建议修改**：

* 新增 `map_tokens`, `tl_tokens`, `route_tokens`, `agent_history_tokens`。
* `scene_encoder` 改成 transformer / vector attention。
* target agent 加 role embedding。

---

## 3.2 Support encoder 不是论文里的 Set Transformer / slot cross-attention

**代码定位**：

* `mfrp/models/mfrp_model.py:67-69`：probe 先 MLP，再 masked mean，然后 linear projection 成 tokens。
* `mfrp/models/support_encoder.py:8-38` 虽然有 attention slots，但 `MFRPModel` 实际没有用这个 `SupportEncoder` 类。

**问题**：masked mean 会丢掉 support probe 的结构，尤其不能表达“附近 intervention 的 response boundary”。

**建议修改**：

* `MFRPModel` 直接使用 `SupportEncoder`。
* support probes 输入中保留完整 `a_i(u;s)` 和 `e_O(o)`。
* 使用 slot queries cross-attend 所有 support probes，而不是 mean pooling。

---

## 3.3 Trajectory distribution 被简化成每 branch 单条轨迹

**论文要求**：`p(Y | M, r)` 可以是 GMM NLL 或 masked ADE/FDE surrogate，但 response distribution 应该能表达多模态。

**代码定位**：

* `mfrp/models/mfrp_model.py:58`：`traj_head = Linear(hidden, branches * future_steps * state_dim)`。
* `mfrp/models/mfrp_model.py:83`：输出 shape 是 `[B,A,K,branches,T,D]`，没有 modes。
* `mfrp/training/losses.py:36-42`：trajectory loss 是 L2/ADE proxy。

**建议修改**：

* 复用或整合 `mfrp/models/response_heads.py:33-46` 中已有的 `BranchConditionedTrajectoryHead`，支持 modes。
* loss 改为 GMM NLL 或至少 minADE over modes。
* 输出 branch-conditioned uncertainty。

---

## 3.4 Burden / margin likelihood 被简化成 SmoothL1，不是论文中的 probabilistic likelihood

**代码定位**：

* `mfrp/training/losses.py:34-35`：burden 和 margin 都是 `smooth_l1_loss`。
* `mfrp/models/mfrp_model.py:84-87` 虽然输出 `sigma`，但 loss 没有使用 sigma。

**建议修改**：

* 对 `burden_mu/sigma`、`margin_mu/sigma` 使用 Gaussian NLL：

  ```python
  nll = 0.5 * ((target - mu) / sigma)**2 + log(sigma)
  ```
* 或实现 quantile / discretized logistic，对应论文 appendix。

---

## 3.5 Geometry loss 不是论文定义的 `|D_p - d_O|`

**论文要求**：预测分布距离 `D_p` 应匹配观测 response distance `d_O`。

**代码定位**：

* `mfrp/training/losses.py:57-73`
* 当前计算：

  ```python
  pred = tv + 0.2 * |margin diff| + 0.2 * |burden diff|
  return (pred / dist).clamp(max=20).mean()
  ```

这不是 `abs(pred - dist)`，而是惩罚 `pred/dist`。这会鼓励 predicted difference 趋近 0，而不是匹配真实 response distance。

**建议修改**：

```python
target = batch["edge_response_distance"]
loss = torch.abs(pred - target).mean()
```

或者：

```python
loss = smooth_l1(pred / scale, target / scale)
```

同时 edge 中应包含 branch、trajectory、burden、margin 的 normalized response semimetric，而不是只用 candidate 距离。

---

## 3.6 Coercion witness 没有 pairwise ranking loss

**论文要求**：event loss + pairwise ranking。

**代码定位**：

* `mfrp/training/losses.py:76-86`：只有 BCE。
* 没有任何 `ranking_pairs`、`cw_rank_loss`、`epsilon_D`、`epsilon_B` 实现。

**建议修改**：

* 数据集保存 `cw_rank_pairs = (u_plus, u_minus, agent)`。
* loss 增加：

  ```python
  rank_loss = softplus(-(kappa_plus - kappa_minus) / tau)
  ```

---

## 3.7 Selector 没有实现论文中完整的 mechanism-feasible filtering

**代码定位**：

* `mfrp/planning/selector.py:7-31`

缺失：

* `gamma_bar` 参数未使用；
* 没有 local boundary sensitivity；
* 没有 per-agent explanation 输出；
* fallback 只是 `nominal + 10*rho + nu`；
* 没有区分 unsafe、coercive、uncertain、boundary-sensitive violation 类型。

**建议修改**：

返回结构中加入：

```python
active_violations = {
    "risk": rho_cal > alpha,
    "uncertainty": nu > nu_bar,
    "boundary": gamma > gamma_bar,
    "coercion": kappa_i.max(dim=agent) > kappa_bar,
}
```

并在 experiments 里统计 rejection reasons。

---

# 4. 数据集构造流程是否理论上满足论文需求

## 4.1 理论上需要什么数据

论文需要的数据不是普通 WOMD log，也不是 log playback。它必须满足：

1. **same-root reset**：所有 candidate 从同一个 `t0` simulator state 开始；
2. **ego intervention**：ego 被强制执行 candidate `u^k`；
3. **reactive surrounding agents**：其他 agent 必须对 ego intervention 反应；
4. **multiple variants / seeds / policies**：同一 candidate 下需要 ceding 和 non-ceding 证据；
5. **support/query 按 candidate 切分**：query candidate 的所有 variants 都不能进入 support；
6. **response primitive**：每个 `(scene, agent, candidate, variant)` 要有 branch、trajectory、burden、safety margin；
7. **priority 只能使用 pre-execution metadata**；
8. **部署时绝不能用 support probes 或 future labels**。

---

## 4.2 当前代码理论上不能完整满足论文需求

### 4.2.1 当前 adapter 没有实现 reactive rollout，只要求外部 cache

**文件**：`examples/mfrp_waymax_adapter.py:325-425`

它要求：

```python
config.adapter.rollout_cache
```

并从 `.npy` 读取 agent trajectory：

* `examples/mfrp_waymax_adapter.py:316-322`
* `examples/mfrp_waymax_adapter.py:404-406`

但它自己没有实现 same-root reset、ego control、Waymax reactive policy rollout。

**评价**：这可以作为 integration hook，但不能算论文实验的数据构建实现。

**建议修改**：

* 明确实现 `reactive_rollout_fn(root_state, ego_candidate, policy_variant, seed)`。
* 保存 rollout metadata：

  * same_root_hash；
  * simulator seed；
  * policy variant；
  * ego candidate id；
  * t0；
  * all agent validity masks。

---

### 4.2.2 Candidate library 过于简化，且负 delay 实际被吃掉

**文件**：`examples/mfrp_waymax_adapter.py:230-250`

```python
delays = [-0.2, 0.0, 0.2, 0.4, 0.8]
delay_steps = max(0, int(round(float(ds) / dt)))
```

负 delay 被 `max(0, ...)` 变成 0，所以所谓 earlier / assertive timing 没有实现。

**建议修改**：

* 对负 delay 应沿轨迹前推或重采样，而不是 clamp 成 0。
* candidate 应覆盖：

  * lane-consistent path；
  * target gap；
  * speed profile；
  * acceleration；
  * lateral commitment；
  * yield / neutral / assertive。
* 对不可行 candidate 加 `valid=False` 和 invalid reason。

---

### 4.2.3 Interaction region 与 entry time 目前不是论文定义

**文件**：

* `mfrp/data/interaction_region.py:24-36`
* `mfrp/data/label_extraction.py:19-46`

当前 `closest_entry_times()` 只是找两条轨迹距离小于 threshold 的第一个点，而且 ego 和 agent entry time 返回同一个 `j*dt`：

```python
return (j * dt, j * dt, d[j])
```

这不符合论文里 conflict region `Z_i^k` 的 oriented-box intersection / lane-graph interval 定义。

**后果**：

* cede / pass / accelerate branch label 会错；
* entry gap `g_i(u)` 不可信；
* priority score 中 neutral ordering 不可信；
* burden 的 delay 项不可信。

**建议修改**：

* 实现 swept oriented box 与 conflict region intersection；
* 分别计算 ego in/out 和 agent in/out；
* actor never enters 时设置 censored mask，不要只给 `inf` 或最近点。

---

### 4.2.4 Safety margin 用圆/点距离，不是 oriented-box separation

**文件**：`mfrp/data/interaction_region.py:13-21`

当前：

```python
d = norm(ego_xy - agent_xy) - (ego_radius + agent_radius)
```

论文要求 minimum signed oriented-box separation。

**建议修改**：

* 用 `(x, y, yaw, length, width)` 计算 oriented box；
* 使用 SAT separating axis theorem 得到 signed distance / penetration depth；
* training、calibration、evaluation 使用同一个 margin 定义。

---

### 4.2.5 Branch label 是 heuristic softmax，不是完整 evidence score

**文件**：`mfrp/data/label_extraction.py:19-46`

问题：

* nonconflict 用固定 `min_dist > 12m`，不是 scenario-specific threshold；
* cede 只看 `tau_agent_in - tau_agent_base_in` 和 ego/agent ordering；
* pass 用 `tau_agent_in + 0.5 < tau_ego_in`，没有 `tau_agent_out`；
* baseline 是 `baseline_agent_traj`，但当前数据构建里可能是 constant velocity，不是 same variant neutral baseline。

**建议修改**：

* 按论文 appendix 重写 evidence：

  * `e_cede`
  * `e_brake`
  * `e_accel`
  * `e_pass`
  * `e_keep`
  * `nonconflict mask`
* 保存 hard branch 和 soft branch；
* branch taxonomy 和 `CEDING_BRANCHES` 统一放 schema。

---

### 4.2.6 Burden baseline 不满足论文“same reactive-policy variant”要求

**文件**：`mfrp/data/label_extraction.py:49-70`

当前 burden 是 agent trajectory 相对 baseline trajectory，但 adapter 中 baseline 倾向于 `constant_velocity_extrapolate` 或 neutral ref，而不是同一 reactive-policy variant 下的 neutral candidate rollout。

论文要求：`y_i^{0,r}`，也就是同 variant `r` 下 neutral candidate 的响应。

**建议修改**：

* 对每个 variant 先确定 neutral candidate `u0`；
* 使用同一 `(scene, agent, variant)` 下 `u0` rollout 作为 baseline；
* 如果 neutral baseline invalid，mask burden 项并降低 witness confidence。

---

### 4.2.7 Support/query split 没有按 conservative / neutral / assertive stratify

**文件**：

* `mfrp/data/support_query.py:7-38`
* `examples/mfrp_waymax_adapter.py:273-283`

当前要么随机，要么取最后一部分 query。论文要求 support set stratified，包括 conservative、neutral、assertive；query 包括 near-neighbor perturbations 和大变化。

**建议修改**：

* candidate metadata 增加 family：

  * conservative
  * neutral
  * assertive
  * yield
  * speed-up
  * lane-commit
* split 时每个 family 至少一个 support；
* query 包含 local neighbor 和 far perturbation。

---

### 4.2.8 `support_probe_features` 只放第一帧轨迹，信息严重不足

**文件**：`mfrp/data/tensors.py:114-120`

```python
traj_summary = trajectory[..., :1, :state_dim]
```

论文里的 `e_O(o)` 应包含 branch、trajectory features、burden、margin、high-pressure、validity masks。只取第一帧几乎无法表示 response motion。

**建议修改**：

* trajectory summary 至少包括：

  * ADE/FDE to baseline；
  * min distance；
  * speed/acc/jerk stats；
  * entry/exit times；
  * final state；
  * mask ratio。
* 或用 small trajectory encoder，而不是手工第一帧。

---

# 5. 论文 Experiments 对应代码缺口

论文 experiments 有 5 个：

1. Same-root held-out response prediction；
2. Surface geometry and coercion attribution；
3. Coercive false-safe offline selection；
4. Closed-loop non-coercive planning；
5. Robustness and ablation stress tests。

当前代码：

* `scripts/eval_mfrp.py:12-27` 只做 branch CE/acc、risk mean、AUROC/ECE；
* 没有 closed-loop simulation；
* 没有 false-safe rejection analysis；
* 没有 support-adapted vs scene-only 对比；
* 没有 ablation runner；
* 没有 boundary/surface geometry 指标；
* 没有 qualitative surface slice 生成。

**建议新增文件**：

* `scripts/eval_heldout_response.py`
* `scripts/eval_surface_geometry.py`
* `scripts/eval_false_safe_selection.py`
* `scripts/eval_closed_loop_waymax.py`
* `scripts/run_ablations.py`
* `scripts/plot_surface_slices.py`

---

# 6. 最优先修改清单

按优先级，我建议你后续让大模型按这个顺序改：

## P0：先让代码能跑通真实数据构建

重点文件：

* `mfrp/data/schema.py`
* `mfrp/data/scene_schema.py`
* `examples/mfrp_waymax_adapter.py`
* `mfrp/data/label_extraction.py`
* `mfrp/data/interaction_region.py`
* `mfrp/data/priority.py`

目标：

* 统一 schema；
* adapter 能 import；
* RootScene / EgoCandidate / SameRootGroup 字段一致；
* 实现 reactive rollout 或明确读取 cache 的格式；
* 补齐 `make_response_observation()` 和 `coercion_witness_label()`。

## P1：修数据理论正确性

重点文件：

* `mfrp/data/interaction_region.py`
* `mfrp/data/label_extraction.py`
* `mfrp/data/tensors.py`
* `mfrp/data/support_query.py`

目标：

* oriented-box margin；
* conflict region entry/exit；
* censored masks；
* same-variant neutral baseline；
* candidate-level ceding/non-ceding aggregation；
* stratified support/query split。

## P2：修模型与 loss 对齐论文

重点文件：

* `mfrp/models/mfrp_model.py`
* `mfrp/models/support_encoder.py`
* `mfrp/models/response_heads.py`
* `mfrp/models/coercion_witness.py`
* `mfrp/training/losses.py`

目标：

* scene encoder 支持 map/route/tl/agent tokens；
* support encoder 使用 Set Transformer / slot attention；
* trajectory head 支持 multimodal；
* burden/margin 使用 NLL；
* geometry loss 改成 `|D_p - d_O|`；
* coercion witness 加 ranking loss 和 confidence mask。

## P3：修部署和实验闭环

重点文件：

* `mfrp/planning/estimators.py`
* `mfrp/planning/selector.py`
* `mfrp/planning/calibration.py`
* `scripts/calibrate_mfrp.py`
* `scripts/eval_mfrp.py`
* `scripts/deploy_mfrp.py`

目标：

* `gamma_bar` 真正生效；
* 输出 violation reasons；
* calibration 做 selector-level calibration；
* eval 覆盖论文五个实验；
* deployment 输入只允许 observed scene + ego candidate + preexec priority。

