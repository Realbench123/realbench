# 开环评测协议（草案 v0）

状态：**草案，数值待组内确认。** 下面标「待定」的项不要当作已定。

开环评测问的是：给定数据集里一个真实时刻的观测，模型输出的动作块和示教的动作块差多少。它不涉及闭环反馈，不能代替真机。

## 为什么要版本化

协议里任何一项变了，曲线就不能和旧曲线直接比。因此：
- 评测记录里写 `protocol.name` 和 `protocol.version`。
- `version` 是本文件下面「协议参数」那个 yaml 块内容的 sha256 前 8 位，由 `tools/protocol_hash.py` 算。
- 改任何参数就得到新的版本号，页面上同一版本号的评测才会叠在同一张曲线里。

## 协议参数

```yaml
name: openloop-fold-v0
status: draft
task: fold-blue-fabric             # 待定：每个任务一份协议
episodes:
  source: val_episodes             # 用该训练运行自己的验证集划分
  list_sha256: null                # 待定：固定一份 episode 列表后写入，保证所有模型评同一批
  note: "现有 offline_infer_claude/run_eval.py 写死了 4 个 episode，和训练的验证集不一致，需要对齐"
sampling:
  stride_frames: 15                # 待定：沿用现有脚本的 15
  max_requests_per_episode: 12     # 待定：沿用现有脚本的 12
  skip_static: true                # 跳过开头和结尾的静止段，判据：相邻动作最大变化 > 1e-3
seeds:
  count: 3                         # 待定：沿用现有脚本的 3
  base: 1000
horizon_steps: 30
prompt: "见各任务的 instruction 字段，评测记录里存原文"
metrics:
  - name: l1_arm
    desc: 12 个臂关节动作块的平均绝对误差（弧度），对种子取平均
  - name: l1_gripper
    desc: 2 个夹爪通道的平均绝对误差，单独报，不并入臂
  - name: l1_by_horizon
    desc: 按块内步位（前 10 步、中 10 步、后 10 步）分别报，看误差是否集中在块尾
  - name: seed_spread
    desc: 同一请求不同种子输出之间的平均分歧，衡量采样噪声
  - name: first_step_err
    desc: 块内第 1 步的误差，和闭环里的接缝有关
report:
  - 同时给出每个请求的明细，不只给均值
  - 种子数少于 3 时标警告
```

## 已知问题（来自对现有文件的核对）

1. **验证集定义不一致。** 数据目录 `README.md` 写「failed IDs 0,1,23 are validation only」，`split.json` 的验证集是 `[0,1,5,22,23]`。是文档写漏还是划分有误，需要数据负责人确认。**在确认前，本协议不固定 episode 列表。**
2. **现有开环脚本评的是叠杯子，不是叠毛巾。** `offline_infer_claude/run_eval.py` 读 `data/episode_{104,116,12,120}.hdf5`，提示词写死 `"Stack the cups."`，属于叠杯子任务。叠毛巾的划分是另一套编号，这四个编号碰巧也出现在叠毛巾的训练集编号里，但两者是不同数据，不能因编号相同就当成同一批。这个脚本不能直接拿来评叠毛巾模型。
3. **这 4 个 episode 是否在叠杯子模型的训练集里，我没查。** 如果在，现有的 `res_tau0_relative.npz` 是训练集上的开环误差，不能当泛化指标。需要叠杯子训练的划分文件才能判断。

## 不在协议里的事

- 训练过程中的验证集 loss 曲线。它属于训练记录，不属于开环评测。
- 闭环指标，见真机评测。
