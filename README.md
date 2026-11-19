# RealBench

一个让整个组的训练、checkpoint、部署配置、开环评测和真机评测放在同一处、能互相追溯的记录库。

要回答的问题：**真机结果变好或变差时，是哪一环变了。** 记录沿这条链组织：

```
数据 → 训练运行 → checkpoint → 部署配置 → 开环 / 真机评测
```

每个环节有固定字段，缺的字段在页面上标红，不留空白。

## 状态

框架已搭好，**正式记录目前为空**。`.local/` 里有两条训练和一条真机评测，是用来测试导入器和页面的真实产物，不进仓库、不对外。

## 目录

| 路径 | 内容 |
|---|---|
| `schema/` | 训练、评测、数据配方三种记录的 JSON Schema |
| `records/runs/`、`records/evals/`、`records/recipes/` | 正式记录，一条一个 json，**会公开** |
| `curves/` | 训练曲线（逐步 loss、grad_norm、lr、步耗时） |
| `protocols/` | 评测协议，改任何一项就是新版本 |
| `tools/` | 导入器、校验器、站点生成器、视频上传 |
| `site/` | 页面模板和脚本，生成单文件 `dist/index.html` |
| `.local/` | 本地样本和草稿，被 `.gitignore` 排除 |

## 日常使用

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 0. 导入数据配方（训练引用它；同一配方可被多次训练共用）
.venv/bin/python tools/import_tau0_recipe.py <含 split.json 和 frame_keep.json 的目录> --id rc-xxx --name "..." --task <task> --robot "..." \
    --source-name <源数据集名> --excluded "27:原因" --readme-train-anchors N --readme-val-anchors M --out records/recipes

# 1. 导入一次训练（τ₀ 训练器；其他训练栈写一个适配器输出同样的记录）
.venv/bin/python tools/import_tau0_run.py <训练输出目录> --id tr-xxx --machine "..." --split-dir <含 split.json 的目录> \
    --kind finetune --kind-note "归类依据" --recipe rc-xxx --out records/runs

# 2. 导入真机运行（每个运行目录是一次试验）。成败不会自动判，要看视频后手工标
.venv/bin/python tools/import_arx63_eval.py <运行目录...> --id ev-xxx --run-id tr-xxx --step N --weight-sha256 <sha> --task <task> --out records/evals

# 3. 视频：默认 dry-run，确认能公开后加 --apply
.venv/bin/python tools/upload_video.py records/evals/ev-xxx.json --trial 1 --camera head --video <原始视频> --repo <org>/realbench-videos

# 4. 校验并预览
.venv/bin/python tools/validate.py
.venv/bin/python tools/build_site.py && open site/dist/index.html
```

## 规则

- **训练类型要人工确认。** 导入器只给建议值，`kind_note` 必须写归类依据。
- **权重哈希必填。** 真机评测写服务实际加载的哈希，校验器会和训练记录里同一步数的哈希比对。端口号分不出模型。
- **服务启动时间不知道就留空**，页面会标缺。不要编。
- **成败由看视频的人标**，`label_by` 写人名。导入器一律写 `unlabeled`。
- **预测在跑之前写**，带时间戳。事后补的标 `after_the_fact`，页面会显示。
- **闭环每个设置至少 3 次**，不足的页面标警告。
- **配方的编号要说清楚。** `split.json` 的编号是转换后数据集的下标，不是源 episode 编号。导入器用 README 里的锚点数反向验证换算，对不上就拒绝写入。
- **“保留帧数”和“锚点数”是两个数。** 训练日志里的 `kept N/M frames` 的 N 是锚点数（整块动作都落在保留区内），页面分开显示。
- **基线要显式指定**（`baseline_id`），页面不猜。
- **汇总字段不可信。** 训练器写的 `train_loss` 在续训后可能是 0，导入器从逐步日志重算。
- **硬件看日志，不看配置里的 notes。** 全局 batch = 每卡 × 累积 × `world_size`（读训练日志）。

## 公开前的约定

`records/` 会公开。校验器会拦截内部路径、内网地址和 ssh 端口。视频上传到公开数据集，上传前确认画面里没有不想公开的人和场景。**同事的训练和评测结果，要经本人同意才能放进 `records/`。**

## 视频

视频放 Hugging Face 公开数据集，路径 `<family>/<run_id>/<eval_id>/trial_<N>.<camera>.mp4`。页面直接用 `<video>` 播放，所以上传前会转成 H.264 + yuv420p + faststart。视频数据集：<https://huggingface.co/datasets/Realbench123/realbench-videos>（公开）。

播放链路的实测结果（2026-11-19，用一段合成的 4 秒测试视频，已删除）：
- 匿名请求（不带令牌）能取到，支持 Range，返回 `access-control-allow-origin`；下载后 sha256 与上传时一致。
- 在无头 Chrome 里，本站的 `<video>` 标签读到元数据、`play()` 成功，独立页面里拖动到 1.0 秒后能继续前进。
- **没测的：** Safari、Firefox、手机浏览器；大文件（几十 MB）下的起播延迟；HF 对站外直链的限流策略。

## 开环协议

草案见 `protocols/openloop-fold-v0.md`，数值待组内确认。协议版本是参数块的内容哈希：`python tools/protocol_hash.py protocols/openloop-fold-v0.md`。
