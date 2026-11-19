#!/usr/bin/env python3
"""τ₀ 训练产物 → realbench 训练记录。

读一个训练输出目录里的 run_spec.json、resolved_config_full.yaml、trainer_state.json、
finch_data_spec/ 和训练日志，输出 records/runs/<id>.json 与 curves/<id>.json。

只读，不改训练目录。目录可以在本机（用 rsync 拉过来的副本），也可以先 ssh 拉小文件再指向本地副本。
训练类型 kind 只给建议值，必须人工确认后写入 kind。
"""
import argparse, hashlib, json, os, re, sys, datetime, glob
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import redact
from _admin_core import blocked_state

VERSION = "tau0-importer-0.1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha256_file(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def to_iso(s, tz):
    """训练日志和 run_spec 里的时间没有时区。云端容器的 date 是 UTC，默认按 UTC 解释，再换成本机时区的 ISO 时间。"""
    if not s:
        return None
    d = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    return d.astimezone().isoformat(timespec="seconds")


def parse_world_size(log_text):
    m = re.search(r"world_size=(\d+), nodes=(\d+)", log_text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def parse_log(log_text):
    """从训练日志头尾读：world size、保留帧数、起止时间、GPU 序号。"""
    out = {"world_size": None, "nodes": None, "frames_kept": None, "frames_total": None,
           "gpu_ids": [], "start": None, "end": None}
    if not log_text:
        return out
    out["world_size"], out["nodes"] = parse_world_size(log_text)
    m = re.search(r"kept ([\d,]+)/([\d,]+) frames", log_text)
    if m:
        out["frames_kept"] = int(m.group(1).replace(",", ""))
        out["frames_total"] = int(m.group(2).replace(",", ""))
    out["gpu_ids"] = sorted({int(x) for x in re.findall(r"\[N\d+,R\d+,GPU(\d+)\]", log_text)})
    ts = re.findall(r"\[INFO\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_text)
    if ts:
        out["start"], out["end"] = ts[0], ts[-1]
    return out


def curve_from_state(state):
    """逐步日志 → 曲线。不用汇总条目（续训后 train_loss 可能为 0，见 summary_bad）。"""
    rows = [x for x in state["log_history"] if "loss" in x and "train_loss" not in x and "step" in x]
    keys = ["loss", "grad_norm", "learning_rate", "step_time_ms"]
    curve = {"step": [r["step"] for r in rows]}
    for k in keys:
        curve[k] = [r.get(k) for r in rows]
    return curve


def implied_global_batch(state):
    """用汇总里的 train_samples_per_second / train_steps_per_second 反推全局 batch。汇总无效时返回 None。"""
    for x in state["log_history"]:
        if "train_samples_per_second" in x and "train_steps_per_second" in x:
            if (x.get("train_runtime") or 0) < 1:
                return None
            return x["train_samples_per_second"] / x["train_steps_per_second"]
    return None


def summary_is_bad(state):
    """训练器末尾的汇总条目：运行时长近 0 或 train_loss 为 0，说明是续训后的无效汇总。"""
    for x in state["log_history"]:
        if "train_loss" in x:
            if x.get("train_loss") == 0 or (x.get("train_runtime") or 1) < 1:
                return True
    return False


def suggest_kind(model_args, base_path):
    """只给建议值。"""
    if not base_path:
        return "scratch"
    if model_args.get("training_time_rtc"):
        return "posttrain"
    return "finetune"


def trainable(model_args):
    m = {"tune_mm_vision": "vision", "tune_mm_mlp": "mlp", "tune_mm_llm": "llm", "tune_vla_dit": "dit"}
    return [v for k, v in m.items() if model_args.get(k)]


def build(args):
    d = args.dir
    R = (lambda x: x) if args.no_redact else redact   # 对外公开：默认去掉绝对路径
    spec = jload(os.path.join(d, "run_spec.json"))
    cfg = yaml.safe_load(open(os.path.join(d, "resolved_config_full.yaml"), encoding="utf-8"))
    state = jload(os.path.join(d, "trainer_state.json"))
    ma, da, ta = spec["model_args"], spec["data_args"], spec["training_args"]

    log_text = ""
    lp = args.log or os.path.join(d, "log", "head_tail.txt")
    if os.path.exists(lp):
        log_text = open(lp, encoding="utf-8", errors="ignore").read()
    lg = parse_log(log_text)

    # 全局 batch 必须用日志里的 world_size 算；run_spec 里的 world_size 是空的
    ws = lg["world_size"] or args.world_size
    per_dev = ta.get("per_device_train_batch_size")
    accum = ta.get("gradient_accumulation_steps") or 1
    if not ws:
        raise SystemExit("读不到 world_size：训练日志缺失，请用 --world-size 指定并在 notes 里说明依据")
    gb = per_dev * accum * ws
    implied = implied_global_batch(state)
    batch_check = None
    if implied is not None:
        batch_check = "一致" if abs(implied - gb) / gb < 0.02 else "不一致"

    # 数据划分
    split = None
    if args.split_dir:
        sp = os.path.join(args.split_dir, "split.json")
        if os.path.exists(sp):
            split = jload(sp)
    n_train = (split or {}).get("train_episodes")
    n_val = (split or {}).get("validation_episodes")
    if n_train is None:
        n_train = args.train_episodes or 0
        n_val = args.val_episodes or 0

    curve = curve_from_state(state)
    losses = [x for x in curve["loss"] if x is not None]
    window = min(10, len(losses))
    final = sum(losses[-window:]) / window if losses else None

    by_step = {}
    for st_ in args.checkpoint_steps or []:
        by_step[int(st_)] = {"step": int(st_), "path": f"checkpoint-{st_}", "sha256": None, "hf_repo": None, "hf_revision": None, "kept": True}
    if args.final_sha:
        fs = state["global_step"]   # 同一步数只留一条，哈希是训练输出根目录 model.safetensors 的
        by_step[fs] = {"step": fs, "path": "model.safetensors", "sha256": args.final_sha, "hf_repo": None, "hf_revision": None, "kept": True}
    ckpts = [by_step[k] for k in sorted(by_step)]

    tz = datetime.timezone(datetime.timedelta(hours=args.log_tz_hours))
    start_iso = args.start or to_iso(spec.get("start_time") or lg["start"], tz)
    runtime = next((x.get("train_runtime") for x in state["log_history"] if "train_runtime" in x), None)
    end_iso = args.end
    if not end_iso and start_iso and runtime and runtime > 1 and not summary_is_bad(state):
        end_iso = (datetime.datetime.fromisoformat(start_iso) + datetime.timedelta(seconds=runtime)).isoformat(timespec="seconds")
    if not end_iso:
        end_iso = to_iso(lg["end"], tz)

    st = args.status or ("finished" if state["global_step"] >= state["max_steps"] else "stopped")
    dirty = spec.get("git_dirty")
    gh = spec.get("git_hash")
    if gh in (None, "", "not_a_git_repo") or str(gh).startswith("not_a_g"):
        gh, dirty = None, None

    rec = {
        "id": args.id,
        "name": args.name or spec.get("notes", "")[:60] or args.id,
        "owner": args.owner or "",
        "family": args.family,
        "baseline_id": args.baseline,
        "recipe_id": args.recipe,
        "kind": args.kind or "finetune",
        "kind_note": args.kind_note or "",
        "kind_suggested": suggest_kind(ma, ma.get("model_name_or_path")),
        "base": {"model_path": R(ma["model_name_or_path"]), "base_run_id": args.base_run},
        "data": {
            "sources": [{"name": da.get("config_name", ""), "episodes": n_train, "weight": None, "kind": "success"}],
            "train_episodes": n_train,
            "val_episodes": n_val,
            "train_anchors": lg["frames_kept"],
            "split_file": "configs/arx_fold_blue_relative/split.json" if split else None,
            "split_sha256": sha256_file(os.path.join(args.split_dir, "split.json")) if split else None,
            "config_name": da.get("config_name", ""),
        },
        "hparams": {
            "max_steps": ta["max_steps"], "per_device_batch": per_dev, "grad_accum": accum, "global_batch": gb,
            "learning_rate": ta["learning_rate"], "scheduler": ta["lr_scheduler_type"],
            "warmup_steps": ta["warmup_steps"], "weight_decay": ta.get("weight_decay"), "seed": ta.get("seed"),
            "trainable": trainable(ma),
            "extra": {k: ma[k] for k in ("loss_type", "num_steps", "training_time_rtc", "rtc_max_delay") if k in ma},
        },
        "hardware": {
            "machine": args.machine, "gpus": ws, "gpu_type": args.gpu_type,
            "gpu_indices": lg["gpu_ids"] or None,
        },
        "code": {
            "repo": args.repo, "git_hash": gh, "dirty": dirty,
            "command": R(spec.get("command")), "config_yaml_sha256": spec.get("yaml_sha256"),
        },
        "time": {"start": start_iso, "end": end_iso, "status_reason": args.status_reason},
        "status": st,
        "loss": {
            "first": losses[0] if losses else None, "final": final, "final_window": window,
            "curve_file": f"curves/{args.id}.json", "summary_bad": summary_is_bad(state),
        },
        "checkpoints": ckpts,
        "deployed_as": [],
        "evals_offline": [],
        "evals_closed": [],
        "notes": args.notes or "",
        "source": {"kind": "auto-import", "path": R(os.path.abspath(d)), "imported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "importer_version": VERSION},
    }
    # notes 里的硬件描述不可信：与日志里的 world_size 不符时提示
    if start_iso and not args.start:
        rec["notes"] = (rec["notes"] + " " if rec["notes"] else "") + \
            f"[导入器] 起止时间来自 run_spec 或训练日志，原文无时区，按 UTC{args.log_tz_hours:+g} 解释。"
    if batch_check:
        rec["notes"] = (rec["notes"] + " " if rec["notes"] else "") + \
            f"[导入器核对] 全局 batch {gb}（{per_dev}×{accum}×{ws}）；由训练器汇总反推为 {implied:.1f}，{batch_check}。"
    spec_notes = str(spec.get("notes", ""))
    m = re.search(r"(\d+)\s*x\s*H\d+", spec_notes)
    if m and int(m.group(1)) != ws:
        rec["notes"] = (rec["notes"] + " " if rec["notes"] else "") + \
            f"[导入器核对] 配置里的 notes 写 {m.group(0)}，训练日志 world_size={ws}，以日志为准。"
    return rec, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="训练输出目录（含 run_spec.json）")
    ap.add_argument("--id", required=True)
    ap.add_argument("--name")
    ap.add_argument("--owner")
    ap.add_argument("--family", default="tau0")
    ap.add_argument("--kind", choices=["scratch", "pretrain", "finetune", "posttrain"])
    ap.add_argument("--kind-note")
    ap.add_argument("--base-run")
    ap.add_argument("--recipe", help="数据配方 id（rc-...）")
    ap.add_argument("--baseline", help="对比用的基线训练 id；不给则页面不显示差异")
    ap.add_argument("--no-redact", action="store_true", help="保留绝对路径（只用于私有仓库）")
    ap.add_argument("--machine", required=True)
    ap.add_argument("--gpu-type")
    ap.add_argument("--repo", default="tau0_vla")
    ap.add_argument("--log", help="训练日志（头部即可）")
    ap.add_argument("--split-dir", help="含 split.json 的目录")
    ap.add_argument("--world-size", type=int)
    ap.add_argument("--train-episodes", type=int)
    ap.add_argument("--val-episodes", type=int)
    ap.add_argument("--output-root")
    ap.add_argument("--checkpoint-steps", nargs="*")
    ap.add_argument("--final-sha")
    ap.add_argument("--start", help="ISO 时间；不给则读 run_spec 的 start_time")
    ap.add_argument("--end", help="ISO 时间；不给则用 start + 训练器记录的运行时长")
    ap.add_argument("--log-tz-hours", type=float, default=0, help="训练日志与 run_spec 时间的 UTC 偏移（小时），云端默认 0")
    ap.add_argument("--status", choices=["finished", "running", "failed", "stopped", "superseded"])
    ap.add_argument("--status-reason")
    ap.add_argument("--notes")
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "runs"), help="输出目录（默认 .local/，不进仓库）")
    args = ap.parse_args()

    st_ = blocked_state(args.id)
    if st_:
        sys.exit(f"{args.id} 在删除清单里（{st_}），不导入。要重新导入，先从 records/_deleted.json 里去掉它。")
    rec, curve = build(args)
    os.makedirs(args.out, exist_ok=True)
    cdir = os.path.join(os.path.dirname(args.out.rstrip("/")), "curves")
    os.makedirs(cdir, exist_ok=True)
    json.dump(rec, open(os.path.join(args.out, args.id + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(curve, open(os.path.join(cdir, args.id + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("写入", os.path.join(args.out, args.id + ".json"))
    print("  建议类型:", rec["kind_suggested"], "| 记录类型:", rec["kind"], "(必须人工确认)")
    print("  全局 batch:", rec["hparams"]["global_batch"], "=", rec["hparams"]["per_device_batch"], "×", rec["hparams"]["grad_accum"], "×", rec["hardware"]["gpus"])
    print("  末值 loss(末 %d 条均值): %.5f" % (rec["loss"]["final_window"], rec["loss"]["final"]))
    print("  汇总不可信:", rec["loss"]["summary_bad"], "| 代码 dirty:", rec["code"]["dirty"])
    if rec["notes"]:
        print("  注:", rec["notes"])


if __name__ == "__main__":
    main()
