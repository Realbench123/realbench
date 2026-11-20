#!/usr/bin/env python3
"""OpenWAM 训练 → realbench 训练记录。

输入是扫描器 scan.json 里的一个候选（--scan + --cand-dir），不直接读共享盘，所以可以在本机运行。
曲线来自扫描器按启动时刻切好的 supervisor.log 进度行（--curves）。

loss 口径：OpenWAM 全局 batch 小，单步 loss 噪声大（最后 200 步标准差约 0.04）。
因此“末值”取最后 N 个记录点的均值，N 默认 200，并把最后一条单步值另存到 notes。
类型 kind 必须人工确认，这里只接受命令行给的值。
"""
import argparse, datetime, json, os, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import redact
from _admin_core import blocked_state

VERSION = "openwam-importer-0.1"
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="scan.json")
    ap.add_argument("--cand-dir", required=True, help="候选的目录结尾，如 openwam-arx-0909-8gpu/2026-11-04_21-36-43")
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--owner", default="")
    ap.add_argument("--kind", required=True, choices=["scratch", "pretrain", "finetune", "posttrain"])
    ap.add_argument("--kind-note", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--gpus", type=int, required=True, help="卡数，来自训练脚本（8）；扫描器读不到，必须人工给")
    ap.add_argument("--gpu-type", default=None)
    ap.add_argument("--recipe", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--start", help="ISO 时间。扫描器读不到起止时间时人工给")
    ap.add_argument("--end")
    ap.add_argument("--final-sha", help="最终 checkpoint 的 sha256")
    ap.add_argument("--final-step", type=int)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--notes", default="")
    ap.add_argument("--repo", default="OpenWAM")
    ap.add_argument("--git-hash", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "runs"))
    a = ap.parse_args()

    st = blocked_state(a.id)
    if st:
        sys.exit(f"{a.id} 在删除清单里（{st}），不导入。")
    scan = json.load(open(a.scan, encoding="utf-8"))
    hits = [c for c in scan["candidates"] if c["dir"].rstrip("/").endswith("/" + a.cand_dir.strip("/"))]
    if len(hits) != 1:
        sys.exit(f"按 {a.cand_dir} 找到 {len(hits)} 个候选，需要恰好 1 个")
    c = hits[0]
    if c["kind"] != "openwam":
        sys.exit(f"候选类型是 {c['kind']}，不是 openwam")
    curve = scan["curves"].get(c["id"])
    if not curve:
        sys.exit("扫描结果里没有这个候选的曲线")

    steps, loss = curve["step"], curve["loss"]
    # 末值用扫描器单独存的全量末段（不降采样）；没有就拒绝，不要拿降采样点冒充
    tail = c.get("tail_loss")
    if not tail:
        sys.exit("扫描结果里没有全量末段 tail_loss，请用新版扫描器重扫")
    pts = [l for s_, l in zip(tail["step"], tail["loss"]) if s_ > tail["step"][-1] - a.window]
    if len(pts) < min(a.window, 50):
        sys.exit(f"最后 {a.window} 步只有 {len(pts)} 个点，不够")
    final, spread = statistics.mean(pts), statistics.pstdev(pts)
    last_single = tail["loss"][-1]

    per, acc, ws = c.get("per_device_batch"), c.get("grad_accum") or 1, a.gpus
    gb = per * acc * ws if per else None
    base = os.path.basename((c.get("base") or "").rstrip("/")) or None
    ckpts = []
    for k in c["checkpoints"]:
        ckpts.append({"step": k["step"], "path": k["name"], "sha256": a.final_sha if (a.final_step and k["step"] == a.final_step) else None,
                      "hf_repo": None, "hf_revision": None, "kept": True})
    ckpts.sort(key=lambda x: x["step"])

    note = (f"[导入器] loss 末值取最后 {a.window} 步内 {len(pts)} 个采样点的均值 {final:.4f}（标准差 {spread:.4f}）；"
            f"最后一条单步值是 {last_single:.4f}，噪声大，不要当收敛值。页面曲线降采样到 {len(steps)} 点，末值用的是未降采样的全量。")
    if a.notes:
        note = a.notes + " " + note
    rec = {
        "id": a.id, "name": a.name, "owner": a.owner, "family": "openwam", "kind": a.kind, "kind_note": a.kind_note,
        "kind_suggested": "finetune", "baseline_id": a.baseline, "recipe_id": a.recipe,
        "base": {"model_path": base or "", "base_run_id": None},
        "data": {"sources": [{"name": os.path.basename((c.get("dataset") or "").rstrip("/")) or "未记录", "episodes": 0, "weight": None, "kind": "other"}],
                 "train_episodes": 0, "val_episodes": 0, "train_anchors": None, "split_file": None, "split_sha256": None,
                 "config_name": os.path.basename((c.get("dataset") or "").rstrip("/")) or ""},
        "hparams": {"max_steps": c["max_steps"], "per_device_batch": per, "grad_accum": acc, "global_batch": gb or 1,
                    "learning_rate": c["learning_rate"], "scheduler": c.get("scheduler") or "未记录",
                    "warmup_steps": 0, "weight_decay": None, "seed": None, "trainable": [],
                    "extra": {k: v for k, v in (("zero_stage", c.get("zero_stage")), ("warmup_ratio", c.get("warmup_ratio")),
                                                ("save_steps", c.get("save_steps"))) if v is not None}},
        "hardware": {"machine": a.machine, "gpus": ws, "gpu_type": a.gpu_type, "gpu_indices": None},
        "code": {"repo": a.repo, "git_hash": a.git_hash, "dirty": None, "command": None, "config_yaml_sha256": None},
        "time": {"start": a.start, "end": a.end, "status_reason": None},
        "status": c["status"],
        "loss": {"first": loss[0], "final": final, "final_window": len(pts), "curve_file": f"curves/{a.id}.json", "summary_bad": False},
        "checkpoints": ckpts, "deployed_as": [], "evals_offline": [], "evals_closed": [], "notes": note,
        "source": {"kind": "auto-import", "path": redact(c["dir"]),
                   "imported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "importer_version": VERSION},
    }
    os.makedirs(a.out, exist_ok=True)
    cdir = os.path.join(os.path.dirname(a.out.rstrip("/")), "curves")
    os.makedirs(cdir, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, a.id + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 曲线统一成页面认识的键：loss 以外的 video/action 另存为 extra 序列，页面暂只画 loss/lr
    out_curve = {"step": steps, "loss": loss, "grad_norm": [None] * len(steps), "learning_rate": curve.get("lr"),
                 "step_time_ms": [None] * len(steps), "video": curve.get("video"), "action": curve.get("action")}
    json.dump(out_curve, open(os.path.join(cdir, a.id + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("写入", os.path.join(a.out, a.id + ".json"))
    print(f"  步数 {c['global_step']}/{c['max_steps']} | 全局 batch {gb}（{per}×{acc}×{ws}） | 末值 {final:.4f}±{spread:.4f}（{len(pts)} 点）| 单步最后值 {last_single:.4f}")


if __name__ == "__main__":
    main()
