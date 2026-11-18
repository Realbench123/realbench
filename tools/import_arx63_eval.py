#!/usr/bin/env python3
"""63 号机真机运行目录 → realbench 真机评测记录（草稿）。

在 63 上运行，或把运行目录拷到本机后运行。读 args.txt 和 cmd_client_*.npz 里的 args，
同一个 --group 下的多个运行目录合成一条评测，每个目录是一次试验。

不替人判成败：每次试验的 result 一律写 unlabeled，等看视频的人标注。
不替人写预测：prediction 为空，页面会提示“没有跑之前写的预测”。
权重哈希、服务启动时间必须由调用者给，因为 args.txt 里只有端口号。
"""
import argparse, json, os, re, sys, glob, datetime, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import redact

VERSION = "arx63-importer-0.1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read_run(d):
    out = {"dir": os.path.basename(d.rstrip("/")), "args_txt": "", "client_args": {}, "t_start": None,
           "rows": 0, "duration": None, "loop_hz": None, "n_rec_parts": 0, "n_req": 0, "has_images": False, "has_raw": False}
    a = os.path.join(d, "args.txt")
    if os.path.exists(a):
        out["args_txt"] = open(a, encoding="utf-8", errors="ignore").read().strip()
    for f in glob.glob(os.path.join(d, "cmd_client_*.npz")):
        z = np.load(f, allow_pickle=True)
        if "args" in z.files:
            try:
                out["client_args"] = json.loads(str(z["args"]))
            except Exception:
                pass
        if "t_start" in z.files:
            out["t_start"] = float(z["t_start"])
        if "rows" in z.files:
            r = z["rows"]
            out["rows"] = int(r.shape[0])
            if r.shape[0] > 1:
                out["duration"] = float(r[-1, 0] - r[0, 0])
                dt = np.diff(r[:, 0])
                dt = dt[dt > 0]
                out["loop_hz"] = float(1.0 / np.median(dt)) if dt.size else None
        break
    out["n_rec_parts"] = len(glob.glob(os.path.join(d, "rec", "part*.npz")))
    out["n_req"] = len(glob.glob(os.path.join(d, "req", "*.json")))
    out["has_images"] = bool(glob.glob(os.path.join(d, "req", "*.jpg")))
    out["has_raw"] = os.path.isdir(os.path.join(d, "rec")) or os.path.isdir(os.path.join(d, "req"))
    return out


def parse_driver(args_txt):
    m = re.search(r"kd=(\d+)", args_txt)
    return ("kd=" + m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="运行目录（每个是一次试验），按试验顺序")
    ap.add_argument("--id", required=True)
    ap.add_argument("--run-id", required=True, help="对应的训练运行 id，如 tr-...")
    ap.add_argument("--step", type=int, required=True, help="服务加载的 checkpoint 步数")
    ap.add_argument("--weight-sha256", required=True, help="服务实际加载的权重哈希。必填，否则无法核对跑的是哪个权重")
    ap.add_argument("--server-started", default=None, help="服务进程启动时间(ISO)。不知道就不填，记录里留空、页面标缺，不要编一个")
    ap.add_argument("--server-model-dir")
    ap.add_argument("--server-protocol", default=None)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instruction")
    ap.add_argument("--robot", default="arx63")
    ap.add_argument("--date", help="YYYY-MM-DD，默认取第一个运行的开始日期（本地时间）")
    ap.add_argument("--owner", default="")
    ap.add_argument("--protocol-name", default="closed-loop-v0")
    ap.add_argument("--protocol-version", default="draft")
    ap.add_argument("--client-git")
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "evals"))
    a = ap.parse_args()

    runs = [read_run(d) for d in a.dirs]
    trials = []
    for i, r in enumerate(runs, 1):
        trials.append({
            "n": i, "result": "unlabeled", "duration_s": round(r["duration"], 1) if r["duration"] else None,
            "failure_mode": None, "label_by": None, "run_dir": r["dir"], "videos": [],
            "note": (f"原始记录 {r['n_rec_parts']} 个分片，{r['n_req']} 个请求" if r["has_raw"] else "导入时目录里没有 rec/ 和 req/，未统计原始分片"),
        })
    first = runs[0]
    cargs = first["client_args"]
    drivers = {parse_driver(r["args_txt"]) for r in runs}
    # 不同试验的客户端参数必须一致，否则不能合成一条评测
    keys = ["replan", "blend", "time_scale", "rtc_mode", "rtc_prefix", "speed", "seconds", "server_url", "task"]
    diffs = {}
    for k in keys:
        vals = {json.dumps(r["client_args"].get(k)) for r in runs}
        if len(vals) > 1:
            diffs[k] = sorted(vals)
    if len(drivers) > 1:
        diffs["driver"] = sorted(str(x) for x in drivers)
    if diffs:
        print("这些运行的配置不一致，不能合成一条评测，请分开建：", json.dumps(diffs, ensure_ascii=False))
        sys.exit(2)

    hz = [r["loop_hz"] for r in runs if r["loop_hz"]]
    t0 = first["t_start"]
    date = a.date or (datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d") if t0 and t0 > 1e9 else None)
    if not date:
        sys.exit("读不到运行日期（t_start 是单调时钟），请用 --date 指定")

    m = re.search(r":(\d+)", cargs.get("server_url", ""))
    rec = {
        "id": a.id, "type": "closed_loop", "date": date, "owner": a.owner, "run_id": a.run_id,
        "checkpoint": {"step": a.step, "sha256": a.weight_sha256},
        "task": a.task, "instruction": a.instruction or cargs.get("task"),
        "protocol": {"name": a.protocol_name, "version": a.protocol_version},
        "deployment": {
            "robot": a.robot, "driver": drivers.pop(),
            "server": {"host": "127.0.0.1(经转发)", "port": int(m.group(1)) if m else None,
                       "protocol": a.server_protocol, "started": a.server_started, "model_dir": redact(a.server_model_dir)},
            "client": {
                "name": "policy_client_cmd.py", "git_hash": a.client_git,
                "args": {k: cargs.get(k) for k in ("seconds", "replan", "blend", "blend_shape", "speed", "time_scale",
                                                   "rtc_mode", "rtc_prefix", "rtc_weight", "rtc_margin", "gripper_vmax", "vmax")},
                "loop_hz": round(sum(hz) / len(hz), 1) if hz else None,
            },
            "extra": {"loop_hz_per_trial": [round(h, 1) for h in hz]},
        },
        "prediction": {"text": "", "falsified_if": None, "written_at": None, "after_the_fact": True},
        "closed_loop": {"trials": trials, "min_trials_required": 3},
        "conclusion": "",
        "tags": a.tag,
        "source": {"kind": "auto-import", "path": redact(os.path.dirname(os.path.abspath(a.dirs[0]))),
                   "imported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "importer_version": VERSION},
    }
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, a.id + ".json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("写入", p, f"| {len(trials)} 次试验，成败均未标注，没有视频，没有预测")


if __name__ == "__main__":
    main()
