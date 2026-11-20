#!/usr/bin/env python3
"""只读扫描共享盘上的训练产物，输出候选清单和曲线。

约束：
- 只读。不写任何扫描范围内的文件，输出只写到 --out。
- 不碰权重：权重文件只取大小，不打开、不哈希。
- 只打开少量小文件：run_spec.json、trainer_state.json、config.yaml、supervisor.log、训练日志头部。
- 只用标准库，能在 CPU 笔记本的系统 python 上直接跑。

输出（--out 目录下）：
  scan.json            最新一次结果
  history/scan-*.json  只有相对上次有变化时才存档
scan.json 字段：scanned_at、candidates（每个训练一条元数据）、curves（降采样曲线）、
changes（相对上次的变化）、errors。
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys
import time

DEFAULT_ROOTS = [
    "/inspire/qb-ilm/project/robot-learning-system/ky26464",
    "/inspire/qb-ilm2/project/robot-learning-system/public/yehangjian",
    "/inspire/qb-ilm2/project/robot-learning-system/public/yifanli",
]
# 不进入的目录：依赖、缓存、数据、代码仓库、本扫描器自己的输出。训练产物不在这些里面。
PRUNE = {
    "opt", ".cache", "__pycache__", ".git", "node_modules", "data", "videos", "meta", "episodes",
    "hf-datasets", "hf-datasets-rtc", "assets", "cache", "tokenizer", "rss2027", "realbench_scan",
    "logs", "site", "experiments", "reference_repos", "wheels", "wheels_pin", "openpi",
}
MAX_DEPTH = 7
SMALL = 40_000_000       # 单个小文件的读取上限
ACTIVE_S = 30 * 60       # 最近这么久有改动，算“运行中”
OPENWAM_CKPT = re.compile(r"^checkpoint_step_(\d+)\.safetensors$")
TQDM = re.compile(r"(\d+)/(\d+) \[([0-9:]+)<[^,\]]*, [^,\]]*, loss=([0-9.]+), video=([0-9.]+), action=([0-9.]+), lr=([0-9.e+-]+)")
LAUNCH = re.compile(r"^launch (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", re.M)


def now_iso(ts=None):
    return datetime.datetime.fromtimestamp(ts or time.time()).astimezone().isoformat(timespec="seconds")


def read(p, limit=SMALL):
    try:
        if os.path.getsize(p) > limit:
            return None
        with open(p, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def jread(p):
    t = read(p, limit=60_000_000)
    if not t:
        return None
    try:
        return json.loads(t)
    except ValueError:
        return None


def mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def fsize(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return None


def to_int(v):
    try:
        return int(str(v).replace("_", ""))
    except (TypeError, ValueError):
        return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def classify(files):
    if "run_spec.json" in files:
        return "tau0"
    if any(OPENWAM_CKPT.match(f) for f in files):
        return "openwam"
    if "trainer_state.json" in files:
        return "hf"
    return None


def sample_idx(n, k=240):
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def curve_from_rows(rows, keys):
    rows = sorted(rows, key=lambda r: r["step"])
    idx = sample_idx(len(rows))
    out = {"step": [rows[i]["step"] for i in idx]}
    for k in keys:
        out[k] = [rows[i].get(k) for i in idx]
    return out


def make_id(kind, d):
    base = re.sub(r"[^a-z0-9]+", "-", os.path.basename(d).lower()).strip("-")[:40] or "run"
    if base.startswith(kind + "-"):
        base = base[len(kind) + 1:] or "run"
    return f"{kind}-{base}-{hashlib.sha1(d.encode()).hexdigest()[:6]}"


def tail_mean(rows, n=10):
    vals = [r["loss"] for r in rows if r.get("loss") is not None][-n:]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def ckpt_dirs(d):
    out = []
    try:
        for name in os.listdir(d):
            m = re.match(r"^checkpoint-(\d+)$", name)
            if m and os.path.isdir(os.path.join(d, name)):
                out.append({"step": int(m.group(1)), "name": name})
    except OSError:
        pass
    return sorted(out, key=lambda x: x["step"])


def decide_status(reached, stopped, last_mtime):
    if reached:
        return "finished"
    if stopped:
        return "stopped"
    if time.time() - last_mtime < ACTIVE_S:
        return "running"
    return "unfinished"


def smoke_of(d, max_steps):
    return ("smoke" in d.lower()) or (max_steps is not None and max_steps <= 50)


def tau0_meta(d, files):
    rs = jread(os.path.join(d, "run_spec.json")) or {}
    ts = jread(os.path.join(d, "trainer_state.json")) or {}
    ma, da, ta = rs.get("model_args", {}), rs.get("data_args", {}), rs.get("training_args", {})
    hist = ts.get("log_history", [])
    rows = [x for x in hist if "loss" in x and "step" in x and "train_loss" not in x]
    summ = [x for x in hist if "train_runtime" in x]
    summary_bad = any(x.get("train_loss") == 0 or (x.get("train_runtime") or 1) < 1 for x in summ)
    world = frames = None
    logs = sorted(glob.glob(os.path.join(d, "log", "*.txt")))
    if logs:
        head = read(logs[0]) or ""
        m = re.search(r"world_size=(\d+)", head)
        world = int(m.group(1)) if m else None
        m = re.search(r"kept ([\d,]+)/([\d,]+) frames", head)
        frames = int(m.group(1).replace(",", "")) if m else None
    gs = ts.get("global_step")
    ms = ts.get("max_steps") or ta.get("max_steps")
    gh = rs.get("git_hash")
    git = None if (not gh or str(gh).startswith("not_a_g")) else gh
    dirty = None if git is None else rs.get("git_dirty")
    per = ta.get("per_device_train_batch_size")
    acc = ta.get("gradient_accumulation_steps") or 1
    final, w = tail_mean(rows)
    last_mtime = max(mtime(os.path.join(d, "trainer_state.json")), mtime(os.path.join(d, "run_spec.json")))
    reached = bool(ms and gs is not None and gs >= ms)
    cand = {
        "kind": "tau0", "dir": d, "base": ma.get("model_name_or_path"), "config_name": da.get("config_name"),
        "rtc": bool(ma.get("training_time_rtc")), "rtc_max_delay": ma.get("rtc_max_delay"),
        "max_steps": ms, "global_step": gs,
        "per_device_batch": per, "grad_accum": acc, "world_size": world,
        "global_batch": per * acc * world if (per and world) else None,
        "learning_rate": ta.get("learning_rate"), "scheduler": ta.get("lr_scheduler_type"),
        "warmup_steps": ta.get("warmup_steps"), "weight_decay": ta.get("weight_decay"), "seed": ta.get("seed"),
        "trainable": [k for k, v in (("vision", "tune_mm_vision"), ("mlp", "tune_mm_mlp"),
                                     ("llm", "tune_mm_llm"), ("dit", "tune_vla_dit")) if ma.get(v)],
        "train_anchors": frames, "git_hash": git, "dirty": dirty,
        "start": rs.get("start_time"), "command": (rs.get("command") or "")[:300],
        "notes": str(rs.get("notes") or "")[:300],
        "loss_first": rows[0]["loss"] if rows else None, "loss_final": final, "loss_final_window": w,
        "summary_bad": summary_bad, "checkpoints": ckpt_dirs(d),
        "status": decide_status(reached, False, last_mtime), "smoke": smoke_of(d, ms),
    }
    curve = curve_from_rows(rows, ["loss", "grad_norm", "learning_rate", "step_time_ms"]) if rows else None
    return cand, curve


def to_utc_ts(s):
    return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp()


def segment_for(log, run_name):
    """supervisor.log 里可能有多次启动。取“在这个运行目录创建之前（+60 秒）最近一次启动”到下一次启动之间的片段。"""
    launches = [(m.start(), m.group(1)) for m in LAUNCH.finditer(log)]
    if not launches:
        return log
    try:
        t_dir = datetime.datetime.strptime(run_name, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return log
    chosen = None
    for pos, ts in launches:
        if to_utc_ts(ts) <= t_dir + 60:
            chosen = pos
    if chosen is None:
        return log
    later = [p for p, _ in launches if p > chosen]
    return log[chosen: later[0] if later else len(log)]


def openwam_meta(d, files):
    cfg = read(os.path.join(d, "config.yaml")) or ""

    def kv(k):
        m = re.search(rf"^\s*{k}:\s*(.+?)\s*$", cfg, re.M)
        return m.group(1).strip("'\"") if m else None

    ms = to_int(kv("max_steps"))
    ckpts = sorted((int(OPENWAM_CKPT.match(f).group(1)), f) for f in files if OPENWAM_CKPT.match(f))
    sup_path = os.path.join(os.path.dirname(d), "supervisor.log")
    sup = read(sup_path, limit=200_000_000) or ""
    seg = segment_for(sup, os.path.basename(d))
    pts = {}
    for m in TQDM.finditer(seg):
        pts[int(m.group(1))] = {"step": int(m.group(1)), "loss": float(m.group(4)), "video": float(m.group(5)),
                                "action": float(m.group(6)), "lr": float(m.group(7)), "elapsed": m.group(3)}
    rows = [pts[k] for k in sorted(pts)]
    last = rows[-1]["step"] if rows else None
    final, w = tail_mean(rows)
    reached = bool(ms and last == ms)
    stopped = ("SignalException" in seg) and not reached
    last_mtime = max(mtime(sup_path), max((mtime(os.path.join(d, f)) for _, f in ckpts), default=0.0))
    cand = {
        "kind": "openwam", "dir": d, "base": kv("finetune_ckpt_path"), "dataset": kv("dataset_dir"),
        "max_steps": ms, "global_step": last, "save_steps": to_int(kv("save_steps")),
        "learning_rate": to_float(kv("learning_rate")), "per_device_batch": to_int(kv("batch_size")),
        "grad_accum": to_int(kv("gradient_accumulation_steps")), "zero_stage": to_int(kv("zero_stage")),
        "scheduler": kv("lr_scheduler"), "warmup_ratio": to_float(kv("warmup_ratio")),
        "checkpoints": [{"step": s, "name": f, "bytes": fsize(os.path.join(d, f))} for s, f in ckpts],
        "log_path": sup_path, "log_points": len(rows), "elapsed_last": rows[-1]["elapsed"] if rows else None,
        "loss_first": rows[0]["loss"] if rows else None, "loss_final": final, "loss_final_window": w,
        "git_hash": None, "dirty": None,
        "status": decide_status(reached, stopped, last_mtime), "smoke": smoke_of(d, ms),
    }
    curve = curve_from_rows(rows, ["loss", "video", "action", "lr"]) if rows else None
    if rows:
        tail = rows[-300:]   # 末段全量，不降采样：末值要用它算，降采样会把噪声放大
        cand["tail_loss"] = {"step": [r["step"] for r in tail], "loss": [r["loss"] for r in tail]}
    return cand, curve


def hf_meta(d, files):
    ts = jread(os.path.join(d, "trainer_state.json")) or {}
    rows = [x for x in ts.get("log_history", []) if "loss" in x and "step" in x and "train_loss" not in x]
    gs, ms = ts.get("global_step"), ts.get("max_steps")
    final, w = tail_mean(rows)
    reached = bool(ms and gs is not None and gs >= ms)
    last_mtime = mtime(os.path.join(d, "trainer_state.json"))
    cand = {
        "kind": "hf", "dir": d, "base": None, "max_steps": ms, "global_step": gs,
        "loss_first": rows[0]["loss"] if rows else None, "loss_final": final, "loss_final_window": w,
        "checkpoints": ckpt_dirs(d), "git_hash": None, "dirty": None,
        "status": decide_status(reached, False, last_mtime), "smoke": smoke_of(d, ms),
    }
    curve = curve_from_rows(rows, ["loss", "grad_norm", "learning_rate"]) if rows else None
    return cand, curve


def describe(kind, d, files):
    if kind == "tau0":
        return tau0_meta(d, files)
    if kind == "openwam":
        return openwam_meta(d, files)
    return hf_meta(d, files)


def changes(prev, cur):
    pmap = {c["id"]: c for c in (prev or {}).get("candidates", [])}
    out = []
    for c in cur:
        p = pmap.get(c["id"])
        if p is None:
            out.append({"id": c["id"], "change": "new"})
            continue
        if p.get("status") != c.get("status"):
            out.append({"id": c["id"], "change": f"status {p.get('status')}->{c.get('status')}"})
        if p.get("global_step") != c.get("global_step"):
            out.append({"id": c["id"], "change": f"step {p.get('global_step')}->{c.get('global_step')}"})
    gone = set(pmap) - {c["id"] for c in cur}
    out += [{"id": g, "change": "gone"} for g in sorted(gone)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="输出目录（不在扫描时被读取）")
    ap.add_argument("--root", action="append", help="扫描根目录，可重复；默认三个目录")
    ap.add_argument("--ignore-file", help="JSON 列表，每项是目录路径的结尾（如 openwam-arx-0909/2026-11-04_07-47-47），命中则标 ignored")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    roots = [os.path.abspath(r) for r in (a.root or DEFAULT_ROOTS)]
    t0 = time.time()
    errors, found = [], []

    for root in roots:
        label = os.path.basename(root.rstrip("/"))
        if not os.path.isdir(root):
            errors.append({"root": root, "error": "不存在或不可读"})
            continue
        base = root.rstrip("/").count("/")
        n_dirs = 0

        def onerr(e):
            errors.append({"path": str(getattr(e, "filename", "")), "error": str(getattr(e, "strerror", e))[:80]})

        for dp, dns, fns in os.walk(root, onerror=onerr):
            n_dirs += 1
            depth = dp.rstrip("/").count("/") - base
            kind = classify(set(fns))
            if kind:
                found.append((label, dp, kind, set(fns)))
                dns[:] = []
                continue
            if depth + 1 > MAX_DEPTH:
                dns[:] = []
                continue
            dns[:] = [x for x in dns if x not in PRUNE and not x.startswith("checkpoint-") and not x.startswith(".")]
        print(f"[{label}] 目录 {n_dirs}，累计候选 {len(found)}，用时 {time.time() - t0:.0f}s", file=sys.stderr)

    ignore = []
    if a.ignore_file and os.path.exists(a.ignore_file):
        ignore = [x.strip("/") for x in json.load(open(a.ignore_file, encoding="utf-8")) if x]
    cands, curves = [], {}
    for label, dp, kind, fns in found:
        try:
            cand, curve = describe(kind, dp, fns)
        except Exception as e:  # 单个运行解析失败不影响其它
            errors.append({"path": dp, "error": f"解析失败: {type(e).__name__}: {str(e)[:120]}"})
            continue
        cand["id"] = make_id(kind, dp)
        cand["root"] = label
        cand["ignored"] = any(dp.rstrip("/").endswith("/" + x) for x in ignore)
        cands.append(cand)
        if curve:
            curves[cand["id"]] = curve

    p = os.path.join(out_dir, "scan.json")
    prev = None
    if os.path.exists(p):
        try:
            prev = json.load(open(p, encoding="utf-8"))
        except ValueError:
            prev = None
    ch = changes(prev, cands)
    ts = now_iso()
    result = {"scanned_at": ts, "elapsed_s": round(time.time() - t0, 1), "roots": roots,
              "candidates": cands, "curves": curves, "changes": ch, "errors": errors[:300]}
    body = json.dumps(result, ensure_ascii=False, indent=1)
    if ch:
        hd = os.path.join(out_dir, "history")
        os.makedirs(hd, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        with open(os.path.join(hd, f"scan-{stamp}.json"), "w", encoding="utf-8") as f:
            f.write(body)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
    os.replace(tmp, p)
    print(f"候选 {len(cands)} 条，曲线 {len(curves)} 条，变化 {len(ch)} 条，错误 {len(errors)} 条，用时 {result['elapsed_s']}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
