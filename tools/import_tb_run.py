#!/usr/bin/env python3
"""TensorBoard 事件文件 + 手工核对过的元数据 → realbench 训练记录和曲线。

适用于没有 run_spec.json / trainer_state.json 的训练栈（LingBot、LeRobot π0.5）。
曲线来自事件文件；其余字段由 --meta JSON 给出，调用者负责只填读到的内容。
meta 里没有的字段一律写成空值或"未记录"，不推断。

末值口径：取最后 N 个记录点的均值（默认 10，和其它导入器一致），并把最后一个单点另存在 notes。
曲线最多降采样到 240 点；末值用的是未降采样的全量。
用法：import_tb_run.py --events FILE --scalar train/loss --meta meta.json --out-dir DIR
"""
import argparse, datetime, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_scalars as tb
from _admin_core import blocked_state

VERSION = "tb-importer-0.1"
ROOT = os.path.dirname(HERE)


def downsample(step, val, k=240):
    n = len(step)
    if n <= k:
        return step, val
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [step[i] for i in idx], [val[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--scalar", required=True, help="总 loss 的标量名，如 train/loss、training/loss、train_rank0/loss")
    ap.add_argument("--lr-scalar")
    ap.add_argument("--grad-scalar")
    ap.add_argument("--meta", required=True, help="元数据 JSON")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "records"))
    ap.add_argument("--curve-dir", default=os.path.join(ROOT, "curves"))
    ap.add_argument("--window", type=int, default=10)
    a = ap.parse_args()

    m = json.load(open(a.meta, encoding="utf-8"))
    if blocked_state(m["id"]):
        sys.exit(f"{m['id']} 在删除清单里（{blocked_state(m['id'])}），不导入。")
    want = {a.scalar} | ({a.lr_scalar} if a.lr_scalar else set()) | ({a.grad_scalar} if a.grad_scalar else set())
    ser = {}
    for tag, s, x in tb.scalars(a.events, want):
        o = ser.setdefault(tag, {})
        o[s] = x       # 同一步重复写入时取后写的
    if a.scalar not in ser:
        sys.exit(f"事件文件里没有 {a.scalar}")
    steps = sorted(ser[a.scalar])
    loss = [ser[a.scalar][s] for s in steps]
    last_step = steps[-1]
    if last_step != m["hparams"]["max_steps"]:
        print(f"注意：最后一个记录点是第 {last_step} 步，计划 {m['hparams']['max_steps']} 步", file=sys.stderr)
    w = min(a.window, len(loss))
    final = statistics.mean(loss[-w:])

    def aligned(tag):
        if not tag or tag not in ser:
            return [None] * len(steps)
        return [ser[tag].get(s) for s in steps]

    cs, cl = downsample(steps, loss)
    pick = lambda arr: [arr[steps.index(s)] for s in cs]
    curve = {"step": cs, "loss": cl, "grad_norm": pick(aligned(a.grad_scalar)),
             "learning_rate": pick(aligned(a.lr_scalar)), "step_time_ms": [None] * len(cs)}
    src_note = f"[导入器] 曲线来自 TensorBoard 事件文件的 {a.scalar}（{len(loss)} 个记录点，步 {steps[0]}–{last_step}）；loss 末值取最后 {w} 个记录点的均值 {final:.4f}，最后一个单点 {loss[-1]:.4f}。"
    rec = {
        "id": m["id"], "name": m["name"], "owner": m.get("owner", ""), "family": m["family"],
        "kind": m["kind"], "kind_note": m.get("kind_note", ""), "kind_suggested": m.get("kind_suggested", m["kind"]),
        "baseline_id": m.get("baseline_id"), "recipe_id": m.get("recipe_id"), "stack_id": m.get("stack_id"),
        "base": m["base"], "data": m["data"], "hparams": m["hparams"], "hardware": m["hardware"], "code": m["code"],
        "time": m.get("time", {"start": None, "end": None, "status_reason": None}),
        "status": m.get("status", "finished"),
        "loss": {"first": loss[0], "final": final, "final_window": w, "curve_file": f"curves/{m['id']}.json", "summary_bad": False},
        "checkpoints": m.get("checkpoints", []), "deployed_as": [], "evals_offline": [], "evals_closed": [],
        "notes": ((m.get("notes") + " ") if m.get("notes") else "") + src_note,
        "source": {"kind": "auto-import", "path": m.get("source_path", ""),
                   "imported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "importer_version": VERSION},
    }
    for d in ("runs",):
        os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)
    os.makedirs(a.curve_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "runs", m["id"] + ".json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(curve, open(os.path.join(a.curve_dir, m["id"] + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"写入 {p} | {len(loss)} 点 首 {loss[0]:.4f} 末均 {final:.4f} 末单点 {loss[-1]:.4f}")


if __name__ == "__main__":
    main()
