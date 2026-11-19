#!/usr/bin/env python3
"""τ₀ 数据配置目录 → realbench 数据配方记录。

读一个 configs/<name>/ 目录里的 split.json、frame_keep.json、README.md，输出 records/recipes/<id>.json。
只读，不碰数据本体。统计量（锚点数、保留帧数）从 frame_keep.json 重算，并和 README 里写的数字核对，对不上就报出来。

编号：split.json 里的 id 是“转换后数据集”的下标；源 episode 编号 = 下标，若下标 >= 被排除的最小源编号则 +1。
这个换算用 README 里的锚点数反过来验证，验证不过就拒绝写入。
"""
import argparse, hashlib, json, os, re, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import redact

VERSION = "tau0-recipe-importer-0.1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def anchors(ids, to_src, keep, horizon):
    tot = 0
    for i in ids:
        k = keep.get(to_src(i))
        if not k:
            return None
        tot += max(0, (k["keep"][1] - k["keep"][0]) - (horizon - 1))
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cfg_dir", help="含 split.json、frame_keep.json、README.md 的目录")
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--owner", default="")
    ap.add_argument("--instruction")
    ap.add_argument("--source-name", required=True, help="原始数据源名称，如 blue_v2")
    ap.add_argument("--source-kind", default="teleop", choices=["teleop", "scripted", "policy_rollout", "sim", "web_video", "other"])
    ap.add_argument("--source-format", default="hdf5")
    ap.add_argument("--excluded", action="append", default=[], metavar="ID:原因", help="被排除的源 episode，可重复，如 27:README 未写原因")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--action-param", default="relative_to_anchor", choices=["absolute", "relative_to_anchor", "delta", "other"])
    ap.add_argument("--action-param-note", default="")
    ap.add_argument("--cameras", nargs="+", default=["head", "left_wrist", "right_wrist"])
    ap.add_argument("--image-size", nargs=2, type=int, default=[224, 224])
    ap.add_argument("--normalization", default="")
    ap.add_argument("--state-desc", default="")
    ap.add_argument("--action-desc", default="")
    ap.add_argument("--readme-train-anchors", type=int, help="README 里写的训练锚点数，用来验证编号换算")
    ap.add_argument("--readme-val-anchors", type=int)
    ap.add_argument("--parent")
    ap.add_argument("--change", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "recipes"))
    a = ap.parse_args()

    sp = jl(os.path.join(a.cfg_dir, "split.json"))
    fkp = os.path.join(a.cfg_dir, "frame_keep.json")
    fk = jl(fkp)
    keep = {int(re.search(r"episode_(\d+)", k).group(1)): v for k, v in fk["episodes"].items()}
    excl = []
    for e in a.excluded:
        i, _, why = e.partition(":")
        excl.append({"id": int(i), "reason": why or "未写原因"})
    xs = sorted(x["id"] for x in excl)

    def to_src(i):
        # 被排除的源编号会让后面的下标整体后移
        s = i
        for x in xs:
            if s >= x:
                s += 1
        return s

    tr, va = sp["train_ids"], sp["val_ids"]
    horizon = a.horizon
    at, av = anchors(tr, to_src, keep, horizon), anchors(va, to_src, keep, horizon)
    check = []
    if a.readme_train_anchors is not None:
        ok = (at == a.readme_train_anchors and av == a.readme_val_anchors)
        check.append(f"编号换算验证：算得 训练 {at} / 验证 {av}，README 写 {a.readme_train_anchors} / {a.readme_val_anchors}，{'一致' if ok else '不一致'}")
        if not ok:
            sys.exit("\n".join(check) + "\n编号换算与 README 对不上，拒绝写入。检查 --excluded 是否完整。")
    elif at is None:
        sys.exit("编号换算后有 episode 找不到帧保留记录，请补 --excluded 或给 --readme-*-anchors 验证")

    lens = [keep[to_src(i)]["source_frames"] for i in tr + va]
    kept = [keep[to_src(i)]["keep"][1] - keep[to_src(i)]["keep"][0] for i in tr + va]
    det = dict(fk.get("detector", {}))
    val_failed_src = sp.get("failed_validation_source_ids", [])
    # 源编号 → 下标（用来把失败集写成与 val_ids 同一套编号）
    inv = {to_src(i): i for i in tr + va}
    val_failed = [inv[s] for s in val_failed_src if s in inv]

    rec = {
        "id": a.id, "name": a.name, "owner": a.owner, "task": a.task, "instruction": a.instruction,
        "robot": a.robot, "parent_id": a.parent, "changes_from_parent": a.change,
        "sources": [{"name": a.source_name, "kind": a.source_kind, "episodes": len(keep) + len(excl), "weight": None,
                     "collected": None, "operators": None, "location": None, "format": a.source_format,
                     "sha256": None, "url": None, "note": "源 episode 总数 = frame_keep 条目数 + 被排除数"}],
        "selection": {
            "rule": "只用成功示教训练；失败集只放验证" if sp.get("train_success_only") else "未记录",
            "included": len(tr) + len(va), "excluded": excl,
            "success_count": len(tr) + sp.get("validation_success_episodes", 0),
            "failed_count": sp.get("validation_failed_episodes", 0),
        },
        "frames": {
            "fps": a.fps, "horizon": horizon,
            "trim": {"method": "运动窗口检测，保留首个运动窗口前和末个运动窗口后的留白，不裁集内部", "pad_before_s": round(det.get("padding_before_frames", 0) / a.fps, 2) if det.get("padding_before_frames") is not None else None,
                     "pad_after_s": round(det.get("padding_after_frames", 0) / a.fps, 2) if det.get("padding_after_frames") is not None else None,
                     "detector": det, "interior_trim": False},
            "source_frames_total": sum(lens), "kept_frames_total": sum(kept), "anchors_train": at, "anchors_val": av,
            "episode_frames": {"values": lens, "unit": "帧"},
            "keep_file": os.path.basename(fkp), "keep_file_sha256": sha(fkp),
        },
        "split": {
            "unit": sp.get("unit", ""), "seed": sp.get("seed"), "train_ids": tr, "val_ids": va, "val_failed_ids": val_failed,
            "id_space": "转换后数据集下标",
            "id_note": ("源 episode 编号 = 下标；下标 ≥ " + "、".join(str(x) for x in xs) + " 时再 +1（源编号 " + "、".join(str(x) for x in xs) + " 被排除）。" if xs else "与源编号相同。"),
            "train_success_only": bool(sp.get("train_success_only")),
            "file": "split.json", "file_sha256": sha(os.path.join(a.cfg_dir, "split.json")),
        },
        "io": {
            "state": a.state_desc, "action": a.action_desc, "action_semantics": "", "action_param": a.action_param,
            "action_param_note": a.action_param_note, "cameras": a.cameras, "image_size": a.image_size,
            "prompt_template": None, "normalization": a.normalization, "norm_stats_sha256": None,
            "state_dim_padded": None, "action_dim_padded": None,
        },
        "used_by": [],
        "notes": " ".join(check),
        "source": {"kind": "auto-import", "path": redact(os.path.abspath(a.cfg_dir)),
                   "imported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "importer_version": VERSION},
    }
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, a.id + ".json")
    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("写入", p)
    for c in check:
        print(" ", c)
    print(f"  训练 {len(tr)} 集 / 验证 {len(va)} 集（失败 {len(val_failed)}）；保留帧 {sum(kept):,} / 源帧 {sum(lens):,}")


if __name__ == "__main__":
    main()
