#!/usr/bin/env python3
"""启智共享盘上的 DATA_CATALOG.json → realbench 数据集记录。

DATA_CATALOG.json 是数据负责人自己维护、并做过校验（视频完整解码、源文件哈希）的清单，
所以这里直接采用它的统计，不重算。清单里没有的信息（采集人、操作员数、采集方案）不编，留空。

批次 batch_id：取 path 里 data/ 之后的第一段（archive/ 下取第二段）。
同一批的不同动作定义、不同子集因此归在一起。
"""
import argparse, datetime, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _admin_core import blocked_state

VERSION = "catalog-importer-0.1"
ROOT = os.path.dirname(HERE)
BASE = "/data/"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def split_path(p):
    rel = p.split(BASE, 1)[1] if BASE in p else p
    parts = [x for x in rel.split("/") if x]
    if parts and parts[0] == "archive":
        batch, rest = parts[1], parts[2:]
        archived = True
    else:
        batch, rest = parts[0], parts[1:]
        archived = False
    return batch, rest, archived


def variant_of(rest):
    keep = [x for x in rest if not x.startswith("lerobot_v3_30fps")]
    tag = "/".join(keep) if keep else ""
    if any(x.startswith("lerobot_v3_30fps_state_t_plus_1") or x.startswith("lerobot_v3_30fps") for x in rest) and not tag:
        tag = "state_t_plus_1"
    return tag or "default"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalog", help="DATA_CATALOG.json")
    ap.add_argument("--robot", default="ARX LIFT2s")
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "datasets"))
    ap.add_argument("--keep-path", action="store_true", help="保留服务器绝对路径（仅本机记录用）")
    a = ap.parse_args()

    cat = json.load(open(a.catalog, encoding="utf-8"))
    when = cat.get("completed_at")
    dp = cat.get("date_policy", "")
    os.makedirs(a.out, exist_ok=True)
    ids = {}
    n = 0
    for d in cat["datasets"]:
        batch, rest, archived = split_path(d["path"])
        var = variant_of(rest)
        did = "ds-" + slug(batch + ("-" + var if var != "default" else ""))
        if did in ids:
            did = did + "-" + slug(d["path"][-20:])
        ids[did] = d["path"]
        if blocked_state(did):
            print("跳过（在删除清单里）:", did)
            continue
        loc = {"where": "inspire", "kind": "converted", "label": "启智共享盘：" + batch + ("/" + var if var != "default" else ""),
               "url": None, "verified": True}
        if a.keep_path:
            loc["path"] = d["path"]
        vm = d.get("video_validation_methods") or {}
        rec = {
            "id": did, "name": batch + (" · " + var if var != "default" else ""), "batch_id": slug(batch),
            "batch_note": "已归档（0906 之前）" if archived else "",
            "derived_from": None, "variant": var, "task": "", "instructions": [], "robot": a.robot,
            "collection_plan_id": None,
            "collected": {"by": None, "date_start": None, "date_end": None,
                          "date_basis": "清单的日期策略：" + dp[:120] if dp else "未记录", "operators": None, "command": None},
            "episodes": d["episodes"], "frames": d["frames"], "fps": d["fps"],
            "episode_frames": {"min": None, "median": None, "max": None},
            "format": "lerobot " + str(d.get("format")),
            "action_semantics": d.get("action_semantics") or "未记录", "cameras": [],
            "validation": {"status": d.get("status") or "unknown", "by": "数据负责人维护的 DATA_CATALOG.json",
                           "video_full_decode": d.get("video_full_decode_evidence") if d.get("video_full_decode_evidence") is not None else d.get("video_full_decode"),
                           "source_episodes_verified": d.get("source_episodes_verified"),
                           "source_episodes_unavailable": d.get("source_episodes_unavailable"),
                           "checked_at": when,
                           "note": "视频文件 %s 个；校验方法 %s；边界抽样 %s 处" % (d.get("video_files"), ",".join(vm) or "未记录", d.get("lerobot_boundary_samples"))},
            "success_labels": {"success": None, "failed": None, "basis": "清单未给成败标签"},
            "locations": [loc],
            "used_by_recipes": [], "notes": "",
            "source": {"kind": "auto-import", "path": "DATA_CATALOG.json（启智共享盘）",
                       "imported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "importer_version": VERSION},
        }
        json.dump(rec, open(os.path.join(a.out, did + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        n += 1
    print(f"写入 {n} 个数据集到 {a.out}；批次数 {len({slug(split_path(d['path'])[0]) for d in cat['datasets']})}")


if __name__ == "__main__":
    main()
