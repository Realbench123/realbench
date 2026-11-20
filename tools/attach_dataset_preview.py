#!/usr/bin/env python3
"""把 make_dataset_preview.py 生成的预览（index.json + mp4）挂到数据集记录的 preview 字段。

- 视频复制到 .local/previews/<ds-id>.mp4（不进仓库，admin_server 在本机提供播放）。
- 共用同一份素材的变体指向同一份内容，各自有自己的文件名，方便按 id 取。
- 复制后重算 sha256 并和 index.json 比对，不一致就拒绝。
- 只改 preview 字段，其余内容原样保留。--url-base 给定时写入公开地址（上传到 HF 之后用）。
"""
import argparse, hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="含 index.json 和 mp4 的目录")
    ap.add_argument("--records", default=os.path.join(ROOT, ".local", "datasets"))
    ap.add_argument("--previews", default=os.path.join(ROOT, ".local", "previews"))
    ap.add_argument("--url-base", default=None, help="公开地址前缀，如 https://huggingface.co/datasets/ORG/REPO/resolve/main/dataset-previews/")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.src, "index.json"), encoding="utf-8"))
    os.makedirs(a.previews, exist_ok=True)
    ok = bad = 0
    for did, v in sorted(idx.items()):
        rp = os.path.join(a.records, did + ".json")
        if not os.path.exists(rp):
            print("跳过（没有这条记录）:", did)
            continue
        if "error" in v:
            print("跳过（生成失败）:", did, v["error"][:100])
            bad += 1
            continue
        src = os.path.join(a.src, v["file"])
        dst = os.path.join(a.previews, did + ".mp4")
        shutil.copyfile(src, dst)
        if sha(dst) != v["sha256"]:
            os.remove(dst)
            sys.exit(f"{did}: 复制后哈希和 index.json 不一致，已删除副本")
        rec = json.load(open(rp, encoding="utf-8"))
        pv = {k: v[k] for k in ("sha256", "bytes", "duration_s", "width", "height", "codec", "pix_fmt", "views", "episodes_total",
                                 "episodes", "selection", "max_clip_s", "shared_with", "owner", "tool") if k in v}
        pv["file"] = did + ".mp4"
        pv["url"] = (a.url_base.rstrip("/") + "/" + did + ".mp4") if a.url_base else None
        pv["episodes"] = [{k: e[k] for k in ("episode_index", "frames", "seconds", "speed") if k in e} for e in v["episodes"]]
        rec["preview"] = pv
        json.dump(rec, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        ok += 1
    print(f"挂接 {ok} 个预览，失败/跳过 {bad} 个；视频在 {a.previews}")


if __name__ == "__main__":
    main()
