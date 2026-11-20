#!/usr/bin/env python3
"""arx_collect_v2 的 hdf5 数据集（每集一个 episode_N.hdf5）→ 一条预览视频。

和 make_dataset_preview.py 的取法一致：只用头部相机，按集序号排序后取正中间的一集，不挑选。
hdf5 里每帧图像是一行变长 JPEG（observations/images/head），直接把 JPEG 字节流喂给 ffmpeg 的 image2pipe，
不依赖 opencv。只读数据集，不改源文件。

只取 meta.active_range 之内的帧（采集时遥操作真正生效的区间），前后的待机帧不放，
这和转成 lerobot 之后保留的帧一致。active_range 缺失就用整集。
输出的 index.json 和 make_dataset_preview.py 同格式，可以直接交给 attach_dataset_preview.py。

依赖：ffmpeg、ffprobe（--ffmpeg-dir）、h5py。
用法：make_hdf5_preview.py --map map.json --out DIR [--ffmpeg-dir DIR]
map.json: [{"id": "ds-…", "path": "<含 episode_*.hdf5 的目录>"}]
"""
import argparse, hashlib, json, os, re, subprocess, sys

VERSION = "hdf5-preview-0.1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def episodes_of(d):
    xs = []
    for f in os.listdir(d):
        m = re.fullmatch(r"episode_(\d+)\.hdf5", f)
        if m:
            xs.append((int(m.group(1)), f))
    return sorted(xs)


def make_one(did, d, out_dir, ff, max_s, height, view):
    import h5py
    eps = episodes_of(d)
    if not eps:
        raise RuntimeError("目录里没有 episode_*.hdf5")
    idx, fname = eps[len(eps) // 2]
    with h5py.File(os.path.join(d, fname), "r") as h:
        a = h.attrs
        fps = int(a.get("frame_rate", 30))
        meta = json.loads(a["meta"]) if "meta" in a else {}
        rows = h[f"observations/images/{view}"]
        n = rows.shape[0]
        lo, hi = (meta.get("active_range") or [0, n])
        lo, hi = max(0, int(lo)), min(n, int(hi))
        if hi - lo < 10:
            lo, hi = 0, n
        succ = a.get("success")
        succ = bool(succ) if succ is not None else meta.get("success")
        dur = (hi - lo) / fps
        speed = max(1.0, dur / max_s)
        out = os.path.join(out_dir, did + ".mp4")
        cmd = [ff + "/ffmpeg", "-y", "-v", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps), "-i", "-",
               "-vf", f"setpts=PTS/{speed:.4f},fps={fps},scale=-2:{height},format=yuv420p",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-r", str(fps), "-movflags", "+faststart", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(lo, hi):
                p.stdin.write(rows[i].tobytes())
            p.stdin.close()
        except BrokenPipeError:
            pass
        err = p.stderr.read().decode(errors="ignore")
        if p.wait() != 0:
            raise RuntimeError("ffmpeg 失败: " + err[-300:])
        instr = a.get("instruction")
    pj = json.loads(subprocess.check_output([ff + "/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=codec_name,width,height,pix_fmt", "-show_entries", "format=duration,size", "-of", "json", out]))
    s = pj["streams"][0]
    flag = {True: "采集时标为成功", False: "采集时标为失败", None: "采集时没有成败标记"}[succ]
    return {"file": os.path.basename(out), "sha256": sha256_file(out), "bytes": int(pj["format"]["size"]),
            "duration_s": round(float(pj["format"]["duration"]), 2), "width": s["width"], "height": s["height"],
            "codec": s["codec_name"], "pix_fmt": s["pix_fmt"], "views": [view], "episodes_total": len(eps),
            "episodes": [{"episode_index": idx, "frames": hi - lo, "seconds": round(dur, 2), "speed": round(speed, 2)}],
            "selection": f"按 episode_index 排序后取正中间的一集，不挑选；只取遥操作生效区间（第 {lo} 到 {hi} 帧）；这一集{flag}",
            "max_clip_s": max_s, "shared_with": [], "owner": did, "tool": VERSION, "instruction": str(instr) if instr else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg-dir", default=os.path.expanduser("~/.local/ffmpeg"))
    ap.add_argument("--max-s", type=float, default=12.0)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--view", default="head")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = {}
    for it in json.load(open(a.map, encoding="utf-8")):
        try:
            index[it["id"]] = make_one(it["id"], it["path"], a.out, a.ffmpeg_dir, a.max_s, a.height, a.view)
            v = index[it["id"]]
            print(f"完成 {it['id']}：第 {v['episodes'][0]['episode_index']}/{v['episodes_total']} 集 {v['duration_s']}s {v['bytes'] / 1e3:.0f}KB；{v['selection']}", flush=True)
        except Exception as e:
            index[it["id"]] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print("失败", it["id"], index[it["id"]]["error"], file=sys.stderr, flush=True)
    json.dump(index, open(os.path.join(a.out, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"总计 {len(index)} 个，失败 {sum('error' in v for v in index.values())} 个")


if __name__ == "__main__":
    main()
