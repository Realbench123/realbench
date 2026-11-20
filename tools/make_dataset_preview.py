#!/usr/bin/env python3
"""LeRobot v3 数据集 → 一条预览视频（每个数据集一条，多路相机并排）。

设计：
- 只读数据集目录。只按时间戳截取每个 episode 的一小段，不整段解码，不改源文件。
- episode 的选法是固定规则：按 episode_index 排序后均匀取 K 个（首、尾必含），不挑“好看的”。
  所以预览代表的是数据的典型样子，不是精选。选了哪几集、各自倍速，都写进结果 JSON，页面上照实显示。
- 每个 episode 的片段最长 MAX_S 秒。超长的整集按倍速压进去，倍速写在结果里，不裁尾巴，
  这样能看到整个动作过程。
- 相机顺序固定：head、left_wrist、right_wrist，其余按名字排。并排成一行。
- 视频体积很小（几 MB），输出 H.264 + yuv420p + faststart，浏览器直接播。
- 同一份视频被多个数据集变体共用（硬链接，只是动作定义不同）时，只做一条，其余指向它。
- 不做文字叠加（静态 ffmpeg 没有 drawtext），说明文字由页面显示。

依赖：ffmpeg、ffprobe（--ffmpeg-dir 指定目录）、pyarrow。
用法：
    make_dataset_preview.py --map map.json --out DIR [--ffmpeg-dir DIR] [-k 4] [--max-s 10] [--jobs 4]
map.json: [{"id": "ds-…", "path": "<lerobot 数据集根目录>"}, ...]
"""
import argparse, concurrent.futures as cf, hashlib, json, os, shutil, subprocess, sys, tempfile, time

VERSION = "dataset-preview-0.1"
ORDER = ["head", "left_wrist", "right_wrist"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cam_keys(info):
    ks = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    def rank(k):
        short = k.split(".")[-1]
        return (ORDER.index(short) if short in ORDER else len(ORDER), k)
    return sorted(ks, key=rank)


def pick(n, k):
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def load_episodes(ds, keys):
    import pyarrow.parquet as pq
    cols = ["episode_index", "length"]
    for k in keys:
        cols += [f"videos/{k}/chunk_index", f"videos/{k}/file_index", f"videos/{k}/from_timestamp", f"videos/{k}/to_timestamp"]
    rows = []
    base = os.path.join(ds, "meta", "episodes")
    for root, _, fs in os.walk(base):
        for f in sorted(fs):
            if f.endswith(".parquet"):
                rows += pq.read_table(os.path.join(root, f), columns=cols).to_pylist()
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def video_file(ds, info, key, chunk, file_idx):
    return os.path.join(ds, info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file_idx))


def fingerprint(ds):
    """判断两个数据集的画面是否同一份：总集数相同，并且第一路相机的全部视频文件（inode+大小）逐个相同。
    只比第一个文件会把“包含它的更大数据集”误判成同一份（0907_blue 52 集 与 0907_blue_T 105 集就共用 file-000）。"""
    info = json.load(open(os.path.join(ds, "meta", "info.json"), encoding="utf-8"))
    k = cam_keys(info)[0]
    d = os.path.dirname(video_file(ds, info, k, 0, 0))
    files = []
    for root, _, fs in sorted(os.walk(os.path.join(ds, "videos", k))):
        for f in sorted(fs):
            if f.endswith(".mp4"):
                st = os.stat(os.path.join(root, f))
                files.append(f"{st.st_ino}:{st.st_size}")
    return f"eps={info['total_episodes']}|" + ",".join(files)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("命令失败: " + " ".join(cmd[:3]) + " … " + r.stderr[-400:])
    return r.stdout


def make_one(did, ds, out_dir, ff, k, max_s, height, fps_out, views=None):
    info = json.load(open(os.path.join(ds, "meta", "info.json"), encoding="utf-8"))
    keys = cam_keys(info)
    if views:
        keys = [kk for kk in keys if kk.split(".")[-1] in views]
    if not keys:
        raise RuntimeError("没有符合要求的视频相机")
    eps = load_episodes(ds, keys)
    chosen = [eps[i] for i in pick(len(eps), k)]
    tmp = tempfile.mkdtemp(prefix="pv-", dir=out_dir)
    clips, meta_eps = [], []
    try:
        for n, e in enumerate(chosen):
            ep_dur = e["length"] / info["fps"]
            starts = [e[f"videos/{kk}/from_timestamp"] for kk in keys]
            ends = [e[f"videos/{kk}/to_timestamp"] for kk in keys]
            dur = min(ends[i] - starts[i] for i in range(len(keys)))
            if dur <= 0:
                dur = ep_dur
            speed = max(1.0, dur / max_s)
            cmd = [ff + "/ffmpeg", "-y", "-v", "error", "-threads", "4"]
            for i, kk in enumerate(keys):
                f = video_file(ds, info, kk, e[f"videos/{kk}/chunk_index"], e[f"videos/{kk}/file_index"])
                if not os.path.exists(f):
                    raise RuntimeError("缺视频文件: " + os.path.basename(f))
                cmd += ["-ss", f"{starts[i]:.3f}", "-t", f"{dur:.3f}", "-i", f]
            parts = "".join(f"[{i}:v]setpts=PTS/{speed:.4f},fps={fps_out},scale=-2:{height}[v{i}];" for i in range(len(keys)))
            stack = "".join(f"[v{i}]" for i in range(len(keys))) + (f"hstack=inputs={len(keys)}" if len(keys) > 1 else "null") + ",format=yuv420p[out]"
            clip = os.path.join(tmp, f"c{n}.mp4")
            cmd += ["-filter_complex", parts + stack, "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                    "-r", str(fps_out), "-movflags", "+faststart", clip]
            run(cmd)
            clips.append(clip)
            meta_eps.append({"episode_index": e["episode_index"], "frames": e["length"], "seconds": round(dur, 2), "speed": round(speed, 2)})
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("".join(f"file '{c}'\n" for c in clips))
        out = os.path.join(out_dir, did + ".mp4")
        run([ff + "/ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out])
        pj = json.loads(run([ff + "/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt",
                             "-show_entries", "format=duration,size", "-of", "json", out]))
        s = pj["streams"][0]
        return {"file": os.path.basename(out), "sha256": sha256_file(out), "bytes": int(pj["format"]["size"]), "duration_s": round(float(pj["format"]["duration"]), 2),
                "width": s["width"], "height": s["height"], "codec": s["codec_name"], "pix_fmt": s["pix_fmt"],
                "views": [kk.split(".")[-1] for kk in keys], "episodes_total": len(eps), "episodes": meta_eps,
                "selection": ("按 episode_index 排序后取正中间的一集，不挑选" if len(chosen) == 1 else f"按 episode_index 排序后均匀取 {len(chosen)} 集（含首尾），不挑选"), "max_clip_s": max_s, "tool": VERSION}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg-dir", default=os.path.expanduser("~/.local/ffmpeg"))
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--max-s", type=float, default=10.0)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--views", default=None, help="只用这些相机，逗号分隔，如 head；默认全部")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", help="只做这个 id")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.map, encoding="utf-8"))
    if a.only:
        items = [x for x in items if x["id"] == a.only]
    index_p = os.path.join(a.out, "index.json")
    index = json.load(open(index_p, encoding="utf-8")) if os.path.exists(index_p) else {}

    groups, order = {}, []
    for it in items:
        try:
            fp = fingerprint(it["path"])
        except Exception as e:
            index[it["id"]] = {"error": f"读不到视频: {type(e).__name__}: {str(e)[:120]}"}
            continue
        if fp not in groups:
            groups[fp] = []
            order.append(fp)
        groups[fp].append(it)

    def work(fp):
        first = groups[fp][0]
        if not a.force and index.get(first["id"], {}).get("file") and os.path.exists(os.path.join(a.out, index[first["id"]]["file"])):
            return fp, index[first["id"]], "已有"
        t0 = time.time()
        return fp, make_one(first["id"], first["path"], a.out, a.ffmpeg_dir, a.k, a.max_s, a.height, a.fps, a.views.split(",") if a.views else None), f"{time.time() - t0:.0f}s"

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(work, fp): fp for fp in order}
        for fu in cf.as_completed(futs):
            fp = futs[fu]
            g = groups[fp]
            try:
                _, res, took = fu.result()
            except Exception as e:
                for it in g:
                    index[it["id"]] = {"error": str(e)[:300]}
                print("失败", g[0]["id"], str(e)[:200], file=sys.stderr, flush=True)
                continue
            for it in g:
                r = dict(res)
                r["shared_with"] = [x["id"] for x in g if x["id"] != it["id"]]
                r["owner"] = g[0]["id"]
                index[it["id"]] = r
            print(f"完成 {g[0]['id']}（共用 {len(g)} 个变体）{res['duration_s']}s {res['bytes'] / 1e6:.1f}MB 用时 {took}", flush=True)
            json.dump(index, open(index_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(index, open(index_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    bad = [k for k, v in index.items() if "error" in v]
    print(f"总计 {len(index)} 个数据集，失败 {len(bad)} 个", flush=True)


if __name__ == "__main__":
    main()
