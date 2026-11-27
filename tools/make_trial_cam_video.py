#!/usr/bin/env python3
"""把真机运行目录里的 req/*_<camera>.jpg 按请求时间戳拼成 1 倍速视频。

和头部相机那 8 条用同一种拼法（concat 清单，每张图显示到下一张到来，fps=10，高 360，H.264 yuv420p），
这样三路的时长和对应关系一致。只拼指定相机，不改已有的头部视频。

用法: make_trial_cam_video.py --runs-dir DIR --camera left_wrist --out DIR [--ffmpeg /opt/homebrew/bin]
输出: <out>/<run_dir>.<camera>.mp4 和 <out>/index_<camera>.json（哈希、大小、张数、覆盖时长）
"""
import argparse, glob, hashlib, json, os, subprocess, sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--camera", required=True, choices=["head", "left_wrist", "right_wrist"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", default="/opt/homebrew/bin")
    ap.add_argument("--runs", nargs="*", help="只处理这些运行目录名；默认 runs-dir 下全部 cmd_*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = a.runs or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.runs_dir, "cmd_*")))
    index = {}
    for name in runs:
        d = os.path.join(a.runs_dir, name, "req")
        js = sorted(glob.glob(os.path.join(d, "[0-9]*.json")))
        imgs = [j[:-5] + f"_{a.camera}.jpg" for j in js]
        miss = [i for i in imgs if not os.path.exists(i)]
        if not js or miss:
            index[name] = {"error": f"缺图 {len(miss)}/{len(js)}"}
            print("跳过", name, index[name]["error"], file=sys.stderr)
            continue
        ts = [json.load(open(j))["sample_monotonic_ns"] for j in js]
        lst = os.path.join(a.out, f"{name}.{a.camera}.txt")
        with open(lst, "w") as f:
            for i, img in enumerate(imgs):
                dur = (ts[i + 1] - ts[i]) / 1e9 if i + 1 < len(ts) else (ts[i] - ts[i - 1]) / 1e9
                f.write(f"file '{img}'\nduration {dur:.4f}\n")
            f.write(f"file '{imgs[-1]}'\n")
        mp4 = os.path.join(a.out, f"{name}.{a.camera}.mp4")
        subprocess.run([a.ffmpeg + "/ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "fps=10,scale=-2:360,format=yuv420p",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart", mp4], check=True)
        pj = json.loads(subprocess.check_output([a.ffmpeg + "/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt",
                                                 "-show_entries", "format=duration,size", "-of", "json", mp4]))
        s = pj["streams"][0]
        index[name] = {"file": os.path.basename(mp4), "frames_src": len(js), "span_s": round((ts[-1] - ts[0]) / 1e9, 1), "duration_s": round(float(pj["format"]["duration"]), 1),
                       "bytes": int(pj["format"]["size"]), "codec": s["codec_name"], "pix": s["pix_fmt"], "wh": f'{s["width"]}x{s["height"]}', "sha256": sha(mp4)}
        os.remove(lst)
        print(name, a.camera, index[name]["frames_src"], "张 →", index[name]["duration_s"], "s，覆盖真实", index[name]["span_s"], "s，", round(index[name]["bytes"] / 1e6, 2), "MB", flush=True)
    json.dump(index, open(os.path.join(a.out, f"index_{a.camera}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
