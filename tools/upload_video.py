#!/usr/bin/env python3
"""把一次真机试验的视频转成网页能直接播的 H.264，上传到 Hugging Face 数据集，并把链接写进评测记录。

默认只做 dry-run：转码到本地、算哈希、打印将要上传的路径和将要写入记录的链接，不上传，不改记录。
加 --apply 才真的上传、写记录。上传前再核对一遍：数据集是公开的，视频里不能出现不想公开的人和场景。

路径约定（和 RoboDojo 一致，便于以后对照）：
    <family>/<run_id>/<eval_id>/trial_<N>.<camera>.mp4
"""
import argparse, hashlib, json, os, subprocess, sys, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def transcode(src, dst, crf=26, max_h=480):
    """H.264 + yuv420p + faststart：所有浏览器都能播，进度条立刻可拖。去掉音轨。"""
    vf = f"scale=-2:'min({max_h},ih)',format=yuv420p"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-an", "-vf", vf, "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True)


def probe(p):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=codec_name,width,height,pix_fmt", "-show_entries", "format=duration,size",
                                   "-of", "json", p])
    j = json.loads(out)
    s = j["streams"][0]
    return {"codec": s["codec_name"], "w": s["width"], "h": s["height"], "pix": s["pix_fmt"],
            "dur": float(j["format"]["duration"]), "mb": int(j["format"]["size"]) / 1e6}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_json", help="评测记录（records/evals/ev-*.json）")
    ap.add_argument("--trial", type=int, required=True, help="第几次试验")
    ap.add_argument("--camera", required=True, help="如 head、left_wrist")
    ap.add_argument("--video", required=True, help="原始视频文件")
    ap.add_argument("--repo", required=True, help="HF 数据集仓库，如 <org>/realbench-videos")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--apply", action="store_true", help="真的上传并写记录（默认只 dry-run）")
    a = ap.parse_args()

    ev = json.load(open(a.eval_json, encoding="utf-8"))
    run = ev["run_id"]
    rp = None
    for d in ("records/runs", ".local/runs"):
        f = os.path.join(ROOT, d, run + ".json")
        if os.path.exists(f):
            rp = json.load(open(f, encoding="utf-8"))
            break
    fam = (rp or {}).get("family") or "misc"    # 族名来自训练记录，不从 id 里拆
    rel = f"{fam}/{run}/{ev['id']}/trial_{a.trial}.{a.camera}.mp4"
    trial = next((t for t in ev["closed_loop"]["trials"] if t["n"] == a.trial), None)
    if not trial:
        sys.exit(f"评测 {ev['id']} 里没有第 {a.trial} 次试验")

    if not shutil.which("ffmpeg"):
        sys.exit("没有 ffmpeg")
    tmp = tempfile.mkdtemp(prefix="realbench-")
    out = os.path.join(tmp, os.path.basename(rel))
    transcode(a.video, out, crf=a.crf)
    info = probe(out)
    ok = info["codec"] == "h264" and info["pix"] == "yuv420p"
    digest = sha256_file(out)
    url = f"https://huggingface.co/datasets/{a.repo}/resolve/main/{rel}"
    print(f"转码完成：{info['codec']} {info['w']}x{info['h']} {info['pix']} {info['dur']:.1f}s {info['mb']:.2f}MB  {'可直接在浏览器播放' if ok else '编码不对，浏览器可能播不了'}")
    print("仓库路径:", rel)
    print("将写入的链接:", url)
    print("sha256:", digest)
    if not a.apply:
        print("\n这是 dry-run，没有上传，也没有改记录。确认视频可以公开后，加 --apply。")
        return
    if not ok:
        sys.exit("编码不是 h264/yuv420p，拒绝上传")
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print("以", who["name"], "身份上传")
    api.upload_file(path_or_fileobj=out, path_in_repo=rel, repo_id=a.repo, repo_type="dataset",
                    commit_message=f"{ev['id']} trial {a.trial} {a.camera}")
    vids = trial.setdefault("videos", [])
    vids[:] = [v for v in vids if v["camera"] != a.camera]
    vids.append({"camera": a.camera, "url": url, "sha256": digest})
    json.dump(ev, open(a.eval_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已上传并写入记录。")


if __name__ == "__main__":
    main()
