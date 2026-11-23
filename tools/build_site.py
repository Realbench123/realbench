#!/usr/bin/env python3
"""records/ 与 curves/ → site/dist/index.html（自包含单页，数据内嵌）。
默认只读 records/；--include-local 同时读 .local/ 里的样本（本地预览用，页面会标出“样本”）。"""
import argparse, json, os, glob, re, datetime, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_all(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        out.append(json.load(open(f, encoding="utf-8")))
    return out


def downsample(curve, n=240):
    L = len(curve["step"])
    if L <= n:
        return curve
    idx = sorted({round(i * (L - 1) / (n - 1)) for i in range(n)})
    return {k: [v[i] for i in idx] for k, v in curve.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-local", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "site", "dist"))
    a = ap.parse_args()

    runs, evals, curves, recipes = [], [], {}, []
    extra = {"datasets": [], "plans": [], "stacks": [], "groups": [], "vendor_tests": []}
    srcs = [(os.path.join(ROOT, "records", "runs"), os.path.join(ROOT, "records", "evals"), os.path.join(ROOT, "curves"), False, os.path.join(ROOT, "records", "recipes"))]
    if a.include_local:
        srcs.append((os.path.join(ROOT, ".local", "runs"), os.path.join(ROOT, ".local", "evals"), os.path.join(ROOT, ".local", "curves"), True, os.path.join(ROOT, ".local", "recipes")))
    for rd, ed, cd, is_local, rcd in srcs:
        for r in load_all(rd):
            r["_sample"] = r["id"].startswith("tr-sample"); r["_pending"] = is_local
            runs.append(r)
            cf = os.path.join(cd, r["id"] + ".json")
            if os.path.exists(cf):
                curves[r["id"]] = downsample(json.load(open(cf)))
        for e in load_all(ed):
            e["_sample"] = e["id"].startswith("ev-sample"); e["_pending"] = is_local
            evals.append(e)
        for c in load_all(rcd):
            c["_sample"] = c["id"].startswith("rc-sample"); c["_pending"] = is_local
            recipes.append(c)
        base = os.path.dirname(rd)
        for k in extra:
            for x in load_all(os.path.join(base, k)):
                x["_pending"] = is_local
                extra[k].append(x)
    for c in recipes:
        c["used_by"] = sorted(r["id"] for r in runs if r.get("recipe_id") == c["id"])   # 由训练记录反推，不信手填
    for d in extra["datasets"]:
        d["used_by_recipes"] = sorted(c["id"] for c in recipes if d["id"] in (c.get("dataset_ids") or []))
    for p in extra["plans"]:
        p["used_by_datasets"] = sorted(d["id"] for d in extra["datasets"] if d.get("collection_plan_id") == p["id"])
    for s in extra["stacks"]:
        s["used_by_runs"] = sorted(r["id"] for r in runs if r.get("stack_id") == s["id"])

    try:
        ver = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        ver = "无提交"
    data = {"runs": runs, "evals": evals, "recipes": recipes, "curves": curves, **extra,
            "built": datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"), "commit": ver}
    tpl = open(os.path.join(ROOT, "site", "src", "template.html"), encoding="utf-8").read()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    js = open(os.path.join(ROOT, "site", "src", "app.js"), encoding="utf-8").read().replace("</script", "<\\/script")
    page = tpl.replace("/*__DATA__*/null", payload).replace("/*@@JS@@*/", js)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "index.html")
    open(out, "w", encoding="utf-8").write(page)
    print("生成", out, "%.1f KB" % (len(page.encode("utf-8")) / 1024), f"| 配方 {len(recipes)} 训练 {len(runs)} 评测 {len(evals)}")


if __name__ == "__main__":
    main()
