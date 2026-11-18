#!/usr/bin/env python3
"""校验 records/ 下和 .local/ 下的记录是否符合 schema，并做交叉检查（引用是否存在、必填项是否缺）。
退出码非零表示有错误，CI 用它拦截提交。"""
import json, os, re, sys, glob, argparse
from jsonschema import Draft202012Validator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_schema(name):
    return json.load(open(os.path.join(ROOT, "schema", name), encoding="utf-8"))


# records/ 里的内容会公开：内部路径、内网地址、ssh 端口不能进
PRIVATE = re.compile(r"/inspire/|/home/[A-Za-z0-9_.-]+|/Users/|\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b|ssh\s+-p")


def check_dir(d, validator, label, errs, warns, seen, public=False):
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            rec = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            errs.append(f"{f}: 不是合法 JSON：{e}")
            continue
        for e in sorted(validator.iter_errors(rec), key=lambda x: list(x.path)):
            loc = "/".join(str(p) for p in e.path) or "(根)"
            errs.append(f"{os.path.relpath(f, ROOT)}: {loc}: {e.message[:160]}")
        if public:
            m = PRIVATE.search(json.dumps(rec, ensure_ascii=False))
            if m:
                errs.append(f"{os.path.relpath(f, ROOT)}: 含有不能公开的内容“{m.group(0)}”，用导入器默认的路径脱敏，或手工删掉")
        rid = rec.get("id")
        if rid in seen:
            errs.append(f"{os.path.relpath(f, ROOT)}: id {rid} 与 {seen[rid]} 重复")
        seen[rid] = os.path.relpath(f, ROOT)
        if os.path.splitext(os.path.basename(f))[0] != rid:
            errs.append(f"{os.path.relpath(f, ROOT)}: 文件名应与 id 一致（{rid}.json）")
        yield f, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-local", action="store_true", help="同时校验 .local/（本地样本）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    runv = Draft202012Validator(load_schema("training_run.schema.json"))
    evv = Draft202012Validator(load_schema("eval.schema.json"))
    errs, warns = [], []
    runs, evals = {}, {}
    seen_r, seen_e = {}, {}
    dirs = [("records/runs", "records/evals")]
    if a.include_local:
        dirs.append((".local/runs", ".local/evals"))
    for rd, ed in dirs:
        for f, r in check_dir(os.path.join(ROOT, rd), runv, "run", errs, warns, seen_r, public=rd.startswith("records")):
            runs[r["id"]] = r
        for f, e in check_dir(os.path.join(ROOT, ed), evv, "eval", errs, warns, seen_e, public=ed.startswith("records")):
            evals[e["id"]] = e

    # 交叉检查
    for rid, r in runs.items():
        if r["kind"] == r.get("kind_suggested") and not r.get("kind_note"):
            warns.append(f"{rid}: 训练类型 {r['kind']} 与导入器建议相同但没有归类依据（kind_note），请确认")
        if r["kind"] != r.get("kind_suggested") and r.get("kind_suggested") not in (None, "unknown") and not r.get("kind_note"):
            warns.append(f"{rid}: 类型 {r['kind']} 与导入器建议 {r['kind_suggested']} 不同，请在 kind_note 写依据")
        if r["code"].get("dirty"):
            warns.append(f"{rid}: 训练时代码有未提交改动，结果不能靠 commit 复现")
        if r["code"].get("git_hash") is None:
            warns.append(f"{rid}: 没有 git 版本信息")
        if not any(c.get("sha256") for c in r.get("checkpoints", [])):
            warns.append(f"{rid}: 没有任何 checkpoint 的 sha256，真机评测无法核对权重")
        if r["data"]["train_episodes"] == 0:
            warns.append(f"{rid}: 训练集数为 0，可能没读到数据划分")
        sl = [c["step"] for c in r.get("checkpoints", [])]
        if len(sl) != len(set(sl)):
            errs.append(f"{rid}: checkpoint 步数有重复条目，同一步只能有一条")
        if r.get("baseline_id") and r["baseline_id"] not in runs:
            errs.append(f"{rid}: 基线 {r['baseline_id']} 不存在")
        for eid in r.get("evals_offline", []) + r.get("evals_closed", []):
            if eid not in evals:
                errs.append(f"{rid}: 引用了不存在的评测 {eid}")
    for eid, e in evals.items():
        if e["run_id"] not in runs:
            errs.append(f"{eid}: 引用了不存在的训练 {e['run_id']}")
            continue
        run = runs[e["run_id"]]
        steps = {c["step"]: c for c in run.get("checkpoints", [])}
        st = e["checkpoint"]["step"]
        if st not in steps:
            warns.append(f"{eid}: checkpoint 步数 {st} 不在训练 {e['run_id']} 的 checkpoint 列表里")
        else:
            rs, es = steps[st].get("sha256"), e["checkpoint"].get("sha256")
            if rs and es and rs != es:
                errs.append(f"{eid}: 评测加载的权重哈希与训练记录的 checkpoint-{st} 不一致，不是同一个权重")
            if e["type"] == "closed_loop" and not es:
                warns.append(f"{eid}: 真机评测没有填实际加载的权重哈希")
        if e["type"] == "closed_loop":
            cl = e.get("closed_loop") or {}
            n = len(cl.get("trials", []))
            need = cl.get("min_trials_required", 3)
            if n < need:
                warns.append(f"{eid}: 只有 {n} 次试验，少于要求的 {need} 次，不能当结论")
            p = e.get("prediction")
            if not p or not p.get("text"):
                warns.append(f"{eid}: 没有跑之前写的预测")
            elif p.get("after_the_fact"):
                warns.append(f"{eid}: 预测是事后补写的")
            for t in cl.get("trials", []):
                if t["result"] == "unlabeled":
                    warns.append(f"{eid} 第 {t['n']} 次: 成败还没人标")
                if not t.get("videos"):
                    warns.append(f"{eid} 第 {t['n']} 次: 没有视频")
            if not e.get("deployment", {}).get("server", {}).get("started"):
                warns.append(f"{eid}: 没记录服务启动时间，之后分不清端口上是哪个模型")

    if not a.quiet:
        print(f"训练 {len(runs)} 条，评测 {len(evals)} 条")
        for w in warns:
            print("警告:", w)
    for e in errs:
        print("错误:", e)
    if errs:
        print(f"\n共 {len(errs)} 个错误")
        sys.exit(1)
    print("校验通过" + (f"，{len(warns)} 条警告" if warns else ""))


if __name__ == "__main__":
    main()
