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
    rcv = Draft202012Validator(load_schema("recipe.schema.json"))
    evv = Draft202012Validator(load_schema("eval.schema.json"))
    errs, warns = [], []
    runs, evals, recipes = {}, {}, {}
    seen_r, seen_e, seen_c = {}, {}, {}
    dirs = [("records/runs", "records/evals", "records/recipes")]
    if a.include_local:
        dirs.append((".local/runs", ".local/evals", ".local/recipes"))
    for rd, ed, cd in dirs:
        for f, r in check_dir(os.path.join(ROOT, rd), runv, "run", errs, warns, seen_r, public=rd.startswith("records")):
            runs[r["id"]] = r
        for f, e in check_dir(os.path.join(ROOT, ed), evv, "eval", errs, warns, seen_e, public=ed.startswith("records")):
            evals[e["id"]] = e
        for f, c in check_dir(os.path.join(ROOT, cd), rcv, "recipe", errs, warns, seen_c, public=cd.startswith("records")):
            recipes[c["id"]] = c

    # 删除清单：已删除或不要导入的 id 不能出现在 records/ 里
    sys.path.insert(0, HERE)
    from _admin_core import tombstones
    gone = {e["id"]: e["state"] for e in tombstones()["entries"]}
    for table in (runs, evals, recipes):
        for rid in table:
            if rid in gone and not rid.endswith("-sample") and "sample" not in rid:
                errs.append(f"{rid}: 在 records/_deleted.json 里标为 {gone[rid]}，不应再出现在记录里")
    # 配方内部一致性
    for cid, c in recipes.items():
        sp = c["split"]
        tr, va = sp["train_ids"], sp["val_ids"]
        if set(tr) & set(va):
            errs.append(f"{cid}: 训练集和验证集有交集 {sorted(set(tr) & set(va))[:5]}")
        if len(tr) + len(va) != c["selection"]["included"]:
            errs.append(f"{cid}: 训练 {len(tr)} + 验证 {len(va)} ≠ selection.included {c['selection']['included']}")
        if not set(sp.get("val_failed_ids", [])) <= set(va):
            errs.append(f"{cid}: val_failed_ids 不全在验证集里")
        if sp.get("train_success_only") and set(sp.get("val_failed_ids", [])) & set(tr):
            errs.append(f"{cid}: 声明只用成功示教训练，但失败集出现在训练集里")
        fr = c["frames"]
        if fr.get("kept_frames_total") and fr.get("source_frames_total") and fr["kept_frames_total"] > fr["source_frames_total"]:
            errs.append(f"{cid}: 保留帧数大于源帧数")
        if len(c["sources"]) > 1 and any(s.get("weight") is None for s in c["sources"]):
            warns.append(f"{cid}: 有多个数据源但部分没有采样权重，配比不完整")
        if not c["io"].get("normalization"):
            warns.append(f"{cid}: 没写归一化方式")
        if c.get("parent_id") and c["parent_id"] not in recipes:
            errs.append(f"{cid}: parent {c['parent_id']} 不存在")
        if c.get("parent_id") and not c.get("changes_from_parent"):
            warns.append(f"{cid}: 有 parent 但没写改了什么")

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
        rc = recipes.get(r.get("recipe_id")) if r.get("recipe_id") else None
        if r.get("recipe_id") and not rc:
            errs.append(f"{rid}: 引用了不存在的配方 {r['recipe_id']}")
        elif not r.get("recipe_id"):
            warns.append(f"{rid}: 没有引用数据配方，数据部分只能看到训练日志里的粗略数字")
        elif rc:
            if len(rc["split"]["train_ids"]) != r["data"]["train_episodes"]:
                errs.append(f"{rid}: 训练集数 {r['data']['train_episodes']} 与配方 {rc['id']} 的 {len(rc['split']['train_ids'])} 不一致")
            ta, la = rc["frames"].get("anchors_train"), r["data"].get("train_anchors")
            if ta and la and ta != la:
                errs.append(f"{rid}: 训练日志的锚点数 {la} 与配方算得的 {ta} 不一致，不是同一个划分")
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

    for cid, c in recipes.items():
        c["used_by"] = sorted(rid for rid, r in runs.items() if r.get("recipe_id") == cid)
    if not a.quiet:
        print(f"配方 {len(recipes)} 条，训练 {len(runs)} 条，评测 {len(evals)} 条")
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
