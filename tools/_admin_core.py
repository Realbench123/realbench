"""记录的删除、恢复与“不再导入”清单。

软删除：文件移进 .local/trash/，不进仓库，可以恢复。
清单 records/_deleted.json 会提交，所以别的机器和自动导入器也知道这条不要再加回来。
只写 id、类型、时间和原因，不写记录内容。原因会随仓库公开，别写敏感信息。

注意：仓库是公开的，git 历史里仍然有这条记录以前的内容。删除不等于从历史里抹掉。
真要抹掉得重写历史，那是另一件事，需要单独决定。
"""
import json, os, re, shutil, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIND = {"tr-": "runs", "ev-": "evals", "rc-": "recipes"}
TYPE = {"tr-": "run", "ev-": "eval", "rc-": "recipe"}
BASES = [os.path.join(ROOT, "records"), os.path.join(ROOT, ".local")]
CURVES = [os.path.join(ROOT, "curves"), os.path.join(ROOT, ".local", "curves")]
TOMB = os.path.join(ROOT, "records", "_deleted.json")
TRASH = os.path.join(ROOT, ".local", "trash")


ID_RE = re.compile(r"^(tr|ev|rc)-[a-z0-9][a-z0-9-]{0,80}$")


def _prefix(i):
    if not isinstance(i, str) or not ID_RE.match(i):
        raise ValueError(f"id 格式不对: {str(i)[:40]!r}（只允许 tr-/ev-/rc- 加小写字母、数字、连字符）")
    for p in KIND:
        if i.startswith(p):
            return p
    raise ValueError(f"不认识的 id: {i}（应以 tr-、ev-、rc- 开头）")


def _jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _jdump(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def is_public(path):
    return os.path.abspath(path).startswith(os.path.join(ROOT, "records") + os.sep)


def locate(i):
    pre = _prefix(i)
    for b in BASES:
        p = os.path.join(b, KIND[pre], i + ".json")
        if os.path.exists(p):
            return p
    return None


def curve_files(i):
    return [os.path.join(c, i + ".json") for c in CURVES if os.path.exists(os.path.join(c, i + ".json"))]


def load_kind(kind):
    out = []
    for b in BASES:
        d = os.path.join(b, kind)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    out.append((os.path.join(d, f), _jload(os.path.join(d, f))))
    return out


def tombstones():
    if not os.path.exists(TOMB):
        return {"version": 1, "entries": []}
    return _jload(TOMB)


def blocked_state(i):
    """已删除(deleted)或不要导入(ignored)返回对应状态，否则 None。导入器写入前调用。"""
    for e in tombstones()["entries"]:
        if e["id"] == i:
            return e["state"]
    return None


def _name(rec):
    return rec.get("name") or f"{rec.get('task', '')} · {rec.get('date', '')}"


def plan(i):
    pre = _prefix(i)
    p = locate(i)
    if not p:
        raise ValueError(f"找不到记录 {i}")
    rec = _jload(p)
    out = {"id": i, "name": _name(rec), "type": TYPE[pre], "delete": [], "modify": [], "blockers": [], "_files": {}}

    def add(rid, rname, rtype, path):
        files = [path] + curve_files(rid)
        out["delete"].append({"id": rid, "name": rname, "type": rtype, "public": is_public(path)})
        out["_files"][rid] = files

    add(i, _name(rec), TYPE[pre], p)
    if pre == "tr-":
        for ep, e in load_kind("evals"):
            if e.get("run_id") == i:
                add(e["id"], _name(e), "eval", ep)
        for _, r in load_kind("runs"):
            if r.get("baseline_id") == i and r["id"] != i:
                out["blockers"].append(f"训练 {r['id']}（{r['name']}）以它为基线，先去改那条的基线，或先删那条")
    elif pre == "ev-":
        rp = locate(rec["run_id"])
        if rp:
            r = _jload(rp)
            for fld in ("evals_closed", "evals_offline"):
                if i in r.get(fld, []):
                    out["modify"].append({"id": r["id"], "what": f"从 {fld} 里去掉 {i}"})
    elif pre == "rc-":
        for _, r in load_kind("runs"):
            if r.get("recipe_id") == i:
                out["blockers"].append(f"训练 {r['id']}（{r['name']}）引用了这个配方")
        for _, c in load_kind("recipes"):
            if c.get("parent_id") == i:
                out["blockers"].append(f"配方 {c['id']}（{c['name']}）以它为 parent")
    out["public"] = any(d["public"] for d in out["delete"])
    return out


def public_plan(pl):
    return {k: v for k, v in pl.items() if not k.startswith("_")}


def _add_tomb(entries):
    t = tombstones()
    have = {(e["id"], e["state"]) for e in t["entries"]}
    for e in entries:
        if (e["id"], e["state"]) not in have:
            t["entries"].append(e)
    _jdump(TOMB, t)


def delete(i, reason=""):
    pl = plan(i)
    if pl["blockers"]:
        raise ValueError("不能删除：" + "；".join(pl["blockers"]))
    now = datetime.datetime.now().astimezone()
    dest = os.path.join(TRASH, now.strftime("%Y%m%d-%H%M%S") + "-" + i)
    moved, deleted_ids, tomb = [], [], []
    for d in pl["delete"]:
        for src in pl["_files"][d["id"]]:
            rel = os.path.relpath(src, ROOT)
            dst = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved.append(rel)
        deleted_ids.append(d["id"])
        if d["public"]:   # 只在 .local 里的样本不进清单
            tomb.append({"id": d["id"], "type": d["type"], "state": "deleted",
                         "at": now.isoformat(timespec="seconds"),
                         "reason": reason if d["id"] == i else f"随 {i} 一起删除"})
    relink = []
    if pl["type"] == "eval":
        rp = locate(_jload(os.path.join(dest, os.path.relpath(pl["_files"][i][0], ROOT)))["run_id"])
        if rp:
            r = _jload(rp)
            for fld in ("evals_closed", "evals_offline"):
                if i in r.get(fld, []):
                    r[fld] = [x for x in r[fld] if x != i]
                    relink.append({"run": r["id"], "field": fld, "id": i})
            _jdump(rp, r)
    if tomb:
        _add_tomb(tomb)
    _jdump(os.path.join(dest, "manifest.json"), {"id": i, "name": pl["name"], "at": now.isoformat(timespec="seconds"),
           "reason": reason, "files": moved, "deleted_ids": deleted_ids, "relink": relink})
    return {"deleted": deleted_ids, "moved": len(moved), "trash": os.path.relpath(dest, ROOT)}


def trash_list():
    out = []
    if os.path.isdir(TRASH):
        for d in sorted(os.listdir(TRASH), reverse=True):
            m = os.path.join(TRASH, d, "manifest.json")
            if os.path.exists(m):
                x = _jload(m)
                x["dir"] = d
                out.append(x)
    return out


def restore(i):
    cand = [x for x in trash_list() if x["id"] == i]
    if not cand:
        raise ValueError(f"回收站里没有 {i}")
    m = cand[0]
    base = os.path.join(TRASH, m["dir"])
    for rel in m["files"]:
        if os.path.exists(os.path.join(ROOT, rel)):
            raise ValueError(f"目标已存在，不覆盖：{rel}")
    for rel in m["files"]:
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(base, rel), dst)
    for rl in m["relink"]:
        rp = locate(rl["run"])
        if rp:
            r = _jload(rp)
            if rl["id"] not in r.setdefault(rl["field"], []):
                r[rl["field"]].append(rl["id"])
                _jdump(rp, r)
    if os.path.exists(TOMB):
        t = tombstones()
        t["entries"] = [e for e in t["entries"] if not (e["id"] in m["deleted_ids"] and e["state"] == "deleted")]
        _jdump(TOMB, t)
    shutil.rmtree(base, ignore_errors=True)
    return {"restored": m["deleted_ids"]}


def ignore(i, reason=""):
    """登记“不要导入”：记录还没进库，但自动导入器以后不要加它。"""
    _prefix(i)
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    _add_tomb([{"id": i, "type": TYPE[_prefix(i)], "state": "ignored", "at": now, "reason": reason}])
    return {"ignored": i}
