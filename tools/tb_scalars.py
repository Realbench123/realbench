#!/usr/bin/env python3
"""不依赖 tensorboard / tensorflow，只用标准库读 TensorBoard 事件文件里的标量。

格式：TFRecord（8 字节长度 + 4 字节长度校验 + 数据 + 4 字节数据校验），
数据是 Event protobuf：wall_time(1, double)、step(2, int64)、summary(5, 消息)，
summary.value(1, 重复消息)：tag(1, 字符串)、simple_value(2, float32)；
较新的写法把标量放在 tensor(8) 里，这里一并处理。

只读。校验和不验证（文件来自共享盘上的复制，复制时同学已核过 SHA256）。
用法：
    tb_scalars.py FILE --list                       # 列出所有标量名及点数
    tb_scalars.py FILE --tags a,b --out out.json    # 导出指定标量 {tag: {step:[], value:[]}}
"""
import argparse, json, struct, sys


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def fields(b):
    i, n = 0, len(b)
    while i < n:
        k, i = varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = varint(b, i)
        elif w == 1:
            v = b[i:i + 8]; i += 8
        elif w == 2:
            ln, i = varint(b, i); v = b[i:i + ln]; i += ln
        elif w == 5:
            v = b[i:i + 4]; i += 4
        else:
            return
        yield f, w, v


def records(path):
    with open(path, "rb") as fh:
        while True:
            h = fh.read(12)
            if len(h) < 12:
                return
            ln = struct.unpack("<Q", h[:8])[0]
            d = fh.read(ln)
            fh.read(4)
            if len(d) < ln:
                return
            yield d


def scalars(path, want=None):
    """yield (tag, step, value)"""
    for d in records(path):
        step = None
        vals = []
        for f, w, v in fields(d):
            if f == 2 and w == 0:
                step = v
            elif f == 5 and w == 2:
                for f2, w2, v2 in fields(v):
                    if f2 != 1 or w2 != 2:
                        continue
                    tag, x = None, None
                    for f3, w3, v3 in fields(v2):
                        if f3 == 1 and w3 == 2:
                            tag = v3.decode("utf-8", "ignore")
                        elif f3 == 2 and w3 == 5:
                            x = struct.unpack("<f", v3)[0]
                        elif f3 == 8 and w3 == 2:      # tensor：float_val(5) 或 tensor_content(4)
                            for f4, w4, v4 in fields(v3):
                                if f4 == 4 and w4 == 2 and len(v4) == 4:
                                    x = struct.unpack("<f", v4)[0]
                                elif f4 == 5 and w4 == 2 and len(v4) >= 4:
                                    x = struct.unpack("<f", v4[:4])[0]
                    if tag is not None and x is not None:
                        vals.append((tag, x))
        if step is None:
            continue
        for tag, x in vals:
            if want is None or tag in want:
                yield tag, step, x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tags")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.list:
        cnt, rng = {}, {}
        for t, s, x in scalars(a.file):
            cnt[t] = cnt.get(t, 0) + 1
            lo, hi = rng.get(t, (s, s))
            rng[t] = (min(lo, s), max(hi, s))
        for t in sorted(cnt):
            print(f"{cnt[t]:6d} 点  步 {rng[t][0]}–{rng[t][1]}  {t}")
        return
    want = set(a.tags.split(",")) if a.tags else None
    out = {}
    for t, s, x in scalars(a.file, want):
        o = out.setdefault(t, {"step": [], "value": []})
        o["step"].append(s)
        o["value"].append(x)
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False)
        print({t: len(v["step"]) for t, v in out.items()})
    else:
        print(json.dumps({t: {"n": len(v["step"]), "first": v["value"][0], "last": v["value"][-1]} for t, v in out.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
