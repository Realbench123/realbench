#!/usr/bin/env python3
"""算协议参数块的内容哈希（前 8 位），写进评测记录的 protocol.version。
只取 ```yaml 块里的内容，去掉注释和空白差异之后再哈希，所以改文字说明不会改版本。"""
import hashlib, re, sys, yaml, json

def protocol_hash(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"```yaml\n(.*?)\n```", txt, re.S)
    if not m:
        raise SystemExit("协议文件里没有 yaml 块")
    data = yaml.safe_load(m.group(1))
    canon = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return data.get("name"), hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]

if __name__ == "__main__":
    name, h = protocol_hash(sys.argv[1])
    print(f"{name} {h}")
