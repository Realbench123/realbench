#!/usr/bin/env python3
"""本地管理服务：在这台电脑上打开页面，才会出现删除键。

    .venv/bin/python tools/admin_server.py          # 打开 http://127.0.0.1:8765/
    .venv/bin/python tools/admin_server.py --ignore tr-xxx --reason "不记录"   # 登记“不要导入”，不启动服务

公开站点是静态页面，写不了仓库，所以没有也不该有删除键。
这里只监听 127.0.0.1。写操作要带启动时随机生成的令牌，并校验 Host 和 Origin，
别的网页即使知道端口也读不到令牌（本服务不发 CORS 头），也就发不出有效的请求。
删除只改本地文件，不会自动提交或推送；发布到公开站点是你另外的一步。
"""
import argparse, json, os, secrets, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _admin_core as core

ROOT = core.ROOT
DIST = os.path.join(ROOT, "site", "dist", "index.html")
PREVIEWS = os.path.join(ROOT, ".local", "previews")
LOCK = threading.Lock()


def rebuild(include_local):
    cmd = [sys.executable, os.path.join(HERE, "build_site.py")] + (["--include-local"] if include_local else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    v = subprocess.run([sys.executable, os.path.join(HERE, "validate.py")] + (["--include-local"] if include_local else []) + ["--quiet"],
                       capture_output=True, text=True)
    return {"build_ok": r.returncode == 0, "validate_ok": v.returncode == 0,
            "validate_out": (v.stdout + v.stderr).strip()[-600:]}


def make_handler(port, token, include_local):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj=None, body=None, ctype="application/json; charset=utf-8"):
            data = body if body is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self):
            return self.headers.get("Host", "") in hosts

        def _preview(self):
            """本机预览视频，支持 Range（浏览器拖进度条要用）。文件名只允许 ds-xxx.mp4，不能带路径。"""
            import re as _re
            name = self.path[len("/preview/"):].split("?", 1)[0]
            if not _re.fullmatch(r"ds-[a-z0-9][a-z0-9-]{0,80}\.mp4", name):
                return self._send(400, {"error": "文件名不合法"})
            p = os.path.join(PREVIEWS, name)
            if not os.path.isfile(p):
                return self._send(404, {"error": "没有这个预览"})
            size = os.path.getsize(p)
            a, b, code = 0, size - 1, 200
            rg = self.headers.get("Range", "")
            m = _re.fullmatch(r"bytes=(\d*)-(\d*)", rg)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    a = int(m.group(1))
                    b = int(m.group(2)) if m.group(2) else size - 1
                else:
                    a = max(0, size - int(m.group(2)))
                b = min(b, size - 1)
                if a > b:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            with open(p, "rb") as f:
                f.seek(a)
                data = f.read(b - a + 1)
            self.send_response(code)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            if code == 206:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "host"})
            if self.path in ("/", "/index.html"):
                if not os.path.exists(DIST):
                    return self._send(404, {"error": "还没有生成站点，先运行 tools/build_site.py"})
                return self._send(200, body=open(DIST, "rb").read(), ctype="text/html; charset=utf-8")
            if self.path.startswith("/preview/"):
                return self._preview()
            if self.path == "/api/ping":
                return self._send(200, {"admin": True, "token": token})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, {"error": "host"})
            org = self.headers.get("Origin")
            if org and org not in origins:
                return self._send(403, {"error": "origin"})
            if not secrets.compare_digest(self.headers.get("X-Admin-Token", ""), token):
                return self._send(403, {"error": "token"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, {"error": "请求体不是合法 JSON"})
            try:
                with LOCK:
                    if self.path == "/api/plan":
                        return self._send(200, core.public_plan(core.plan(body["id"])))
                    if self.path == "/api/delete":
                        res = core.delete(body["id"], str(body.get("reason", ""))[:200])
                        res.update(rebuild(include_local))
                        return self._send(200, {"ok": True, **res})
                    if self.path == "/api/restore":
                        res = core.restore(body["id"])
                        res.update(rebuild(include_local))
                        return self._send(200, {"ok": True, **res})
                    if self.path == "/api/trash":
                        return self._send(200, {"items": core.trash_list()})
            except (ValueError, KeyError) as e:
                return self._send(400, {"ok": False, "error": str(e)})
            return self._send(404, {"error": "not found"})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-local", action="store_true", help="不显示 .local/ 里的样本")
    ap.add_argument("--ignore", metavar="ID", help="登记“不要导入”并退出")
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    if a.ignore:
        print(core.ignore(a.ignore, a.reason))
        return
    token = secrets.token_urlsafe(24)
    include_local = not a.no_local
    print("重新生成站点...", rebuild(include_local))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.port, token, include_local))
    print(f"管理页 http://127.0.0.1:{a.port}/   Ctrl+C 退出（只监听本机）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
