"""导入器共用的小函数。"""
import re

# 路径必须从非单词字符之后开始，所以 src/tau0_vla/train.py 这类相对路径不会被误改
_ABS_PATH = re.compile(r"(?<![\w.\-])(?:/[^\s/]+)+")


def redact(s):
    """对外公开前，把字符串里的绝对路径换成末级名称。None 原样返回。"""
    if s is None:
        return s
    return _ABS_PATH.sub(lambda m: m.group(0).rsplit("/", 1)[-1], s)
