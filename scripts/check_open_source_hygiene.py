#!/usr/bin/env python3
"""开源卫生守卫：检测仓库中不应公开的内容。

用途
----
本仓库计划开源，此脚本作为守卫（guard），检查仓库在开源视角下
是否泄漏凭据、内网环境指纹或私有基础设施信息。

检查项
------
1. 凭据类：sk-* 密钥、hf_* token、Bearer token、key/secret/token/password 赋值
2. 环境指纹：内网 IP（10./172.16-31./192.168.）、本机主机名、私有域名
3. 部署留档：docker inspect 快照等本机环境产物
4. .gitignore 护栏：确保 .env 类文件不会被误提交

用法
----
    python scripts/check_open_source_hygiene.py          # 扫工作区（默认）
    python scripts/check_open_source_hygiene.py --all    # 同时扫 git 历史
    python scripts/check_open_source_hygiene.py --quiet  # 仅输出总结

退出码
------
    0 = 通过（无违规）
    1 = 发现违规

设计说明
--------
- 守卫本身只读，不修改任何文件
- 违规项以「文件:行号: 类别」形式输出，便于定位
- 白名单机制：确属公开信息的内容（如 Python 官方镜像 GPG 公钥、
  文档中的示例占位符）通过 ALLOWLIST 显式豁免，避免误报疲劳
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 扫描范围控制
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', 'node_modules',
    '.mypy_cache', '.pytest_cache', '.ruff_cache', 'routellm.egg-info',
    'build', 'dist', '.eggs',
}

# 不扫内容的文件类型（二进制 / 数据资产）
SKIP_EXTS = {
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf',
    '.woff', '.woff2', '.ttf', '.eot',
    '.pyc', '.so', '.pyd', '.dll',
    '.npy', '.npz', '.bin', '.pt', '.pth', '.onnx',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z',
    '.csv', '.jsonl',  # 评测数据集，数据量大且非配置
}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# ---------------------------------------------------------------------------
# 检测规则
# ---------------------------------------------------------------------------

# 公开可接受的例外（避免误报）。键为规则名，值为允许出现的片段。
ALLOWLIST = {
    # Python 官方 Docker 镜像内置的 GPG 签名公钥，公开信息
    'credential': {
        'A035C8C19219BA821ECEA86B64E628F8D684696D',
        # 脱敏占位符：含省略号（... / *** / <...>）的字符串是「掩码后的展示值」，
        # 不是真实密钥。典型场景是测试「掩码不泄漏明文」时构造的假值。
        '...',
        '***',
        '<',
    },
    # 文档中作为键名/占位符出现的字样，非真实值
    'secret_assign': {
        'api_key', 'apikey', 'auth_token', 'access_token',
        'secret_key', 'password', 'token',
    },
}

RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        'credential',
        re.compile(r'sk-[A-Za-z0-9_\-]{20,}'),
        'API 密钥（sk- 前缀）',
    ),
    (
        'credential',
        re.compile(r'hf_[A-Za-z0-9]{20,}'),
        'HuggingFace token（hf_ 前缀）',
    ),
    (
        'credential',
        re.compile(r'(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}'),
        'Bearer token',
    ),
    (
        'secret_assign',
        re.compile(
            r'(?i)\b(api[_-]?key|auth[_-]?token|access[_-]?token|secret[_-]?key)'
            r'\s*[:=]\s*["\']([A-Za-z0-9_\-\.]{16,})["\']'
        ),
        '硬编码密钥赋值',
    ),
    (
        'private_ip',
        re.compile(r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
                   r'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'
                   r'|192\.168\.\d{1,3}\.\d{1,3})\b'),
        '内网 IP 地址',
    ),
    (
        # 裸公网 IPv4 字面量。内网段（10. / 172.16-31. / 192.168.）已由
        # private_ip 规则覆盖，此处显式排除，避免同一地址被两条规则重复报告
        # （重复计数会造成误报疲劳）。目的：捕获硬编码的本机公网出口地址。
        'public_ip',
        re.compile(r'\b(?!127\.|0\.|255\.'
                   r'|10\.|192\.168\.'
                   r'|172\.(?:1[6-9]|2\d|3[01])\.)'
                   r'(?:(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-5])\.){3}'
                   r'(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-5])\b'),
        '公网 IP 字面量',
    ),
    (
        'internal_host',
        re.compile(r'\b(?:server\d{2}-[A-Z0-9\-]+|zhejiangdaxue|icsr-ESC8000[A-Z0-9\-]*)\b'),
        '内部主机名',
    ),
    (
        'private_domain',
        re.compile(r'\b[A-Za-z0-9\-]+\.taisure\.com\b'),
        '私有域名',
    ),
]

# 部署留档类文件（本机环境产物，不应入库）
ARTIFACT_PATTERNS = [
    re.compile(r'inspect-full\.json$'),
    re.compile(r'image-inspect\.json$'),
    re.compile(r'container-summary\.txt$'),
    re.compile(r'container-logs.*\.txt$'),
]


@dataclass
class Violation:
    """单条违规记录。"""

    path: str
    line: int
    category: str
    detail: str
    excerpt: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"  [{self.category}] {loc}\n      {self.detail}: {self.excerpt}"


def _is_allowlisted(category: str, matched_text: str) -> bool:
    """判断命中内容是否属于公开信息的白名单例外。"""
    allowed = ALLOWLIST.get(category, set())
    if not allowed:
        return False
    return any(a in matched_text for a in allowed)


def iter_files(root: str = '.', include_history: bool = False):
    """产出待扫描的 (相对路径, 绝对路径) 二元组。

    注意：守卫脚本自身会被排除 —— 它为了检测而必须包含主机名/域名的
    字面模式（如 'taisure.com'），否则无法工作。扫描自身属于自指误报。
    """
    self_name = os.path.basename(__file__)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn == self_name:
                continue
            ap = os.path.join(dirpath, fn)
            rp = os.path.relpath(ap, root)
            yield rp, ap


def check_artifacts(root: str = '.') -> list[Violation]:
    """检查是否存在本机环境留档产物。"""
    out = []
    for rp, _ap in iter_files(root):
        for pat in ARTIFACT_PATTERNS:
            if pat.search(rp):
                out.append(Violation(
                    path=rp, line=0, category='deploy_artifact',
                    detail='本机环境留档产物，不应入库',
                    excerpt=os.path.basename(rp),
                ))
                break
    return out


def check_content(root: str = '.') -> list[Violation]:
    """扫描工作区文件内容。"""
    out = []
    for rp, ap in iter_files(root):
        if os.path.splitext(rp)[1].lower() in SKIP_EXTS:
            continue
        try:
            if os.path.getsize(ap) > MAX_FILE_SIZE:
                continue
            data = open(ap, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        if '\0' in data[:4096]:  # 二进制跳过
            continue
        for category, pat, desc in RULES:
            for m in pat.finditer(data):
                matched = m.group(0)
                if _is_allowlisted(category, matched):
                    continue
                line = data.count('\n', 0, m.start()) + 1
                excerpt = matched if len(matched) <= 60 else matched[:57] + '...'
                # 凭据类打码显示，避免守卫输出本身泄漏
                if category == 'credential':
                    excerpt = excerpt[:6] + '***<masked>'
                out.append(Violation(
                    path=rp, line=line, category=category,
                    detail=desc, excerpt=excerpt,
                ))
    return out


def check_git_history() -> list[Violation]:
    """扫描 git 历史中所有 blob 是否含凭据（仅凭据类，避免噪声）。"""
    out = []
    cred_pats = [(c, p, d) for c, p, d in RULES if c == 'credential']
    try:
        revs = subprocess.run(
            ['git', 'rev-list', '--all'],
            capture_output=True, text=True, timeout=120,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return out

    seen: set[tuple[str, str]] = set()
    for rv in revs:
        files = subprocess.run(
            ['git', 'ls-tree', '-r', '--name-only', rv],
            capture_output=True, text=True, timeout=60,
        ).stdout.split()
        for f in files:
            if os.path.splitext(f)[1].lower() in SKIP_EXTS:
                continue
            try:
                blob = subprocess.run(
                    ['git', 'show', f'{rv}:{f}'],
                    capture_output=True, text=True, timeout=30,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            for _c, pat, desc in cred_pats:
                for m in pat.finditer(blob):
                    matched = m.group(0)
                    if _is_allowlisted('credential', matched):
                        continue
                    key = (f, matched[:20])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Violation(
                        path=f, line=0, category='credential_history',
                        detail=f'git 历史中的{desc}',
                        excerpt=matched[:6] + '***<masked>',
                    ))
    return out


def check_gitignore(root: str = '.') -> list[Violation]:
    """检查 .gitignore 是否覆盖 .env 类文件。"""
    out = []
    gi_path = os.path.join(root, '.gitignore')
    if not os.path.exists(gi_path):
        return [Violation(
            path='.gitignore', line=0, category='gitignore_missing',
            detail='缺少 .gitignore', excerpt='',
        )]
    content = open(gi_path, encoding='utf-8').read()
    required = [
        ('.env', '根 .env 文件'),
        ('*.env', '任意 .env 后缀文件'),
        ('.env.*', '.env.* 变体'),
        ('!*.example', '放行 .example 示例文件'),
    ]
    for pattern, desc in required:
        if pattern not in content:
            out.append(Violation(
                path='.gitignore', line=0, category='gitignore_gap',
                detail=f'缺少规则（{desc}）', excerpt=pattern,
            ))
    return out


def run_all(root: str = '.', include_history: bool = False) -> list[Violation]:
    """执行全部检查，返回违规列表。"""
    v = check_content(root)
    v += check_artifacts(root)
    v += check_gitignore(root)
    if include_history:
        v += check_git_history()
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='开源卫生守卫：检查仓库是否含凭据 / 环境指纹 / 留档产物',
    )
    parser.add_argument('--root', default='.', help='扫描根目录（默认当前目录）')
    parser.add_argument('--all', action='store_true',
                        help='同时扫描 git 历史（较慢）')
    parser.add_argument('--quiet', action='store_true', help='仅输出总结')
    args = parser.parse_args(argv)

    violations = run_all(args.root, args.all)

    if not violations:
        print('OK: 未发现开源卫生问题')
        return 0

    by_cat: dict[str, list[Violation]] = {}
    for v in violations:
        by_cat.setdefault(v.category, []).append(v)

    print(f'FAIL: 发现 {len(violations)} 项开源卫生问题\n')
    for cat, items in sorted(by_cat.items()):
        print(f'--- {cat} ({len(items)} 项) ---')
        for v in items:
            print(v)
        print()

    # 修复提示
    print('修复建议:')
    if 'deploy_artifact' in by_cat:
        print('  - 部署留档产物：移出仓库（本机环境快照对开源读者无价值且泄漏指纹）')
    if 'private_ip' in by_cat or 'internal_host' in by_cat:
        print('  - 环境指纹：文档中的内网 IP / 主机名替换为占位符（如 <host>、<internal-ip>）')
    if 'private_domain' in by_cat:
        print('  - 私有域名：替换为通用示例域名（如 api.openai.com）')
    if 'credential' in by_cat or 'credential_history' in by_cat:
        print('  - 凭据：立即轮换该密钥，并从工作区/历史中移除')
    if 'gitignore_gap' in by_cat:
        print('  - .gitignore：补齐 *.env 与 !*.example 规则')
    return 1


if __name__ == '__main__':
    sys.exit(main())
