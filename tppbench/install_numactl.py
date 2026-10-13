"""Provide `numactl` outside the Python environment.

CPU-only. The two conda-forge packages (numactl and its libnuma dependency,
fetched locally from conda.anaconda.org and checked against the sha256 that
api.anaconda.org lists) are extracted into a separate prefix under
<data-dir>/numactl. A directory that contains nothing
but a `numactl` symlink is created for the server's PATH, so that putting it
first on PATH cannot shadow python, vllm or any other tool. The shared vLLM env
is not touched. The receipt records the path and the hash of the binary.

Packages (conda-forge linux-64, build hb03c661_4, uploaded 2026-08-07):
  numactl-2.0.18-hb03c661_4.conda  25aa780b35407d8380f234d341e9933ec886dfaadb6ca3549fa0132221a311b0
  libnuma-2.0.18-hb03c661_4.conda  db0a0fc67104196afe6732ff055b30829f984f4cf1e8efff4ea5b2bbb1640da2
The binary's RPATH is $ORIGIN/../lib, so it loads the extracted libnuma; its
only other dependency is the system libc (checked locally with ldd). ldd prints
that path unnormalized (`libnuma.so.1 => <prefix>/bin/../lib/libnuma.so.1
(0x...)`), so the receipt compares the realpath of
the `libnuma.so.1 =>` target with the resolved extracted libnuma.

Re-running `install` when the prefix already exists (e.g. extraction finished
but the receipt was not written) takes the receipt-only path: the two packages
are extracted again into a temporary directory beside the prefix and the
prefix must match that tree exactly (every path, file sha256 and symlink
target; `path-bin/` must hold only the `numactl -> ../bin/numactl` link).
Nothing in the prefix is modified; any difference stops with the list.
"""
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time

from . import spec
from .spec import digest, write_json

PACKAGES = {
    'numactl-2.0.18-hb03c661_4.conda': '25aa780b35407d8380f234d341e9933ec886dfaadb6ca3549fa0132221a311b0',
    'libnuma-2.0.18-hb03c661_4.conda': 'db0a0fc67104196afe6732ff055b30829f984f4cf1e8efff4ea5b2bbb1640da2',
}
PATH_DIR_NAME = 'path-bin'


def path_dir(prefix):
    return Path(prefix) / PATH_DIR_NAME


def check_packages(directory):
    found = {}
    for name, sha in PACKAGES.items():
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing package {path}')
        actual = digest(path)
        if actual != sha:
            raise RuntimeError(f'Package hash mismatch for {name}: {actual}')
        found[name] = {'path': str(path), 'sha256': actual}
    return found


def install(packages, prefix, extract=None):
    prefix = Path(prefix)
    if prefix.exists():
        raise FileExistsError(f'{prefix} exists; installation is never overwritten')
    found = check_packages(packages)
    if extract is None:
        from conda_package_handling.api import extract  # base conda, not the vLLM env
    temp = prefix.with_name(prefix.name + '.incomplete')
    if temp.exists():
        raise FileExistsError(f'{temp} exists from an earlier attempt; inspect it first')
    temp.mkdir(parents=True)
    for name in PACKAGES:
        extract(str(Path(packages) / name), dest_dir=str(temp))
    binary = temp / 'bin/numactl'
    if not binary.is_file() or not (temp / 'lib/libnuma.so.1').exists():
        raise RuntimeError('Extracted packages lack bin/numactl or lib/libnuma.so.1')
    (temp / PATH_DIR_NAME).mkdir()
    (temp / PATH_DIR_NAME / 'numactl').symlink_to('../bin/numactl')
    temp.replace(prefix)
    return found


def tree(root, skip=()):
    """{relative path: ('dir',) | ('link', target) | ('file', sha256)}; links not followed."""
    root = Path(root)
    out = {}
    for current, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(current) / name
            rel = str(path.relative_to(root))
            if rel.split(os.sep)[0] in skip:
                continue
            if path.is_symlink():
                out[rel] = ('link', os.readlink(path))
            elif path.is_dir():
                out[rel] = ('dir',)
            else:
                out[rel] = ('file', digest(path))
    return out


def verify_existing(packages, prefix, extract=None):
    """Receipt-only path: the existing prefix must equal a fresh extraction."""
    prefix = Path(prefix)
    found = check_packages(packages)
    if extract is None:
        from conda_package_handling.api import extract  # base conda, not the vLLM env
    with tempfile.TemporaryDirectory(prefix=prefix.name + '.verify-', dir=prefix.parent) as temp:
        for name in PACKAGES:
            extract(str(Path(packages) / name), dest_dir=temp)
        expected = tree(temp)
    actual = tree(prefix, skip=(PATH_DIR_NAME,))
    diff = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
    if diff:
        raise RuntimeError(f'{prefix} differs from the packages at: {diff[:20]}')
    if tree(path_dir(prefix)) != {'numactl': ('link', '../bin/numactl')}:
        raise RuntimeError(f'{path_dir(prefix)} must hold only numactl -> ../bin/numactl')
    return found


LDD_LIBNUMA = re.compile(r'^\s*libnuma\.so\.1\s*=>\s*(.*?)\s*(?:\(0x[0-9a-fA-F]+\))?\s*$')


def ldd_libnuma(stdout):
    """The path ldd resolves libnuma.so.1 to; error if absent, repeated or 'not found'."""
    hits = [m.group(1) for m in map(LDD_LIBNUMA.match, stdout.splitlines()) if m]
    if len(hits) != 1:
        raise RuntimeError(f'ldd shows {len(hits)} libnuma.so.1 lines')
    if not hits[0] or hits[0] == 'not found' or not hits[0].startswith('/'):
        raise RuntimeError(f'ldd does not resolve libnuma.so.1: {hits[0]!r}')
    return hits[0]


def receipt(prefix, packages=None, run=subprocess.run):
    prefix = Path(prefix)
    binary = prefix / 'bin/numactl'
    entry = path_dir(prefix) / 'numactl'
    if entry.resolve() != binary.resolve():
        raise RuntimeError('PATH entry does not resolve to the extracted binary')
    if sorted(p.name for p in path_dir(prefix).iterdir()) != ['numactl']:
        raise RuntimeError('The PATH directory must contain only numactl')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    ldd = run(['ldd', str(binary)], capture_output=True, text=True, timeout=20, env=env)
    show = run([str(entry), '--show'], capture_output=True, text=True, timeout=20, env=env)
    if show.returncode != 0:
        raise RuntimeError(f'numactl --show failed: {show.stderr}')
    if ldd.returncode != 0:
        raise RuntimeError(f'ldd failed: {ldd.stderr}')
    libnuma = (prefix / 'lib/libnuma.so.1').resolve()
    loaded = ldd_libnuma(ldd.stdout)
    if os.path.realpath(loaded) != str(libnuma):
        raise RuntimeError(f'numactl does not load the extracted libnuma: {loaded}')
    return {'numactl': str(binary), 'numactl_sha256': digest(binary),
            'path_dir': str(path_dir(prefix)), 'libnuma': str(libnuma),
            'libnuma_sha256': digest(libnuma), 'packages': packages,
            'ldd': ldd.stdout, 'numactl_show': show.stdout,
            'host': platform.node(), 'wall_ns': time.time_ns()}


def cli(packages, prefix=None, receipt_path=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if not shutil.which('ldd'):
        raise RuntimeError('ldd missing')
    prefix = Path(prefix or spec.data_dir() / 'numactl')
    receipt_path = Path(receipt_path or spec.data_dir() / 'setup/numactl-install.json')
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    if prefix.exists():
        found, mode = verify_existing(packages, prefix), 'receipt-only (existing prefix verified)'
    else:
        found, mode = install(packages, prefix), 'extracted'
    data = {**receipt(prefix, found), 'install_mode': mode}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, data)
    print(json.dumps({'receipt': str(receipt_path), 'numactl_sha256': data['numactl_sha256']}))
