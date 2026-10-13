"""CPU-only fetch of the pinned Qwen2.5-72B-Instruct into a HuggingFace cache layout.

Metadata comes from the hub API for the pinned revision (one JSON file, see README); each
weight file is fetched in eight byte ranges from `--endpoint` (huggingface.co or a mirror),
under a per-blob flock in the cache's lock directory, checked for size and upstream content
digest, and published atomically (rename into blobs/, then an atomic symlink into
snapshots/<revision>/).

Two separate modes, so that hashing 145 GB can be scheduled when nothing is being timed:

* download: fetches. Weight (LFS) files are left as complete, size-checked byte-range parts
  and are NOT hashed or published. Non-LFS files (configs, tokenizer; 10 files, about
  11 MB) are fetched, digest-checked and published at once, so that prompt construction
  can start before the weights are verified.
* verify: hashes. For every file, a published blob is re-hashed; otherwise the parts are
  streamed once into `<sha256>.incomplete` while hashing, checked for size and sha256,
  renamed to the blob, linked into the snapshot, and the parts are removed.
  MODEL_READY.json is written only when all 47 files verified.
"""
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .spec import MODEL, MODEL_FILES, MODEL_SAFETENSORS_BYTES, REVISION, write_json

RANGES = 8
RANGE_THRESHOLD = 32 * 1024 * 1024
SPACE_MARGIN = 12 * 1024 ** 3
TAG = 'tppbench'
ENDPOINT = 'https://huggingface.co'


def digest(path, lfs):
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_metadata(path):
    data = json.loads(Path(path).read_text())
    if data.get('sha') != REVISION or data.get('id', MODEL) != MODEL:
        raise RuntimeError('Metadata is not the pinned revision')
    items = data['siblings']
    if len(items) != MODEL_FILES:
        raise RuntimeError(f'Expected {MODEL_FILES} files, metadata lists {len(items)}')
    weights = [i for i in items if i['rfilename'].endswith('.safetensors')]
    if sum(i['size'] for i in weights) != MODEL_SAFETENSORS_BYTES or not all(i.get('lfs') for i in weights):
        raise RuntimeError('Safetensors byte total or LFS records differ from the pinned revision')
    for item in items:
        name = Path(item['rfilename'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(item['rfilename'])
    # Small files first, so construction can start before the weights arrive.
    return sorted(items, key=lambda i: (bool(i.get('lfs')), i['rfilename']))


class Paths:
    def __init__(self, item, hf_home):
        self.item = item
        self.name, self.size = item['rfilename'], item['size']
        self.lfs = item.get('lfs')
        self.expected = self.lfs['sha256'] if self.lfs else item['blobId']
        base = Path(hf_home) / 'hub' / ('models--' + MODEL.replace('/', '--'))
        self.blob = base / 'blobs' / self.expected
        self.target = base / 'snapshots' / REVISION / self.name
        self.parts = base / 'blobs' / (self.expected + f'.{TAG}-parts')
        self.lock = Path(hf_home) / 'hub' / '.locks' / base.name / (self.expected + '.lock')
        self.width = math.ceil(self.size / (RANGES if self.size >= RANGE_THRESHOLD else 1))

    def starts(self):
        return list(range(0, self.size, self.width))

    def part(self, start):
        return self.parts / str(start)

    def part_length(self, start):
        return min(self.size, start + self.width) - start

    def parts_complete(self):
        return all(self.part(s).exists() and self.part(s).stat().st_size == self.part_length(s)
                   for s in self.starts())

    def make_dirs(self):
        for d in (self.blob.parent, self.target.parent, self.lock.parent):
            d.mkdir(parents=True, exist_ok=True)


STATUS_LINE = re.compile(r'^HTTP/\S+\s+(\d{3})\b')
CONTENT_RANGE = re.compile(r'^bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$')


def response_hops(header_text):
    """Split curl --dump-header output into responses, in order:
    [{'status': int, 'headers': {lower-case name: value}}]. 1xx interim
    responses are dropped. With --location, every redirect hop is listed."""
    hops = []
    for line in header_text.splitlines():
        line = line.rstrip('\r')
        m = STATUS_LINE.match(line)
        if m:
            hops.append({'status': int(m[1]), 'headers': {}})
        elif hops and ':' in line:
            name, _, value = line.partition(':')
            hops[-1]['headers'][name.strip().lower()] = value.strip()
    return [h for h in hops if h['status'] >= 200]


def judge_final(header_text, first, end, size, have, chunk_size):
    """Decide from the FINAL response only (an earlier redirect hop's
    Content-Range proves nothing; a mirror has answered with a 307 carrying
    'content-range: bytes 45805-79024/79025', then a 200 with the whole file).
    Returns 'append' (206 for exactly bytes first-end/size), 'whole' (200,
    nothing held yet, body is the full file), 'restart' (200 otherwise: delete
    the part and start the range again from its beginning) or 'discard'."""
    hops = response_hops(header_text)
    if not hops:
        return 'discard'
    final = hops[-1]
    if final['status'] == 206:
        m = CONTENT_RANGE.match(final['headers'].get('content-range', ''))
        if m and (int(m[1]), int(m[2])) == (first, end) and m[3] in (str(size), '*') \
                and chunk_size <= end - first + 1:
            return 'append'
        return 'discard'
    if final['status'] == 200:
        return 'whole' if have == 0 and chunk_size == size else 'restart'
    return 'discard'


# A pass is one curl run of at most 300 s. Only passes that add no bytes count
# toward the stall limit (a slow mirror needs several productive passes per
# range); the pass ceiling bounds a range that crawls (200 x 300 s ~ 17 h).
MAX_STALLED_PASSES = 10
MAX_PASSES = 200


def fetch_range(paths, start, run=subprocess.run, sleep=time.sleep, endpoint=ENDPOINT):
    """Resume a range with fresh curl passes;
    append only what the final response confirms (judge_final). Stops after
    MAX_STALLED_PASSES consecutive passes without progress, or MAX_PASSES in all."""
    end = start + paths.part_length(start) - 1
    p = paths.part(start)
    url = f'{endpoint.rstrip("/")}/{MODEL}/resolve/{REVISION}/{paths.name}'
    stalled = 0
    for number in range(1, MAX_PASSES + 1):
        have = p.stat().st_size if p.exists() else 0
        if have == end - start + 1:
            return p
        if have > end - start + 1:
            raise RuntimeError(f'Range overflow: {p}')
        chunk = paths.parts / (str(start) + '.chunk')
        headers = paths.parts / (str(start) + '.headers')
        headers.unlink(missing_ok=True)
        chunk.unlink(missing_ok=True)        # never judge a previous pass's body
        argv = ['curl', '--fail', '--location', '--silent', '--show-error',
                '--connect-timeout', '20', '--max-time', '300',
                '--range', f'{start + have}-{end}', '--dump-header', str(headers),
                '--output', str(chunk), url]
        print(json.dumps({'argv': argv}), flush=True)
        run(argv, check=False)
        header_text = headers.read_text() if headers.exists() else ''
        size = chunk.stat().st_size if chunk.exists() else 0
        action = judge_final(header_text, start + have, end, paths.size, have, size) if chunk.exists() else 'discard'
        print(json.dumps({'range': [start + have, end], 'chunk_bytes': size, 'action': action}), flush=True)
        if action == 'append':
            with p.open('ab') as dest, chunk.open('rb') as src:
                shutil.copyfileobj(src, dest)
        elif action == 'whole':
            # The full file arrived: keep exactly this part's bytes (all of it
            # when the file is one range). The end-of-file hash still decides.
            temp = p.with_name(p.name + '.whole')
            with temp.open('wb') as dest, chunk.open('rb') as src:
                src.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = src.read(min(remaining, 8 * 1024 * 1024))
                    if not block:
                        raise RuntimeError('Full-file body shorter than its declared size')
                    dest.write(block)
                    remaining -= len(block)
            temp.replace(p)
        elif action == 'restart':
            p.unlink(missing_ok=True)
        chunk.unlink(missing_ok=True)
        now = p.stat().st_size if p.exists() else 0
        stalled = 0 if now > have else stalled + 1
        print(json.dumps({'range': [start, end], 'pass': number, 'held_before': have, 'held_after': now,
                          'stalled_passes': stalled, 'max_stalled': MAX_STALLED_PASSES,
                          'max_passes': MAX_PASSES}), flush=True)
        if now == end - start + 1:
            return p
        if stalled >= MAX_STALLED_PASSES:
            raise RuntimeError(f'Incomplete range: {paths.name} {start}-{end}: '
                               f'{stalled} consecutive passes without progress ({now} bytes held)')
        sleep(min(stalled + 1, 10))
    raise RuntimeError(f'Incomplete range: {paths.name} {start}-{end}: pass ceiling {MAX_PASSES} reached')


def publish_link(paths):
    if paths.target.is_symlink() or paths.target.exists():
        if paths.target.resolve() != paths.blob.resolve():
            raise RuntimeError(f'Conflicting shared snapshot path: {paths.target}')
        return
    temporary = paths.target.with_name(paths.target.name + f'.{TAG}-link')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(paths.blob, paths.target.parent))
    temporary.replace(paths.target)


def assemble_verified(paths):
    """Stream parts once into .incomplete while hashing; publish only on match."""
    temp = paths.blob.with_name(paths.expected + '.incomplete')
    h = hashlib.sha256() if paths.lfs else hashlib.sha1()
    if not paths.lfs:
        h.update(f'blob {paths.size}\0'.encode())
    with temp.open('wb') as dest:
        for start in paths.starts():
            with paths.part(start).open('rb') as src:
                for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                    h.update(block)
                    dest.write(block)
        dest.flush()
        os.fsync(dest.fileno())
    if temp.stat().st_size != paths.size or h.hexdigest() != paths.expected:
        temp.unlink()
        raise RuntimeError(f'Digest or size mismatch: {paths.name}; parts kept for inspection')
    temp.replace(paths.blob)
    shutil.rmtree(paths.parts)


def download_one(item, hf_home, run=subprocess.run, sleep=time.sleep, endpoint=ENDPOINT):
    paths = Paths(item, hf_home)
    paths.make_dirs()
    with paths.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if paths.blob.exists():
            # Never hashed here; verify mode re-hashes every published blob.
            if not paths.lfs:
                publish_link(paths)
            return {'file': paths.name, 'state': 'blob-present-unverified'}
        paths.parts.mkdir(exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=RANGES) as pool:
            list(pool.map(lambda s: fetch_range(paths, s, run, sleep, endpoint), paths.starts()))
        if not paths.parts_complete():
            raise RuntimeError(f'Parts incomplete after download: {paths.name}')
        if paths.lfs:
            return {'file': paths.name, 'state': 'parts-complete-unhashed', 'bytes': paths.size}
        assemble_verified(paths)        # non-LFS: a few MB, digest-checked now
        publish_link(paths)
        return {'file': paths.name, 'state': 'verified-published', 'bytes': paths.size}


def verify_one(item, hf_home):
    paths = Paths(item, hf_home)
    paths.make_dirs()
    with paths.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if paths.blob.exists():
            if paths.blob.stat().st_size != paths.size or digest(paths.blob, paths.lfs) != paths.expected:
                raise RuntimeError(f'Existing shared blob is corrupt: {paths.blob}; do not overwrite')
        elif paths.parts.is_dir() and paths.parts_complete():
            assemble_verified(paths)
        else:
            raise RuntimeError(f'Not downloaded yet: {paths.name}')
        publish_link(paths)
        return {'file': paths.name, 'state': 'verified-published', 'digest': paths.expected, 'bytes': paths.size}


def present_bytes(item, hf_home):
    """Bytes of this file already on disk (a published blob, or downloaded parts)."""
    paths = Paths(item, hf_home)
    if paths.blob.exists():
        return paths.size
    if paths.parts.is_dir():
        return min(paths.size, sum(p.stat().st_size for p in paths.parts.iterdir()
                                   if p.name.isdigit()))
    return 0


def download(metadata, hf_home, run=subprocess.run, sleep=time.sleep, free=None, endpoint=ENDPOINT):
    items = load_metadata(metadata)
    need = sum(i['size'] - present_bytes(i, hf_home) for i in items) + SPACE_MARGIN
    available = shutil.disk_usage(hf_home).free if free is None else free
    if available < need:
        raise RuntimeError(f'Insufficient free space: {available} < {need}')
    rows = [download_one(item, hf_home, run, sleep, endpoint) for item in items]
    return {'mode': 'download', 'endpoint': endpoint, 'files': rows,
            'note': 'weights unhashed; run verify mode before timing'}


def verify(metadata, hf_home):
    items = load_metadata(metadata)
    rows = [verify_one(item, hf_home) for item in items]
    if len(rows) != MODEL_FILES or any(r['state'] != 'verified-published' for r in rows):
        raise RuntimeError('Not every file verified')
    return {'mode': 'verify', 'files': rows}


def snapshot_dir(hf_home):
    return Path(hf_home) / 'hub' / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / REVISION


def cli(mode, metadata, hf_home, ready, endpoint=ENDPOINT):
    """`download` or `verify`; verify writes `ready` (MODEL_READY.json) naming the snapshot."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    metadata, hf_home, ready = Path(metadata), Path(hf_home), Path(ready)
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    ready.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'download':
        result = download(metadata, hf_home, endpoint=endpoint)
        write_json(ready.parent / f'download-{time.strftime("%Y%m%dT%H%M%S")}.json',
                   {**result, 'metadata_sha256': metadata_sha})
        print('DOWNLOADED: parts complete; weights not yet hashed (run verify)', flush=True)
        return
    result = verify(metadata, hf_home)
    write_json(ready, {'repo': MODEL, 'revision': REVISION, 'snapshot': str(snapshot_dir(hf_home).resolve()),
                       'files': len(result['files']), 'bytes': sum(r['bytes'] for r in result['files']),
                       'metadata_sha256': metadata_sha, 'verified': result['files']})
    print('COMPLETE: the pinned model snapshot verified', flush=True)
