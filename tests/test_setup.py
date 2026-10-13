"""CPU-only tests of the setup steps: numactl install and probe, the fetcher's
download/verify modes, prompt banks. curl and subprocesses are mocked."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tppbench import construct
from tppbench import fetch_model
from tppbench import install_numactl
from tppbench import probe_numactl
from tppbench import runner
from tppbench.spec import BANK_LEVELS, BANK_MEASURED, BURST_REQUESTS, COMMIT, MEASURED, MODEL, REVISION, cpu_list


class Probe(unittest.TestCase):
    def output(self, node, inner_mems='0-1', inner_cpus=None):
        cpus = inner_cpus or ('0-21,52-73' if node == 0 else '26-43,78-95')
        return (f'Cpus_allowed_list:\t0-103\nMems_allowed_list:\t0-1\n'
                f'Cpus_allowed_list:\t{cpus}\nMems_allowed_list:\t{inner_mems}\n')

    @staticmethod
    def show(node, membind=None):
        """Realistic `numactl --show` output (numactl 2.0.x) at the inner level;
        membind=False drops the membind line. Note the trailing space."""
        cpus = ' '.join(str(c) for c in cpu_list('0-21,52-73' if node == 0 else '26-43,78-95'))
        lines = ['policy: bind', f'preferred node: {node}', f'physcpubind: {cpus} ', 'cpubind: 0 1 ',
                 'nodebind: 0 1 ', f'membind: {node if membind is None else membind} ', 'preferred: ']
        return '\n'.join(l for l in lines if membind is not False or not l.startswith('membind')) + '\n'

    def evaluate(self, node, stdout, rc=0, show=None, show_rc=0):
        return probe_numactl.evaluate(node, rc, stdout, show_rc, self.show(node) if show is None else show)

    def test_exact_declared_command(self):
        self.assertEqual(probe_numactl.command(1), [
            'taskset', '-c', '22-25,74-77', 'numactl', '--cpunodebind=0,1', '--membind=0,1', 'sh', '-c',
            "grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status; numactl --physcpubind=26-43,78-95 "
            "--membind=1 grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status"])
        self.assertEqual(probe_numactl.command(0, restricted=False)[0], 'numactl')
        self.assertEqual(probe_numactl.show_command(1), [
            'taskset', '-c', '22-25,74-77', 'numactl', '--cpunodebind=0,1', '--membind=0,1', 'sh', '-c',
            'numactl --physcpubind=26-43,78-95 --membind=1 numactl --show'])

    def test_comparison_inner_level(self):
        self.assertTrue(self.evaluate(0, self.output(0))['pass'])
        self.assertTrue(self.evaluate(1, self.output(1))['pass'])
        self.assertIn('membind: 0 \n', self.show(0))          # trailing space tolerated
        self.assertEqual(self.evaluate(0, self.output(0))['membind_tokens'], ['0'])
        self.assertFalse(self.evaluate(0, self.output(0), rc=1)['pass'])
        self.assertFalse(self.evaluate(0, self.output(0), show_rc=1)['pass'])
        self.assertFalse(self.evaluate(0, self.output(0, inner_cpus='22-25,74-77'))['pass'])
        bad_outer = self.output(0).replace('0-103', '22-25,74-77', 1)
        self.assertFalse(self.evaluate(0, bad_outer)['pass'])
        self.assertFalse(self.evaluate(0, 'garbage')['pass'])
        # Inner Mems_allowed_list must equal the outer level's (0-1), not the node.
        result = self.evaluate(0, self.output(0, inner_mems='0'))
        self.assertFalse(result['pass'])
        self.assertFalse(result['checks']['inner_mems_equal_outer'])
        # Wrong node, extra node, missing line, two lines.
        for show in (self.show(0, membind=1), self.show(0, membind='0 1'), self.show(0, membind=False)):
            result = self.evaluate(0, self.output(0), show=show)
            self.assertFalse(result['pass'])
            self.assertFalse(result['checks']['inner_membind_its_node'])
        self.assertIsNone(self.evaluate(0, self.output(0), show=self.show(0, membind=False))['membind_tokens'])
        self.assertFalse(self.evaluate(0, self.output(0), show=self.show(0) + 'membind: 0\n')['pass'])
        self.assertFalse(self.evaluate(1, self.output(1), show=self.show(1, membind=0))['pass'])

    def probe(self, outputs, show=None):
        with tempfile.TemporaryDirectory() as root:
            binary = Path(root) / 'bin/numactl'
            binary.parent.mkdir()
            binary.write_text('x')
            (Path(root) / 'path-bin').mkdir()
            (Path(root) / 'path-bin/numactl').symlink_to(binary)
            receipt = Path(root) / 'install.json'
            receipt.write_text(json.dumps({'numactl': str(binary), 'path_dir': str(Path(root) / 'path-bin'),
                                           'numactl_sha256': hashlib.sha256(b'x').hexdigest()}))
            calls = []

            def run(argv, **kwargs):
                calls.append(argv)
                node = 0 if '--membind=0 ' in argv[-1] else 1
                if argv[-1].endswith('numactl --show'):
                    out = (show or (lambda r, n: self.show(n)))(argv[0] == 'taskset', node)
                    return SimpleNamespace(returncode=0, stdout=out, stderr='')
                return SimpleNamespace(returncode=0, stdout=outputs(argv[0] == 'taskset', node), stderr='')
            return probe_numactl.probe(receipt, run), calls

    def test_modes(self):
        good = lambda restricted, node: self.output(node)
        result, calls = self.probe(good)
        self.assertEqual(result['mode'], 'restricted')
        self.assertEqual(sum(c[0] == 'taskset' and 'grep' in c[-1] for c in calls), 2)
        self.assertEqual(sum(c[0] == 'taskset' and c[-1].endswith('--show') for c in calls), 2)
        self.assertEqual(len(calls), 4)
        row = result['chains'][0]['commands'][0]
        self.assertEqual(row['show_stdout'], self.show(0))          # everything printed is recorded
        fallback = lambda restricted, node: self.output(node).replace('0-103', '22-25,74-77', 1) if restricted \
            else self.output(node)
        self.assertEqual(self.probe(fallback)[0]['mode'], 'fallback')
        self.assertEqual(self.probe(lambda r, n: self.output(n, inner_mems=str(n)))[0]['mode'], 'neither')
        wrong = lambda r, n: self.show(n, membind=1 - n)
        self.assertEqual(self.probe(good, show=wrong)[0]['mode'], 'neither')
        only_unrestricted = lambda r, n: self.show(n, membind=False) if r else self.show(n)
        self.assertEqual(self.probe(good, show=only_unrestricted)[0]['mode'], 'fallback')


class Install(unittest.TestCase):
    def test_hash_refusal_and_layout(self):
        with tempfile.TemporaryDirectory() as root:
            packages = Path(root) / 'pkgs'
            packages.mkdir()
            for name in install_numactl.PACKAGES:
                (packages / name).write_text('wrong')
            with self.assertRaises(RuntimeError):
                install_numactl.install(packages, Path(root) / 'prefix', extract=lambda *a, **k: None)

            def extract(path, dest_dir):
                Path(dest_dir, 'bin').mkdir(exist_ok=True)
                Path(dest_dir, 'lib').mkdir(exist_ok=True)
                Path(dest_dir, 'bin/numactl').write_text('bin')
                Path(dest_dir, 'lib/libnuma.so.1').write_text('lib')
            with patch.object(install_numactl, 'digest', side_effect=lambda p: install_numactl.PACKAGES.get(
                    Path(p).name, 'x')):
                install_numactl.install(packages, Path(root) / 'prefix', extract=extract)
            self.assertEqual(sorted(p.name for p in (Path(root) / 'prefix/path-bin').iterdir()), ['numactl'])
            with self.assertRaises(FileExistsError):
                install_numactl.install(packages, Path(root) / 'prefix', extract=extract)

    UNNORMALIZED_LDD = ('\tlinux-vdso.so.1 (0x00007ffd5a1f2000)\n'
                  '\tlibnuma.so.1 => /opt/bench/numactl/bin/../lib/libnuma.so.1 '
                  '(0x00007f3a1c600000)\n'
                  '\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f3a1c200000)\n'
                  '\t/lib64/ld-linux-x86-64.so.2 (0x00007f3a1c81c000)\n')

    def test_unnormalized_ldd_line(self):
        self.assertEqual(install_numactl.ldd_libnuma(self.UNNORMALIZED_LDD),
                         '/opt/bench/numactl/bin/../lib/libnuma.so.1')
        with self.assertRaises(RuntimeError):
            install_numactl.ldd_libnuma('\tlibnuma.so.1 => not found\n')
        with self.assertRaises(RuntimeError):
            install_numactl.ldd_libnuma(self.UNNORMALIZED_LDD.replace('libnuma', 'libother'))

    def make_prefix(self, root):
        prefix = Path(root) / 'numactl'
        (prefix / 'bin').mkdir(parents=True)
        (prefix / 'lib').mkdir()
        (prefix / 'bin/numactl').write_text('bin')
        (prefix / 'lib/libnuma.so.1.0.0').write_text('lib')
        (prefix / 'lib/libnuma.so.1').symlink_to('libnuma.so.1.0.0')
        (prefix / 'path-bin').mkdir()
        (prefix / 'path-bin/numactl').symlink_to('../bin/numactl')
        return prefix

    def test_receipt_with_unnormalized_ldd_path(self):
        with tempfile.TemporaryDirectory() as root:
            prefix = self.make_prefix(root)
            # The unnormalized ldd form (<prefix>/bin/../lib/libnuma.so.1), rooted at this prefix.
            ldd = self.UNNORMALIZED_LDD.replace('/opt/bench/numactl', str(prefix))
            run = lambda argv, **k: SimpleNamespace(returncode=0, stderr='',
                                                    stdout=ldd if argv[0] == 'ldd' else 'membind: 0 1 \n')
            data = install_numactl.receipt(prefix, {}, run=run)
            self.assertEqual(data['libnuma'], str((prefix / 'lib/libnuma.so.1.0.0').resolve()))
            # Another libnuma (system copy) or "not found" is refused.
            for bad in (ldd.replace(f'{prefix}/bin/../lib', '/usr/lib/x86_64-linux-gnu'),
                        ldd.replace(f'{prefix}/bin/../lib/libnuma.so.1 (0x00007f3a1c600000)', 'not found')):
                run = lambda argv, **k: SimpleNamespace(returncode=0, stderr='',
                                                        stdout=bad if argv[0] == 'ldd' else 'membind: 0 1 \n')
                with self.assertRaises(RuntimeError):
                    install_numactl.receipt(prefix, {}, run=run)

    def test_receipt_only_path_for_existing_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            packages = Path(root) / 'pkgs'
            packages.mkdir()
            for name in install_numactl.PACKAGES:
                (packages / name).write_text('pkg')
            prefix = self.make_prefix(root)

            def extract(path, dest_dir):
                if 'numactl-' in Path(path).name:
                    Path(dest_dir, 'bin').mkdir(exist_ok=True)
                    Path(dest_dir, 'bin/numactl').write_text('bin')
                else:
                    Path(dest_dir, 'lib').mkdir(exist_ok=True)
                    Path(dest_dir, 'lib/libnuma.so.1.0.0').write_text('lib')
                    Path(dest_dir, 'lib/libnuma.so.1').symlink_to('libnuma.so.1.0.0')
            fake = lambda p: install_numactl.PACKAGES.get(Path(p).name) or hashlib.sha256(
                Path(p).read_bytes()).hexdigest()
            with patch.object(install_numactl, 'digest', side_effect=fake):
                self.assertEqual(set(install_numactl.verify_existing(packages, prefix, extract=extract)),
                                 set(install_numactl.PACKAGES))
                self.assertEqual(sorted(p.name for p in Path(root).iterdir()), ['numactl', 'pkgs'])  # temp removed
                (prefix / 'lib/libnuma.so.1.0.0').write_text('changed')
                with self.assertRaises(RuntimeError):
                    install_numactl.verify_existing(packages, prefix, extract=extract)
                (prefix / 'lib/libnuma.so.1.0.0').write_text('lib')
                (prefix / 'lib/extra').write_text('x')
                with self.assertRaises(RuntimeError):
                    install_numactl.verify_existing(packages, prefix, extract=extract)
                (prefix / 'lib/extra').unlink()
                (prefix / 'path-bin/other').write_text('x')
                with self.assertRaises(RuntimeError):
                    install_numactl.verify_existing(packages, prefix, extract=extract)


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.small = b'{"config": 1}\n'
        self.big = bytes(range(256)) * 4      # 1024 bytes, 8 ranges of 128
        self.items = [
            {'rfilename': 'config.json', 'size': len(self.small),
             'blobId': hashlib.sha1(f'blob {len(self.small)}\0'.encode() + self.small).hexdigest()},
            {'rfilename': 'model-00001-of-00001.safetensors', 'size': len(self.big), 'blobId': 'x',
             'lfs': {'sha256': hashlib.sha256(self.big).hexdigest(), 'size': len(self.big)}}]
        self.metadata = self.root / 'meta.json'
        self.metadata.write_text(json.dumps({'id': MODEL, 'sha': REVISION, 'siblings': self.items}))
        self.patches = [patch.object(fetch_model, 'MODEL_FILES', 2),
                        patch.object(fetch_model, 'MODEL_SAFETENSORS_BYTES', len(self.big)),
                        patch.object(fetch_model, 'RANGE_THRESHOLD', 64)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def curl(self, data_by_name):
        def run(argv, check=False):
            start, end = map(int, argv[argv.index('--range') + 1].split('-'))
            name = argv[-1].rsplit('/', 1)[1]
            data = data_by_name[name]
            Path(argv[argv.index('--output') + 1]).write_bytes(data[start:end + 1])
            Path(argv[argv.index('--dump-header') + 1]).write_text(
                f'HTTP/1.1 206\r\ncontent-range: bytes {start}-{end}/{len(data)}\r\n')
            return SimpleNamespace(returncode=0)
        return run

    def download(self, data=None):
        data = data or {'config.json': self.small, 'model-00001-of-00001.safetensors': self.big}
        return fetch_model.download(self.metadata, self.root / 'hf', self.curl(data),
                                    lambda s: None, free=10 ** 12)

    def test_download_does_not_hash_weights_and_verify_publishes(self):
        lfs_hashes = []
        real = fetch_model.assemble_verified
        with patch.object(fetch_model, 'assemble_verified',
                          side_effect=lambda p: lfs_hashes.append(p.name) or real(p)):
            result = self.download()
        self.assertEqual([r['state'] for r in result['files']], ['verified-published', 'parts-complete-unhashed'])
        self.assertEqual(lfs_hashes, ['config.json'])
        paths = fetch_model.Paths(self.items[1], self.root / 'hf')
        self.assertEqual(len(paths.starts()), 8)
        self.assertFalse(paths.blob.exists())
        self.assertFalse(paths.target.exists())
        verified = fetch_model.verify(self.metadata, self.root / 'hf')
        self.assertEqual(len(verified['files']), 2)
        self.assertEqual(paths.target.read_bytes(), self.big)
        self.assertFalse(paths.parts.exists())

    def test_digest_mismatch_never_publishes(self):
        corrupt = bytearray(self.big)
        corrupt[500] ^= 1
        self.download({'config.json': self.small, 'model-00001-of-00001.safetensors': bytes(corrupt)})
        with self.assertRaises(RuntimeError):
            fetch_model.verify(self.metadata, self.root / 'hf')
        paths = fetch_model.Paths(self.items[1], self.root / 'hf')
        self.assertFalse(paths.blob.exists())
        self.assertTrue(paths.parts.exists())       # kept for inspection

    # Header shapes seen from a hub mirror: the 307 hop carries a
    # Content-Range, the final /api/resolve-cache/ hop is a 200 with the whole file.
    HOP_307 = ('HTTP/2 307 \r\ndate: Fri, 09 Oct 2026 08:12:31 GMT\r\ncontent-type: text/plain; charset=utf-8\r\n'
               'content-range: bytes {cr}\r\nlocation: /api/resolve-cache/models/{model}/{rev}/{name}\r\n\r\n')
    HOP_200 = ('HTTP/2 200 \r\ndate: Fri, 09 Oct 2026 08:12:31 GMT\r\ncontent-type: application/octet-stream\r\n'
               'content-length: {length}\r\naccept-ranges: bytes\r\n\r\n')
    HOP_206 = ('HTTP/2 206 \r\ncontent-type: application/octet-stream\r\ncontent-range: bytes {cr}\r\n'
               'content-length: {length}\r\n\r\n')

    def hops(self, *parts):
        return ''.join(parts)

    def test_judge_final_response_only(self):
        mirror = self.hops(self.HOP_307.format(cr='45805-79024/79025', model=MODEL, rev=REVISION, name='x'),
                           self.HOP_200.format(length=79025))
        self.assertEqual([h['status'] for h in fetch_model.response_hops(mirror)], [307, 200])
        # The mirror case: part already held, the 307's Content-Range must not count.
        self.assertEqual(fetch_model.judge_final(mirror, 45805, 79024, 79025, 6000, 79025), 'restart')
        # Nothing held yet and the whole file arrived: accepted whole.
        self.assertEqual(fetch_model.judge_final(mirror, 0, 79024, 79025, 0, 79025), 'whole')
        self.assertEqual(fetch_model.judge_final(mirror, 0, 79024, 79025, 0, 79000), 'restart')
        good = self.hops(self.HOP_307.format(cr='45805-79024/79025', model=MODEL, rev=REVISION, name='x'),
                         self.HOP_206.format(cr='45805-79024/79025', length=33220))
        self.assertEqual(fetch_model.judge_final(good, 45805, 79024, 79025, 6000, 33220), 'append')
        self.assertEqual(fetch_model.judge_final(good, 45805, 79024, 79025, 6000, 1000), 'append')   # cut short
        self.assertEqual(fetch_model.judge_final(good, 45805, 79024, 79025, 6000, 33221), 'discard')  # too long
        self.assertEqual(fetch_model.judge_final(good, 45000, 79024, 79025, 0, 100), 'discard')      # wrong start
        self.assertEqual(fetch_model.judge_final(good, 45805, 70000, 79025, 0, 100), 'discard')      # wrong end
        self.assertEqual(fetch_model.judge_final(good, 45805, 79024, 80000, 0, 100), 'discard')      # wrong total
        self.assertEqual(fetch_model.judge_final('', 0, 9, 10, 0, 10), 'discard')
        cont = 'HTTP/1.1 100 Continue\r\n\r\n' + self.HOP_206.format(cr='0-9/10', length=10)
        self.assertEqual(fetch_model.judge_final(cont, 0, 9, 10, 0, 10), 'append')
        failed = self.hops(self.HOP_307.format(cr='0-9/10', model=MODEL, rev=REVISION, name='x'),
                           'HTTP/2 502 \r\n\r\n')
        self.assertEqual(fetch_model.judge_final(failed, 0, 9, 10, 0, 0), 'discard')

    def range_paths(self):
        paths = fetch_model.Paths(self.items[1], self.root / 'hf')
        paths.make_dirs()
        paths.parts.mkdir()
        return paths

    def scripted_curl(self, answers, calls):
        """answers: one function per pass, (first, end) -> (header text, body bytes)."""
        def run(argv, check=False):
            first, end = map(int, argv[argv.index('--range') + 1].split('-'))
            calls.append((first, end))
            headers, body = answers.pop(0)(first, end)
            Path(argv[argv.index('--dump-header') + 1]).write_text(headers)
            Path(argv[argv.index('--output') + 1]).write_bytes(body)
            return SimpleNamespace(returncode=0)
        return run

    def test_range_ignored_after_redirect_restarts_the_part(self):
        paths, calls, size = self.range_paths(), [], len(self.big)
        redirect = lambda first, end: self.HOP_307.format(cr=f'{first}-{end}/{size}', model=MODEL, rev=REVISION,
                                                          name=paths.name)
        answers = [
            # pass 1: a short 206 (cut by --max-time): 40 of the 128 bytes held
            lambda f, e: (redirect(f, e) + self.HOP_206.format(cr=f'{f}-{e}/{size}', length=e - f + 1),
                          self.big[f:f + 40]),
            # pass 2: the mirror case, 307 with Content-Range, then 200 with the whole file
            lambda f, e: (redirect(f, e) + self.HOP_200.format(length=size), self.big),
            # pass 3: from the part's beginning again, a correct 206
            lambda f, e: (redirect(f, e) + self.HOP_206.format(cr=f'{f}-{e}/{size}', length=e - f + 1),
                          self.big[f:e + 1])]
        part = fetch_model.fetch_range(paths, 640, self.scripted_curl(answers, calls), lambda s: None)
        self.assertEqual(calls, [(640, 767), (680, 767), (640, 767)])
        self.assertEqual(part.read_bytes(), self.big[640:768])

    def test_leftover_chunk_is_never_appended(self):
        # A chunk left by an earlier (killed) pass, then a pass that writes a
        # correct 206 header but no body: nothing is appended.
        paths, calls, size = self.range_paths(), [], len(self.big)
        (paths.parts / '640.chunk').write_bytes(self.big[640:768])

        def run(argv, check=False):
            first, end = map(int, argv[argv.index('--range') + 1].split('-'))
            calls.append((first, end))
            Path(argv[argv.index('--dump-header') + 1]).write_text(
                self.HOP_206.format(cr=f'{first}-{end}/{size}', length=end - first + 1))
            return SimpleNamespace(returncode=28)          # curl timed out before any body
        with self.assertRaises(RuntimeError):
            fetch_model.fetch_range(paths, 640, run, lambda s: None)
        self.assertEqual(len(calls), 10)
        self.assertEqual(set(calls), {(640, 767)})
        self.assertFalse(paths.part(640).exists())
        self.assertFalse((paths.parts / '640.chunk').exists())

    def creeping_curl(self, calls, step, stall_after=None):
        """Every pass a correct 206 carrying `step` bytes; from pass `stall_after` on,
        passes time out with no body (a stall)."""
        size = len(self.big)

        def run(argv, check=False):
            first, end = map(int, argv[argv.index('--range') + 1].split('-'))
            calls.append((first, end))
            Path(argv[argv.index('--dump-header') + 1]).write_text(
                self.HOP_206.format(cr=f'{first}-{end}/{size}', length=end - first + 1))
            if stall_after is None or len(calls) < stall_after:
                Path(argv[argv.index('--output') + 1]).write_bytes(self.big[first:min(first + step, end + 1)])
            return SimpleNamespace(returncode=28)
        return run

    def test_slow_range_completes_beyond_ten_passes(self):
        paths, calls = self.range_paths(), []
        part = fetch_model.fetch_range(paths, 640, self.creeping_curl(calls, 5), lambda s: None)
        self.assertEqual(len(calls), 26)                     # 128 bytes, 5 per pass
        self.assertEqual(part.read_bytes(), self.big[640:768])

    def test_stall_counter_resets_on_progress_then_ten_stalls_raise(self):
        paths, calls = self.range_paths(), []
        # Progress on passes 1-11, then nothing: 10 more passes, then the stop.
        with self.assertRaisesRegex(RuntimeError, '10 consecutive passes without progress'):
            fetch_model.fetch_range(paths, 640, self.creeping_curl(calls, 5, stall_after=12), lambda s: None)
        self.assertEqual(len(calls), 21)
        self.assertEqual(paths.part(640).read_bytes(), self.big[640:695])     # 11 passes x 5 bytes kept

    def test_pass_ceiling(self):
        paths, calls = self.range_paths(), []
        with patch.object(fetch_model, 'MAX_PASSES', 15), \
                self.assertRaisesRegex(RuntimeError, 'pass ceiling 15'):
            fetch_model.fetch_range(paths, 640, self.creeping_curl(calls, 1), lambda s: None)
        self.assertEqual(len(calls), 15)
        self.assertEqual(paths.part(640).stat().st_size, 15)

    def test_whole_file_200_with_nothing_held(self):
        paths, calls, size = self.range_paths(), [], len(self.big)
        answers = [lambda f, e: (self.HOP_307.format(cr=f'{f}-{e}/{size}', model=MODEL, rev=REVISION,
                                                     name=paths.name) + self.HOP_200.format(length=size), self.big)]
        part = fetch_model.fetch_range(paths, 256, self.scripted_curl(answers, calls), lambda s: None)
        self.assertEqual(calls, [(256, 383)])
        self.assertEqual(part.read_bytes(), self.big[256:384])     # this part's bytes of the whole body

    def test_verify_refuses_missing_download(self):
        with self.assertRaises(RuntimeError):
            fetch_model.verify(self.metadata, self.root / 'hf')

    def test_endpoint_names_the_pinned_revision(self):
        urls = []
        data = {'config.json': self.small, 'model-00001-of-00001.safetensors': self.big}
        curl = self.curl(data)
        fetch_model.download(self.metadata, self.root / 'hf', lambda argv, check=False: urls.append(argv[-1])
                             or curl(argv, check), lambda s: None, free=10 ** 12, endpoint='https://mirror.example/')
        self.assertIn(f'https://mirror.example/{MODEL}/resolve/{REVISION}/config.json', urls)

    def test_metadata_pins(self):
        self.metadata.write_text(json.dumps({'id': MODEL, 'sha': '0' * 40, 'siblings': self.items}))
        with self.assertRaises(RuntimeError):
            fetch_model.load_metadata(self.metadata)


class Banks(unittest.TestCase):
    def bank(self):
        banks, total = construct.bank_layout()
        rows = [{'id': i, 'first_text_token': i, 'token_ids': [i] + [0] * 1535, 'prompt_tokens': 1536,
                 'messages': construct.messages(str(i))} for i in range(total)]
        gate_e = {'sha256': construct.digest(construct.GATE_E_FILE),
                  'prompts': [{**p, 'prompt_tokens': 40} for p in runner.gate_e_reference()]}
        return {'model': MODEL, 'model_revision': REVISION, 'source_commit': COMMIT, 'banks': banks,
                'requests': rows, 'gate_e_non_english': gate_e}

    def test_layout_and_loader(self):
        banks, total = construct.bank_layout()
        self.assertEqual(total, 2 * sum(BANK_MEASURED[c] + 2 * c for c in BANK_LEVELS) + BURST_REQUESTS)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'prompts.json'
            path.write_text(json.dumps(self.bank()))
            bank = runner.load_prompts(path)
            items = runner.gate_e_items(bank)
            self.assertEqual(len(items), 35)
            self.assertEqual([i['id'] for i in items[-5:]], ['ne-zh', 'ne-ja', 'ne-ko', 'ne-ru', 'ne-ar'])
            warm, measured = runner.level_rows(bank, 32, 1)
            self.assertEqual((len(warm), len(measured)), (64, 160))
            # c = 16 has 160 measured after a warm-up of 2c = 32.
            self.assertEqual(MEASURED, {1: 30, 8: 80, 16: 160, 32: 160})
            warm, measured = runner.level_rows(bank, 16, 0)
            self.assertEqual((len(warm), len(measured)), (32, 160))
            data = self.bank()
            data['gate_e_non_english']['prompts'][2]['messages'][1]['content'] += ' '      # a changed prompt
            path.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                runner.load_prompts(path)

    def test_gate_e_file_is_ascii_and_has_five_non_english_prompts(self):
        raw = construct.GATE_E_FILE.read_bytes()
        raw.decode('ascii')
        prompts = json.loads(raw)['prompts']
        self.assertEqual(len(prompts), 5)
        self.assertTrue(all(re.search(r'[^\x00-\x7f]', p['text']) for p in prompts))


class FreeSpace(Fetch):
    def test_already_downloaded_bytes_are_not_required_again(self):
        self.download()
        total = sum(i['size'] for i in self.items)
        present = sum(fetch_model.present_bytes(i, self.root / 'hf') for i in self.items)
        self.assertEqual(present, total)
        # A rerun needs only the margin now; with less free space than the full size it still proceeds.
        fetch_model.download(self.metadata, self.root / 'hf', self.curl({}), lambda s: None,
                             free=fetch_model.SPACE_MARGIN)
        with self.assertRaises(RuntimeError):
            fetch_model.download(self.metadata, self.root / 'hf', self.curl({}),
                                 lambda s: None, free=fetch_model.SPACE_MARGIN - 1)


class ContentDigest(unittest.TestCase):
    def test_digest_ignores_paths_and_layout(self):
        artifact = Banks().bank()
        artifact['gate_e_non_english']['file'] = '/somewhere/gate_e_prompts.json'
        first = construct.content_digest(artifact)
        artifact['gate_e_non_english']['file'] = 'gate_e_prompts.json'
        artifact['tokenizer_files'] = {'tokenizer.json': 'x'}
        self.assertEqual(construct.content_digest(artifact), first)
        artifact['requests'][0]['token_ids'][1] = 7
        self.assertNotEqual(construct.content_digest(artifact), first)


if __name__ == '__main__':
    unittest.main()
