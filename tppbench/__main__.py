"""Command line: python -m tppbench <command> ... (see --help of each).

Setup:   numactl-install, numactl-probe, fetch, prompts, verify-prompts
Phases:  start-test, settings, transport, timed (run in this order; each needs the receipts
         of the ones before it)
Reading: report
"""
import argparse
import json
from pathlib import Path
import sys

from . import spec


def paths(p):
    p.add_argument('--data-dir', type=Path,
                   help='setup files, prompts, caches and locks (default $TPPB_DATA_DIR or ./tppb-data)')
    p.add_argument('--model', type=Path,
                   help='local snapshot of the pinned model (default $TPPB_MODEL or the snapshot under $HF_HOME)')


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tppbench', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('numactl-install', help='extract numactl 2.0.18 from the two pinned conda-forge packages')
    q.add_argument('--packages', type=Path, required=True, help='directory holding the two .conda files')
    q.add_argument('--prefix', type=Path, help='install prefix (default <data-dir>/numactl)')
    paths(q)
    q = sub.add_parser('numactl-probe', help='CPU-only check that the server\'s numactl binding chain works here')
    paths(q)
    q = sub.add_parser('fetch', help='download (8 byte ranges per file) or verify the pinned model files')
    q.add_argument('mode', choices=('download', 'verify'))
    q.add_argument('--metadata', type=Path, required=True,
                   help='hub API JSON of the pinned revision (README, "Model")')
    q.add_argument('--hf-home', type=Path, required=True, help='HuggingFace cache root to fill')
    q.add_argument('--endpoint', default='https://huggingface.co', help='hub or mirror base URL')
    paths(q)
    q = sub.add_parser('prompts', help='build the prompt file from a vLLM v0.30.0 checkout (CPU only)')
    q.add_argument('--vllm-source', type=Path, required=True)
    q.add_argument('--output', type=Path, help='default <data-dir>/prompts.json')
    paths(q)
    q = sub.add_parser('verify-prompts', help='independent replay of every stored excerpt and token list')
    q.add_argument('--vllm-source', type=Path, required=True)
    q.add_argument('--prompts', type=Path, help='default <data-dir>/prompts.json')
    q.add_argument('--output', type=Path, help='default <data-dir>/setup/prompts-verified.json')
    paths(q)
    from . import runner
    for phase in runner.PHASES:
        runner.add_arguments(sub.add_parser(phase, help=f'phase {phase} (whole machine; README "Phases")'))
    q = sub.add_parser('report', help='recompute every reported number of a timed run from its raw records')
    q.add_argument('run', type=Path, help='output directory of the timed phase')
    q.add_argument('--json', type=Path, help='also write the numbers as JSON')
    q.add_argument('--statistic', choices=('median', 'mean'), default='median',
                   help='summary over the five rounds in the reading table and mean TPOT (default median); '
                        'verdicts use per-round ratios and do not depend on it')
    args = p.parse_args(argv)

    if args.cmd == 'report':
        from . import report
        rep = report.report(args.run, args.statistic)
        print(report.text_report(rep))
        if args.json:
            args.json.write_text(json.dumps(rep, indent=1, default=str) + '\n')
        return 2 if rep.get('error') else 1 if rep['warnings'] else 0
    if args.cmd in runner.PHASES:
        return 0 if runner.main(args.cmd, args) else 1
    conf = spec.configure(args.data_dir, args.model)
    if args.cmd == 'numactl-install':
        from . import install_numactl
        install_numactl.cli(args.packages, args.prefix)
        return 0
    if args.cmd == 'numactl-probe':
        from . import probe_numactl
        return 0 if probe_numactl.cli() else 1
    if args.cmd == 'fetch':
        from . import fetch_model
        fetch_model.cli(args.mode, args.metadata, args.hf_home, conf['data'] / 'setup/MODEL_READY.json',
                        args.endpoint)
        return 0
    if args.cmd == 'prompts':
        from . import construct
        construct.build_file(args.vllm_source, conf['model'], args.output or conf['data'] / 'prompts.json')
        return 0
    if args.cmd == 'verify-prompts':
        from . import verify_prompts
        result = verify_prompts.verify(args.prompts or conf['data'] / 'prompts.json', args.vllm_source,
                                       conf['model'])
        output = args.output or conf['data'] / 'setup/prompts-verified.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        spec.write_json(output, result)
        print(json.dumps(result))
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
