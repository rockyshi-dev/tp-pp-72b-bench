"""Every reported number of a timed run, recomputed from its raw records.

Input: the output directory of the `timed` phase. For every accepted round, arm and level,
the per-request records (`measured/client-*.jsonl`) and the `/metrics` snapshots taken
before and after the level are read again; each request's TTFT, token gap and mean TPOT are
recomputed from its timestamps, and gates K, T, C and G and the level's values from those
(`gates.level`, the same function the run used), combined as the run combines them (gate K
equality per round, level retries). Gates N, Q and M depend on placement and
telemetry evidence and are taken from the stored cell. The decision (Tables A and B) is
recomputed with `gates.decision`. Every recomputed value, gate and verdict is compared with
what the run stored; a difference is a warning (exit code 1).

Printed: the reading table (per level, metric and arm: the median, or with `statistic='mean'`
the mean, over the five rounds, or the five-round range when it spans 2% or more), mean TPOT, every pairwise verdict with its
five ratios' range, Table A (TP8 against PP8: verdicts, row, throughput clause), Table B
(cell winners, row), and a gate summary over the accepted cells.
"""
import copy
import json
from pathlib import Path
import re
import statistics

from . import client
from . import gates
from .spec import ARMS, MEASURED, ROUNDS, capacity_needed, levels

METRIC_LABEL = {'ttft_s': ('median TTFT', 's', 1.0), 'token_gap_s': ('median token gap', 'ms', 1e3),
                'output_tokens_per_second': ('output throughput', 'tok/s', 1.0)}
RANGE_SPREAD = 0.02
ROUND_DIR = re.compile(r'round-\d+-attempt-\d+')


DECIMALS = {'s': 2, 'ms': 1, 'tok/s': 1}


def fixed(x, unit):
    return 'n/a' if x is None else f'{x:.{DECIMALS[unit]}f}'


def cell_directory(run, stored):
    """The cell's directory under `run`: the stored path may come from another machine, so
    only its components from the `round-K-attempt-N` directory on are kept."""
    parts = Path(stored).parts
    for i, part in enumerate(parts):
        if ROUND_DIR.fullmatch(part):
            return Path(run, *parts[i:])
    raise ValueError(f'Cell directory without a round-K-attempt-N component: {stored}')


def part_directory(round_dir, arm, retry=False):
    """The directory of one arm's part of a round, as the timed phase names it: `<arm>` or,
    for a level retry, `<arm>-level-retry`; with `-m-rerun` appended when gate M made the
    part run a second time (the second run is the one that counts)."""
    base = Path(round_dir) / (arm + ('-level-retry' if retry else ''))
    rerun = base.parent / (base.name + '-m-rerun')
    return rerun if rerun.is_dir() else base


def read_rows(directory):
    rows = []
    for path in sorted((directory / 'measured').glob('client-*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                for key in ('content_chunks', 'ttft_s', 'token_gap_s', 'mean_tpot_s'):
                    row.pop(key, None)
                rows.append(client.summarize(row))   # recomputed from send_ns and content_ns
    return rows


def recompute_dir(directory, c, step):
    rows = read_rows(directory)
    before = (directory / 'metrics-before.txt').read_text()
    after = (directory / 'metrics-after.txt').read_text()
    startup = (directory.parent / 'metrics-startup.txt').read_text()
    return gates.level(rows, before, after, startup, MEASURED[c], capacity_needed(step))


COMPARED_VALUES = ('ttft_s', 'token_gap_s', 'output_tokens_per_second', 'mean_tpot_s', 'kv_capacity_tokens',
                   'dropped', 'chunk_complete', 'prefix_queries', 'prefix_hits', 'preemptions')
COMPARED_GATES = ('K', 'T', 'C', 'G')


def close(a, b):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def recompute_round(run, stored_round, step, arms, warnings):
    """One accepted round, combined exactly as the timed phase combines it: every arm's part
    (all levels), K equality over those cells; a level is retried when K holds in every arm and
    T or C fails in some arm; then K equality over original and retry cells together, each
    replacement's K ANDed with its original's, and a replacement's G forced false when the
    original passed T and C but failed G."""
    levels_ = list(stored_round['levels'])
    any_cell = next(iter(stored_round['levels'][levels_[0]].values()))
    round_dir = cell_directory(run, any_cell['directory']).parent.parent
    where = f'round {stored_round["round"]}'
    parts = {arm: {k: recompute_dir(part_directory(round_dir, arm) / f'c{k}', int(k), step) for k in levels_}
             for arm in arms}
    gates.round_capacity_k(parts)
    computed_retry = [k for k in levels_ if all(parts[a][k]['K'] for a in arms)
                      and any(parts[a][k]['rerun_level'] for a in arms)]
    stored_retry = [k for k in levels_ if any('-level-retry' in Path(stored_round['levels'][k][a]['directory']).parent.name
                                               for a in arms)]
    if computed_retry != stored_retry:
        warnings.append(f'{where}: level retry recomputed at {computed_retry}, the run retried {stored_retry}')
    if stored_retry:
        more = {arm: {k: recompute_dir(part_directory(round_dir, arm, retry=True) / f'c{k}', int(k), step)
                      for k in stored_retry} for arm in arms}
        gates.round_capacity_k({**parts, **{a + '-retry': more[a] for a in arms}})
        for arm in arms:
            for k, replacement in more[arm].items():
                original = parts[arm][k]
                replacement['K'] = original['K'] and replacement['K']
                if original['T'] and original['C'] and not original['G']:
                    replacement['G'] = False
            parts[arm].update(more[arm])
    new = {'round': stored_round['round'], 'attempt': stored_round['attempt'],
           'order': stored_round.get('order'), 'levels': {}}
    for k in levels_:
        new['levels'][k] = {}
        for arm in arms:
            cell, stored = parts[arm][k], stored_round['levels'][k][arm]
            for key in ('N', 'Q', 'M', 'quiet', 'placement', 'directory'):
                if key in stored:
                    cell[key] = stored[key]
            new['levels'][k][arm] = cell
    return new


def recompute_rounds(run, accepted, step, arms):
    """(rounds with recomputed cells, warnings)."""
    rounds, warnings = [], []
    for stored_round in accepted:
        new = recompute_round(run, stored_round, step, arms, warnings)
        for key in new['levels']:
            for arm in arms:
                cell, stored = new['levels'][key][arm], stored_round['levels'][key][arm]
                where = f'round {new["round"]} c={key} {arm}'
                for name in COMPARED_VALUES:
                    if name in stored and not close(cell.get(name), stored.get(name)):
                        warnings.append(f'{where}: {name} recomputed {cell.get(name)!r}, stored {stored.get(name)!r}')
                for name in COMPARED_GATES:
                    if name in stored and bool(cell[name]) != bool(stored[name]):
                        warnings.append(f'{where}: gate {name} recomputed {cell[name]}, stored {stored[name]}')
        rounds.append(new)
    return rounds, warnings


def per_round(rounds, arm, c, metric):
    return [r['levels'][str(c)][arm][metric] for r in rounds]


STATISTICS = {'median': statistics.median, 'mean': statistics.fmean}


def reading_value(values, scale, unit, statistic='median'):
    if any(v is None for v in values):
        return {'text': 'n/a', 'value': None, 'min': None, 'max': None, 'range_shown': False}
    lo, hi, central = min(values), max(values), STATISTICS[statistic](values)
    shown = lo > 0 and (hi - lo) / lo >= RANGE_SPREAD
    text = f'{fixed(lo * scale, unit)}-{fixed(hi * scale, unit)}' if shown else fixed(central * scale, unit)
    return {'text': text, 'value': central * scale, 'min': lo * scale, 'max': hi * scale, 'range_shown': shown}


def compare_decision(new, stored):
    warnings = []
    if stored is None:
        return ['no stored decision-for-review.json']
    for table in ('table_a', 'table_b'):
        a, b = (new.get(table) or {}), (stored.get(table) or {})
        if a.get('row') != b.get('row'):
            warnings.append(f'{table} row recomputed {a.get("row")}, stored {b.get("row")}')
    a, b = new.get('table_a') or {}, stored.get('table_a') or {}
    for key in ('throughput_clause_contradicted', 'docs_correction', 'confirms_docs'):
        if a.get(key) != b.get(key):
            warnings.append(f'table A {key} recomputed {a.get(key)}, stored {b.get(key)}')
    if (new.get('table_b') or {}).get('cell_winners') != (stored.get('table_b') or {}).get('cell_winners'):
        warnings.append('table B cell winners differ from the stored ones')
    if new.get('pair_verdicts') != stored.get('pair_verdicts'):
        warnings.append('pairwise verdicts differ from the stored ones')
    for pair, cells in (new.get('pair_ratios') or {}).items():
        for key, ratios in cells.items():
            old = (stored.get('pair_ratios') or {}).get(pair, {}).get(key)
            if old is None or len(old) != len(ratios) or not all(close(x, y) for x, y in zip(ratios, old)):
                warnings.append(f'ratios {pair} {key} differ from the stored ones')
    return warnings


def gate_summary(rounds, arms):
    cells = [r['levels'][k][a] for r in rounds for k in r['levels'] for a in arms]
    quiet = [c.get('quiet') or {} for c in cells]
    foreign = [v for q in quiet for v in (q.get('foreign_cpu_fraction') or {}).values() if v is not None]
    intervals = [(q.get('telemetry_gaps') or {}).get('max_interval_s') for q in quiet]
    intervals = [v for v in intervals if v is not None]
    return {'cells': len(cells),
            'pass': {g: sum(bool(c.get(g)) for c in cells) for g in ('K', 'T', 'C', 'G', 'N', 'Q', 'M')},
            'kv_capacity_tokens': sorted({c['kv_capacity_tokens'] for c in cells}),
            'preemptions': sum(c['preemptions'] for c in cells),
            'dropped': sum(c['dropped'] for c in cells),
            'prefix_hits': sum(c['prefix_hits'] for c in cells),
            'chunk_complete_fraction_min': min(c['chunk_complete'] / c['total'] for c in cells) if cells else None,
            'foreign_cpu_fraction_max': max(foreign) if foreign else None,
            'telemetry_interval_max_s': max(intervals) if intervals else None}


def report(run, statistic='median'):
    run = Path(run)
    if statistic not in STATISTICS:
        raise ValueError(statistic)
    out = {'run': str(run), 'statistic': statistic, 'warnings': [], 'error': None}
    accepted = json.loads((run / 'accepted-rounds.json').read_text())
    stored_decision = None
    if (run / 'decision-for-review.json').exists():
        stored_decision = json.loads((run / 'decision-for-review.json').read_text())
    study = (stored_decision or {}).get('study') or json.loads((run / 'run-manifest.json').read_text())['study']
    study = copy.deepcopy(study)
    numbers = [r['round'] for r in accepted]
    if sorted(numbers) != list(range(1, ROUNDS + 1)) or study.get('terminal_gate'):
        out['error'] = (f'accepted rounds {numbers}, terminal gate {study.get("terminal_gate")}: '
                        f'not a complete run of {ROUNDS} rounds; no tables')
        return out
    arms = [a for a in ARMS if a in study['arms']]
    step = study['step']
    rounds, warnings = recompute_rounds(run, accepted, step, arms)
    out['warnings'] += warnings
    decision = gates.decision(study, rounds)
    out['warnings'] += compare_decision(decision, stored_decision)
    winners = (decision.get('table_b') or {}).get('cell_winners') or {}
    reading = []
    for c in levels(step):
        for metric, (label, unit, scale) in METRIC_LABEL.items():
            if metric == 'output_tokens_per_second' and c == 1:
                continue
            row = {'c': c, 'metric': metric, 'label': label, 'unit': unit,
                   'winner': winners.get(f'{metric}:c{c}'), 'arms': {}}
            for arm in arms:
                row['arms'][arm] = reading_value(per_round(rounds, arm, c, metric), scale, unit, statistic)
                row['arms'][arm]['rounds'] = [None if v is None else v * scale
                                              for v in per_round(rounds, arm, c, metric)]
            reading.append(row)
    tpot = {c: {arm: reading_value(per_round(rounds, arm, c, 'mean_tpot_s'), 1e3, 'ms', statistic) for arm in arms}
            for c in levels(step)}
    out.update(step=step, arms=arms, rounds=[{'round': r['round'], 'attempt': r['attempt']} for r in rounds],
               reading=reading, mean_tpot_ms=tpot, decision=decision, gates=gate_summary(rounds, arms))
    return out


def text_report(rep):
    if rep.get('error'):
        return 'ERROR: ' + rep['error']
    arms, lines = rep['arms'], []
    lines.append(f'Run {rep["run"]}: ladder step {rep["step"]}, arms {", ".join(arms)}, rounds ' +
                 ', '.join(f'{r["round"]} (attempt {r["attempt"]})' for r in rep['rounds']))
    lines.append('')
    lines.append(f'Reading ({rep["statistic"]} over the five rounds; the five-round range where the rounds '
                 f'span 2% or more; * = cell winner)')
    lines.append('| c | Metric | ' + ' | '.join(arms) + ' |')
    lines.append('|---|---|' + '---|' * len(arms))
    for row in rep['reading']:
        cells = [row['arms'][a]['text'] + (' *' if row['winner'] == a else '') for a in arms]
        lines.append(f'| {row["c"]} | {row["label"]} ({row["unit"]}) | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('Mean TPOT (ms; per request (last - first content chunk) / (completion tokens - 1), '
                 f'median over requests, then {rep["statistic"]} over rounds)')
    for c, by_arm in rep['mean_tpot_ms'].items():
        lines.append(f'  c = {c}: ' + ', '.join(f'{a} {by_arm[a]["text"]}' for a in arms))
    d = rep['decision']
    lines.append('')
    lines.append('Pairwise verdicts (ratio range over the five rounds; r < 1 favours the first arm)')
    for pair, cells in d['pair_verdicts'].items():
        for key, verdict in cells.items():
            r = d['pair_ratios'][pair][key]
            lines.append(f'  {pair:16s} {key:32s} {verdict:24s} {min(r):.3f}-{max(r):.3f}')
    a = d['table_a']
    lines.append('')
    lines.append(f'Table A (TP8 against PP8): row {a["row"]}' +
                 ''.join(f', {k} = {v}' for k, v in a.items() if k not in ('row', 'verdicts')))
    for key, verdict in a['verdicts'].items():
        lines.append(f'  {key:32s} {verdict}')
    b = d['table_b']
    lines.append('')
    lines.append(f'Table B (cell winners): row {b["row"]}')
    for key, winner in b['cell_winners'].items():
        lines.append(f'  {key:32s} {winner or "no winner"}')
    lines.append('')
    if d['against_expectation']:
        lines.append('Matches the expected outcome stated in the README: no (' + '; '.join(d['expectation_check']) + ')')
    else:
        lines.append('Matches the expected outcome stated in the README: yes')
    g = rep['gates']
    lines.append('')
    lines.append(f'Gates over {g["cells"]} accepted cells: ' +
                 ', '.join(f'{k} {v}/{g["cells"]}' for k, v in g['pass'].items()))
    lines.append(f'  KV capacity tokens {g["kv_capacity_tokens"]}, preemptions {g["preemptions"]}, '
                 f'dropped {g["dropped"]}, prefix hits {g["prefix_hits"]}, '
                 f'lowest 128-chunk fraction {g["chunk_complete_fraction_min"]}, '
                 f'largest foreign CPU fraction {g["foreign_cpu_fraction_max"]}, '
                 f'largest telemetry interval {g["telemetry_interval_max_s"]} s')
    lines.append('')
    if rep['warnings']:
        lines.append(f'WARNINGS ({len(rep["warnings"])}): recomputed values differ from the stored ones')
        lines += ['  ' + w for w in rep['warnings']]
    else:
        lines.append('Every recomputed value, gate and verdict equals the stored one.')
    return '\n'.join(lines)
