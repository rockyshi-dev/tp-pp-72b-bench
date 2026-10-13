"""Closed-loop SSE collector and gate E's sequential requests.

One client process per eight concurrent streams, pinned to the client CPUs; no
vLLM imports. Gate E requests are non-streaming, sent one at a time, each on a
fresh connection.
"""
import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time
import urllib.request

from .spec import GATE_E_REPETITION_PENALTY, MODEL, OUTPUT_TOKENS

CONCURRENCY = (1, 8, 16, 32)


def payload(row):
    return {'model': MODEL, 'messages': row['messages'], 'max_tokens': OUTPUT_TOKENS,
            'min_tokens': OUTPUT_TOKENS, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}}


def gate_e_payload(messages):
    """Gate E: non-streaming, temperature 0, max_tokens = min_tokens =
    128, logprobs with the top 5. `return_tokens_as_token_ids` makes every token
    field "token_id:<id>" (vllm/entrypoints/generate/base/serving.py:330-345) so
    tokens are compared by identity; decoded strings can be equal for different
    byte-level tokens of non-English text; that field changes no sampling
    parameter.

    `repetition_penalty: 1.0` is sent explicitly, so the model's
    generation_config default (1.05) does not apply (an explicit value is kept,
    chat_completion/protocol.py:678-682). With temperature 0 and no penalty the
    emitted token is the argmax of the same logits whose raw logprobs are
    returned. This applies to gate E requests only: `payload` (start-test,
    transport and timed requests) leaves it unset."""
    return {'model': MODEL, 'messages': messages, 'max_tokens': OUTPUT_TOKENS,
            'min_tokens': OUTPUT_TOKENS, 'temperature': 0, 'stream': False,
            'logprobs': True, 'top_logprobs': 5, 'return_tokens_as_token_ids': True,
            'repetition_penalty': GATE_E_REPETITION_PENALTY}


def summarize(row):
    stamps = row['content_ns']
    row['content_chunks'] = len(stamps)
    row['ttft_s'] = (stamps[0] - row['send_ns']) / 1e9 if stamps else None
    row['token_gap_s'] = (statistics.median(b-a for a, b in zip(stamps, stamps[1:])) / 1e9
                          if len(stamps) == OUTPUT_TOKENS else None)
    n = (row.get('usage') or {}).get('completion_tokens', 0)
    row['mean_tpot_s'] = ((stamps[-1] - stamps[0]) / 1e9 / (n-1)
                          if len(stamps) > 1 and n > 1 else None)
    return row


async def request(session, url, prompt):
    row = {'id': prompt['id'], 'send_ns': time.monotonic_ns(), 'content_ns': [],
           'contents': [], 'usage': None, 'finish_reason': None, 'done': False, 'error': None}
    try:
        async with session.post(url + '/v1/chat/completions', json=payload(prompt)) as response:
            row['status'] = response.status
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:1000]}')
            data = []
            async for line in response.content:
                line = line.decode('utf-8').rstrip('\r\n')
                if line.startswith('data:'):
                    data.append(line[5:].lstrip())
                elif line == '' and data:
                    stamp = time.monotonic_ns()
                    event = '\n'.join(data)
                    data = []
                    if event == '[DONE]':
                        row['done'] = True
                        break
                    chunk = json.loads(event)
                    if chunk.get('error'):
                        raise RuntimeError(str(chunk['error']))
                    if chunk.get('usage') is not None:
                        row['usage'] = chunk['usage']
                    choices = chunk.get('choices', [])
                    if choices:
                        if len(choices) != 1 or choices[0]['index'] != 0:
                            raise RuntimeError('Unexpected choices')
                        choice = choices[0]
                        content = choice.get('delta', {}).get('content')
                        if content:
                            row['content_ns'].append(stamp)
                            row['contents'].append(content)
                        if choice.get('finish_reason') is not None:
                            row['finish_reason'] = choice['finish_reason']
            if not row['done'] or row['usage'] is None or not row['content_ns']:
                raise RuntimeError('Incomplete stream: missing DONE, usage, or text')
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['end_ns'] = time.monotonic_ns()
    return summarize(row)


def worker(url, streams, cpus, barrier, output):
    os.sched_setaffinity(0, cpus)

    async def run():
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=1800)
        connector = aiohttp.TCPConnector(limit=8, keepalive_timeout=1.0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            barrier.wait(timeout=90)
            with open(output, 'x') as dest:
                async def loop(items):
                    for row in items:
                        result = await request(session, url, row)
                        dest.write(json.dumps(result, allow_nan=False) + '\n')
                        dest.flush()
                await asyncio.gather(*(loop(items) for items in streams))
    asyncio.run(run())


def split(prompts, concurrency, cpus):
    """Stream i gets prompts[i::c]; process k owns streams 8k..8k+7 and an equal
    share of the eight client CPUs."""
    if concurrency not in CONCURRENCY or len(cpus) != 8 or len(prompts) % concurrency:
        raise ValueError('Wrong concurrency, client CPU count, or unbalanced closed loop')
    count = max(1, concurrency // 8)
    plan = []
    for index in range(count):
        streams = [prompts[i::concurrency] for i in range(index*8, min((index+1)*8, concurrency))]
        plan.append((streams, list(cpus[index*(8//count):(index+1)*(8//count)])))
    return plan


def phase(url, prompts, concurrency, cpus, directory):
    """Fresh workers synchronize before sending; each worker owns a fixed sequence."""
    plan = split(prompts, concurrency, cpus)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(len(plan))
    processes = []
    try:
        for index, (streams, assigned) in enumerate(plan):
            proc = ctx.Process(target=worker, args=(url, streams, assigned, barrier,
                                                   directory / f'client-{index}.jsonl'))
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()
        if any(proc.exitcode != 0 for proc in processes):
            raise RuntimeError('Client process failed; level has incomplete evidence')
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()
    rows = [json.loads(line) for p in sorted(directory.glob('*.jsonl')) for line in p.read_text().splitlines()]
    if sorted(r['id'] for r in rows) != sorted(r['id'] for r in prompts):
        raise RuntimeError('Missing or duplicate requests')
    return rows


def gate_e_requests(url, items, directory, timeout=1800, opener=urllib.request.urlopen):
    """Sequential (c = 1): each request is sent after the previous completed,
    on a new connection; raw responses are saved."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for item in items:
        body = json.dumps(gate_e_payload(item['messages'])).encode()
        request_ = urllib.request.Request(url + '/v1/chat/completions', data=body, method='POST',
                                          headers={'Content-Type': 'application/json', 'Connection': 'close'})
        row = {'id': item['id'], 'send_ns': time.monotonic_ns(), 'error': None, 'response': None}
        try:
            with opener(request_, timeout=timeout) as response:
                row['status'] = response.status
                row['response'] = json.loads(response.read().decode())
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        row['end_ns'] = time.monotonic_ns()
        (directory / f'{item["id"]}.json').write_text(json.dumps(row, indent=1) + '\n')
        results.append(row)
    return results
