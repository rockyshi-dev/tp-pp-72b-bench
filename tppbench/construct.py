"""CPU-only, offline prompt list for the Qwen2.5-72B-Instruct tokenizer.

Unmodified excerpts of
the pinned vLLM source rendered under the Qwen2.5-72B-Instruct chat template to
exactly 1536 tokens, an explicit empty system message, distinct first text
tokens and distinct first 16-token blocks. Banks: for each level in
BANK_LEVELS and each of two retry banks, a disjoint warm-up (2c) and measured
set; plus one burst bank of 64 for the start tests. The five gate E
non-English prompts (gate_e_prompts.json) are hashed and token-counted here.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .spec import (BANK_LEVELS, BANK_MEASURED, BURST_REQUESTS, COMMIT, MODEL, PROMPT_TOKENS, REVISION,
                   digest, output_path, write_json)

GATE_E_FILE = Path(__file__).with_name('gate_e_prompts.json')


def messages(text):
    # An explicit empty system message suppresses Qwen's automatic shared prose.
    return [{'role': 'system', 'content': ''}, {'role': 'user', 'content': text}]


def bank_layout():
    banks, pos = {}, 0
    for retry in (0, 1):
        for c in BANK_LEVELS:
            n = BANK_MEASURED[c]
            banks[f'{c}:{retry}'] = {'warmup': list(range(pos, pos + 2 * c)),
                                     'measured': list(range(pos + 2 * c, pos + 2 * c + n))}
            pos += 2 * c + n
    banks['burst'] = {'warmup': [], 'measured': list(range(pos, pos + BURST_REQUESTS))}
    return banks, pos + BURST_REQUESTS


def build(tokenizer, source, count):
    used, first_blocks, rows = set(), set(), []
    for file in sorted((source / 'vllm').rglob('*.py')):
        text = file.read_text()
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets, ids = enc['offset_mapping'], enc['input_ids']
        for start, token in enumerate(ids):
            if token in used or len(ids) - start < PROMPT_TOKENS:
                continue
            lo = offsets[start][0]
            for end in range(start + 1490, min(len(ids), start + 1540)):
                hi = offsets[end][0]
                excerpt = text[lo:hi]
                content_ids = tokenizer.encode(excerpt, add_special_tokens=False)
                if not content_ids or content_ids[0] in used:
                    break
                msg = messages(excerpt)
                rendered = tokenizer.apply_chat_template(msg, tokenize=True,
                                                         add_generation_prompt=True, return_dict=False)
                if len(rendered) == PROMPT_TOKENS:
                    first_block = tuple(rendered[:16])
                    if first_block in first_blocks:
                        break
                    rows.append({'id': len(rows), 'messages': msg,
                                 'prompt_tokens': PROMPT_TOKENS, 'first_text_token': content_ids[0],
                                 'token_ids': rendered, 'source_file': str(file.relative_to(source)),
                                 'source_sha256': digest(file), 'char_start': lo, 'char_end': hi})
                    used.add(content_ids[0])
                    first_blocks.add(first_block)
                    if len(rows) % 64 == 0:
                        print(json.dumps({'constructed': len(rows), 'target': count}), flush=True)
                    if len(rows) == count:
                        return rows
                    break
    raise RuntimeError(f'Only {len(rows)} distinct first-text-token prompts; need {count}')


def gate_e_prompts(tokenizer, path=GATE_E_FILE):
    data = json.loads(Path(path).read_text())
    rows = []
    for item in data['prompts']:
        msg = messages(item['text'])
        ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=False)
        rows.append({'id': item['id'], 'language': item['language'], 'messages': msg,
                     'prompt_tokens': len(ids)})
    if len(rows) != 5 or len({r['id'] for r in rows}) != 5:
        raise RuntimeError('Gate E needs exactly five distinct non-English prompts')
    return {'file': Path(path).name, 'sha256': digest(path), 'prompts': rows}


def content_digest(artifact):
    """sha256 of what the server is sent (every request's chat messages and rendered token ids,
    the bank layout, the gate E messages), independent of file paths and JSON layout."""
    body = {'banks': artifact['banks'],
            'requests': [[r['messages'], r['token_ids']] for r in artifact['requests']],
            'gate_e': [[p['id'], p['messages']] for p in artifact['gate_e_non_english']['prompts']]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def build_file(source, snapshot, output):
    """Build the prompt file from a clean vLLM checkout at COMMIT and the model snapshot."""
    source, snapshot, out = Path(source), Path(snapshot), output_path(output)
    if out.exists():
        raise FileExistsError(out)
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise RuntimeError('Wrong source commit')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'],
                               text=True).strip():
        raise RuntimeError('Source has tracked modifications')
    os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    banks, total = bank_layout()
    rows = build(tokenizer, source, total)
    artifact = {'model': MODEL, 'model_revision': REVISION, 'source_commit': COMMIT,
                'system_message': 'explicit empty', 'banks': banks, 'requests': rows,
                'gate_e_non_english': gate_e_prompts(tokenizer),
                'tokenizer_files': {p.name: digest(p) for p in snapshot.iterdir()
                                    if p.is_file() and p.suffix in ('.json', '.txt')}}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, artifact)
    summary = {'output': str(out), 'sha256': digest(out), 'content_sha256': content_digest(artifact),
               'requests': len(rows), 'distinct_first_text_tokens': len({r['first_text_token'] for r in rows}),
               'rendered_tokens': PROMPT_TOKENS, 'gate_e_sha256': artifact['gate_e_non_english']['sha256']}
    print(json.dumps(summary), flush=True)
    return summary
