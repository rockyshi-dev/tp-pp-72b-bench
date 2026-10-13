"""Independent CPU-only replay of every stored source excerpt and chat token list."""
import os
from pathlib import Path

from .construct import gate_e_prompts, messages
from .runner import load_prompts
from .construct import content_digest
from .spec import PROMPT_TOKENS, digest


def verify(path, source, snapshot):
    os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    data = load_prompts(path)
    source_text, source_hash = {}, {}
    for row in data['requests']:
        filename = row['source_file']
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source path')
        if filename not in source_text:
            source_text[filename] = (source/filename).read_text()
            source_hash[filename] = digest(source/filename)
        excerpt = source_text[filename][row['char_start']:row['char_end']]
        if source_hash[filename] != row['source_sha256'] or row['messages'] != messages(excerpt):
            raise ValueError(f'Source excerpt mismatch in prompt {row["id"]}')
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                                            add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list) or len(ids) != PROMPT_TOKENS or ids != row['token_ids']:
            raise ValueError(f'Rendered token mismatch in prompt {row["id"]}')
        if tokenizer.encode(excerpt, add_special_tokens=False)[0] != row['first_text_token']:
            raise ValueError('First text token mismatch')
    for name, sha in data['tokenizer_files'].items():
        if digest(snapshot/name) != sha:
            raise ValueError(f'Tokenizer/config file changed: {name}')
    if gate_e_prompts(tokenizer)['prompts'] != data['gate_e_non_english']['prompts']:
        raise ValueError('Gate E prompts differ from the stored record')
    return {'prompts_sha256': digest(path), 'content_sha256': content_digest(data), 'prompts': len(data['requests']), 'rendered_tokens': PROMPT_TOKENS,
            'distinct_first_text_tokens': len({r['first_text_token'] for r in data['requests']}),
            'distinct_first_16_token_blocks': len({tuple(r['token_ids'][:16]) for r in data['requests']}),
            'source_files': len(source_text), 'gate_e_sha256': data['gate_e_non_english']['sha256'],
            'verified': True}
