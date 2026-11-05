"""Bind output POCs to the exact YUV consumed by the frozen detector.

The canonical decoder is unchanged. A separately built, output-instrumented
VVdeC must produce identical bytes before its POC map can authorize an episode.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

W, H = 480, 320
FB8 = W * H * 3 // 2
FB10 = FB8 * 2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_output_map(rows, frames):
    expected = [(i, i, W, H, 10) for i in range(frames)]
    actual = [(r['output_index'], r['poc'], r['width'], r['height'], r['bit_depth'])
              for r in rows]
    if actual != expected:
        raise RuntimeError('VVdeC output-frame/POC mapping is incomplete or out of order')


def require_auditor(decoder):
    decoder = Path(decoder)
    auditor = decoder.with_name(decoder.stem + '_poc_audit' + decoder.suffix)
    if not auditor.is_file():
        raise RuntimeError('Missing POC auditor. Run build_vvdec_audit.py --help: ' + str(auditor))
    return auditor


def verify_decode_mapping(stream, decoded, decoder, frames, threads=4):
    stream, decoded, decoder = map(Path, (stream, decoded, decoder))
    if decoded.stat().st_size != frames * FB10:
        raise RuntimeError('Decoded YUV size/frame count mismatch')
    auditor = require_auditor(decoder)
    token = uuid.uuid4().hex
    audit_yuv = decoded.parent / ('audit_' + token + '.yuv')
    mapping = decoded.parent / ('output_poc_' + token + '.jsonl')
    log = decoded.parent / ('decoder_audit_' + token + '.log')
    stream_sha = digest(stream)
    try:
        env = dict(os.environ, VVDEC_OUTPUT_POC_MAP=str(mapping.resolve()))
        with log.open('w', encoding='utf-8') as f:
            subprocess.run([str(auditor), '-b', str(stream), '-o', str(audit_yuv),
                            '--threads', str(threads)], env=env, stdout=f,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
        rows = [json.loads(s) for s in mapping.read_text().splitlines() if s.strip()]
        validate_output_map(rows, frames)
        yuv_sha = digest(decoded)
        if audit_yuv.stat().st_size != frames * FB10 or digest(audit_yuv) != yuv_sha:
            raise RuntimeError('Auditor YUV differs from canonical decoder output')
        if digest(stream) != stream_sha:
            raise RuntimeError('Bitstream changed during POC audit')
        result = {'version': 'output-poc-byte-parity-v1', 'verified': True,
                  'frames': frames, 'stream_sha256': stream_sha,
                  'decoded_sha256': yuv_sha, 'decoder_sha256': digest(decoder),
                  'auditor_sha256': digest(auditor), 'map_sha256': digest(mapping),
                  'map_file': mapping.name, 'log_file': log.name}
        (decoded.parent / 'alignment_mapping.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        audit_yuv.unlink(missing_ok=True)


def temporal_alignment_check(ref8, decoded10, frames, *, bitstream=None,
                             decoder=None, decoder_threads=4):
    import numpy as np
    ref8, decoded10 = Path(ref8), Path(decoded10)
    if frames < 3 or ref8.stat().st_size != frames * FB8:
        raise RuntimeError('Reference YUV size/frame count mismatch')
    if bitstream is None or decoder is None:
        raise RuntimeError('Output POC evidence is required; image similarity alone is insufficient')
    mapping = verify_decode_mapping(bitstream, decoded10, decoder, frames, decoder_threads)
    a = np.memmap(ref8, dtype=np.uint8, mode='r', shape=(frames, FB8))
    b = np.memmap(decoded10, dtype='<u2', mode='r', shape=(frames, FB8))
    sums = np.zeros(3)
    wins = np.zeros(3, dtype=int)
    samples = []
    try:
        for t in range(1, frames - 1):
            y = b[t, :W*H].astype(np.float64)
            errors = [float(np.mean((a[t+d, :W*H].astype(np.float64)*4-y)**2))
                      for d in (-1, 0, 1)]
            sums += errors
            # Ties carry no evidence for a shift (e.g. static video).
            if min(errors[0], errors[2]) < errors[1]:
                wins[int(np.argmin(errors))] += 1
            else:
                wins[1] += 1
            if t in {frames//4, frames//2, 3*frames//4}:
                samples.append(dict(frame=t, mse_minus1=errors[0],
                                    mse_0=errors[1], mse_plus1=errors[2]))
        means = sums / (frames-2)
        # Reject strong systematic disagreement even with a valid decode map:
        # the reference itself may have been shifted/replaced.
        for i in (0, 2):
            if means[i] < .75*means[1] and wins[i] > .6*(frames-2):
                raise RuntimeError('Systematic reference/decoded temporal shift despite valid POC map')
        return {'checked': True, 'mapping': mapping, 'sampled': samples,
                'frames_compared': frames-2, 'mean_mse_shifts': means.tolist(),
                'best_shift_counts': wins.tolist(),
                'suspect_count': sum(min(s['mse_minus1'], s['mse_plus1']) < .75*s['mse_0']
                                     for s in samples)}
    finally:
        del a, b
