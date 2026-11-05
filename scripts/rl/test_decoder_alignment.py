"""Fail closed on mapping/byte errors; distinguish static video and true shifts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import numpy as np

import decoder_alignment as d
from build_vvdec_audit import instrument, ANCHOR


def rows(n):
    return [dict(output_index=i, poc=i, width=d.W, height=d.H, bit_depth=10)
            for i in range(n)]


class AlignmentTests(unittest.TestCase):
    def test_map_rejects_missing_duplicate_reordered_or_wrong_format(self):
        d.validate_output_map(rows(6), 6)
        cases = [rows(5), rows(6)[::-1]]
        for key, value in [('poc', 0), ('width', 481), ('bit_depth', 8), ('output_index', 0)]:
            case = rows(6); case[2][key] = value; cases.append(case)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                d.validate_output_map(case, 6)

    def fixture(self, root, shift=0, static=False):
        n=12
        rng=np.random.default_rng(9)
        a=rng.integers(16, 230, (n, d.FB8), dtype=np.uint8)
        if static: a[:]=a[0]
        b=np.roll(a, shift, axis=0).astype('<u2')*4
        ref=root/'ref.yuv'; dec=root/'decoded.yuv'
        ref.write_bytes(a.tobytes()); dec.write_bytes(b.tobytes())
        return ref, dec, n

    def test_static_and_correct_video_pass_but_true_shifts_fail(self):
        with TemporaryDirectory() as folder, mock.patch.object(d, 'verify_decode_mapping', return_value={'verified': True}):
            root=Path(folder)
            for static in (False, True):
                ref,dec,n=self.fixture(root, static=static)
                result=d.temporal_alignment_check(ref,dec,n,bitstream=root/'b',decoder=root/'d')
                self.assertTrue(result['mapping']['verified'])
            for shift in (-1,1):
                ref,dec,n=self.fixture(root,shift)
                with self.assertRaisesRegex(RuntimeError, 'Systematic'):
                    d.temporal_alignment_check(ref,dec,n,bitstream=root/'b',decoder=root/'d')

    def test_missing_evidence_and_truncation_fail(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);ref,dec,n=self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'evidence'):
                d.temporal_alignment_check(ref,dec,n)
            ref.write_bytes(b'broken')
            with self.assertRaisesRegex(RuntimeError,'Reference'):
                d.temporal_alignment_check(ref,dec,n)
            dec.write_bytes(b'broken')
            with self.assertRaisesRegex(RuntimeError,'size'):
                d.verify_decode_mapping(root/'b',dec,root/'d',n)

    def test_auditor_parity_failure_and_success(self):
        with TemporaryDirectory() as folder:
            root=Path(folder); _,dec,n=self.fixture(root)
            decoder=root/'decoder';decoder.write_bytes(b'canonical binary')
            (root/'decoder_poc_audit').write_bytes(b'audit binary')
            stream=root/'b.266';stream.write_bytes(b'bitstream')
            original=dec.read_bytes()
            def run(cmd, **kwargs):
                Path(cmd[cmd.index('-o')+1]).write_bytes(original)
                Path(kwargs['env']['VVDEC_OUTPUT_POC_MAP']).write_text(''.join(json.dumps(r)+'\n' for r in rows(n)))
            with mock.patch.object(d.subprocess,'run',side_effect=run):
                self.assertTrue(d.verify_decode_mapping(stream,dec,decoder,n)['verified'])
                dec.write_bytes(bytes([original[0]^1])+original[1:])
                with self.assertRaisesRegex(RuntimeError,'differs'):
                    d.verify_decode_mapping(stream,dec,decoder,n)
            self.assertEqual(list(root.glob('audit_*.yuv')),[])

    def test_instrumentation_requires_one_exact_output_site(self):
        self.assertIn('VVDEC_OUTPUT_POC_MAP',instrument(ANCHOR))
        for text in ('unsupported',ANCHOR+ANCHOR):
            with self.assertRaises(ValueError):instrument(text)


if __name__=='__main__': unittest.main()
