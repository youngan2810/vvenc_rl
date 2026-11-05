"""Build a separate output-instrumented VVdeC v3.2.0; never replace the decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

VVDEC_COMMIT = '649f0b2fafee977c998d7e4d674f8b88d952e3a3'
APP_SHA256 = '2520f94fe10464f6e0609c86d5dd6c2fe8148b659d8b6a46c00a7e38cc9f83c8'

ANCHOR = '''        if( 0 != writeYUVToFile( outStream, outputFrame, outputParams.y4mOutput, outputParams.pyuvOutput ) )
        {
          *logStream << "vvdecapp [error]: write of rec. yuv failed for picture seq. " <<  outputFrame->sequenceNumber << std::endl;
          return false;
        }'''
TRACE = r'''
        // Audit only: record POC after this exact progressive frame was written.
        if( const char* mapPath = std::getenv("VVDEC_OUTPUT_POC_MAP") )
        {
          static uint64_t outputIndex = 0;
          if( !outputFrame->picAttributes ) return false;
          std::ofstream mapFile(mapPath, std::ios::app);
          mapFile << "{\"output_index\":" << outputIndex++
                  << ",\"poc\":" << outputFrame->picAttributes->poc
                  << ",\"width\":" << outputFrame->width
                  << ",\"height\":" << outputFrame->height
                  << ",\"bit_depth\":" << outputFrame->bitDepth << "}\n";
          mapFile.flush();
          if( !mapFile ) return false;
        }
'''


def instrument(text):
    if text.count(ANCHOR) != 1:
        raise ValueError('Not pristine supported VVdeC v3.2.0 output source')
    return '#include <cstdlib>\n' + text.replace(ANCHOR, ANCHOR + TRACE)


def build(source, work, decoder, extra=()):
    source, work, decoder = map(Path, (source, work, decoder))
    # Reject modified source; audit observations must come from the pinned tag.
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    tag = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'v3.2.0^{commit}'], text=True).strip()
    if commit != tag or commit != VVDEC_COMMIT or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Use a clean VVdeC checkout at v3.2.0')
    if work.exists():
        raise ValueError('Audit build output must be a new directory')
    def ignore_root(directory, names):
        return set(names) & {'.git', 'build', 'bin', 'lib'} if Path(directory) == source else set()
    shutil.copytree(source, work / 'source', ignore=ignore_root)
    app = work / 'source/source/App/vvdecapp/vvdecapp.cpp'
    original = app.read_text(encoding='utf-8')
    if hashlib.sha256(original.encode()).hexdigest() != APP_SHA256:
        raise ValueError('VVdeC app source hash mismatch')
    app.write_text(instrument(original), encoding='utf-8')
    subprocess.run(['cmake', '-S', str(work/'source'), '-B', str(work/'build'),
                    '-DCMAKE_BUILD_TYPE=Release', *extra], check=True)
    subprocess.run(['cmake', '--build', str(work/'build'), '--config', 'Release', '-j', '4'], check=True)
    candidates = list((work/'source/bin').rglob('vvdecapp.exe' if decoder.suffix=='.exe' else 'vvdecapp'))
    if len(candidates) != 1:
        raise ValueError('Expected one audit vvdecapp executable: ' + str(candidates))
    target = decoder.with_name(decoder.stem + '_poc_audit' + decoder.suffix)
    if target.exists():
        raise ValueError('Refusing to overwrite existing auditor')
    shutil.copy2(candidates[0], target)
    metadata = dict(vvdec_commit=commit, original_app_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    auditor_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix(target.suffix+'.json').write_text(json.dumps(metadata, indent=2))
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Clean canonical VVdeC v3.2.0 checkout')
    p.add_argument('--work', type=Path, required=True, help='New separate build directory')
    p.add_argument('--decoder', type=Path, required=True, help='Canonical decoder path (unchanged)')
    a = p.parse_args()
    build(a.source, a.work, a.decoder)
