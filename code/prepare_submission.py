"""Bundle sources, benchmark data, experiment logs and a selected checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    if args.output.exists():
        p.error('Choose a new output filename.')
    files = {f: f.relative_to(root).as_posix() for f in root.iterdir()
             if f.is_file() and f.suffix in ('.py', '.md', '.txt', '.json', '.csv')}
    for folder in ('data', 'configs', 'tests'):
        for f in (root/folder).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:
                files[f] = f.relative_to(root).as_posix()
    for f in (root/'runs').rglob('*'):
        if f.is_file() and f.suffix in ('.json', '.md', '.log', '.npy'):
            files[f] = f.relative_to(root).as_posix()
    files[args.checkpoint.resolve()] = 'submission/best.pt'
    for name in ('best.pt', 'checkpoint.pt', 'metrics.json'):
        baseline_file = root/'runs/baseline-timing'/name
        if baseline_file.exists():
            files[baseline_file] = f'runs/baseline-timing/{name}'
    # This older, self-trained checkpoint is a continuation-training input,
    # never an evaluation dependency of submission/best.pt.
    files[root.parent/'GUIDE.md'] = 'GUIDE.md'
    manifest = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, relative in sorted(files.items(), key=lambda item: item[1]):
            contents = path.read_bytes()
            bundle.writestr(relative, contents)
            manifest[relative] = {'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}
        bundle.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2))
    print(json.dumps({'bundle': str(args.output), 'file_count': len(manifest),
                      'uncompressed_file_bytes': sum(v['bytes'] for v in manifest.values()),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
