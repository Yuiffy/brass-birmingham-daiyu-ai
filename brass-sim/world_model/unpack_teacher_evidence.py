"""Verify and unpack the published synthetic-teacher evidence into a fresh directory."""
import argparse, gzip, hashlib, json, shutil, zipfile
from pathlib import Path


def unpack(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty evidence output directory')
    manifest = json.loads((source / 'manifest.json').read_text())
    for row in manifest:
        file = (source / row['file']).resolve()
        if not file.is_relative_to(source.resolve()):
            raise ValueError('Manifest path escapes evidence directory')
        if file.stat().st_size != row['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Evidence hash mismatch: ' + row['file'])
    output.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        file = source / row['file']
        target = output / row['file']
        target.parent.mkdir(parents=True, exist_ok=True)
        if file.suffix == '.zip':
            with zipfile.ZipFile(file) as archive:
                directory = target.with_suffix('')
                for member in archive.namelist():
                    if not (directory / member).resolve().is_relative_to(directory.resolve()):
                        raise ValueError('Archive path escapes snapshot directory')
                archive.extractall(directory)
        elif file.suffix == '.gz':
            target = target.with_suffix('')
            with gzip.open(file, 'rb') as stream, target.open('wb') as dest:
                shutil.copyfileobj(stream, dest)
            if hashlib.sha256(target.read_bytes()).hexdigest() != row['originalSHA256']:
                raise ValueError('Unpacked hash mismatch: ' + row['file'])
        else:
            shutil.copy2(file, target)
    return len(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='world_model/experiments/human-teacher-20261009/reproducibility')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print('Verified and unpacked', unpack(args.source, args.out), 'artifacts')
