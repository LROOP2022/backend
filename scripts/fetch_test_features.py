"""Download the supplemental TEST feature database and verify its checksum."""
import hashlib
import sys
import urllib.request
from pathlib import Path


def main(url, target):
    expected = (Path(__file__).resolve().parents[1] / 'ml_backend' / 'test_features.sha256').read_text().strip()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as source, target.open('wb') as output:
        while block := source.read(1024 * 1024):
            output.write(block)
            digest.update(block)
    if digest.hexdigest() != expected:
        target.unlink(missing_ok=True)
        raise ValueError('TEST features checksum mismatch')
    print(f'OK: {target}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
