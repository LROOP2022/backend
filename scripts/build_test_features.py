"""Build an ID lookup for the original TEST features and word_count.

Usage: python -m scripts.build_test_features TEST.csv text_data.tsv output.sqlite
The inputs may also be the encrypted Hackathon-task.zip when pyzipper is installed:
  python -m scripts.build_test_features --zip Hackathon-task.zip --password '...' output.sqlite
"""
import argparse
import csv
import io
import sqlite3
import struct
import zlib
from contextlib import ExitStack
from pathlib import Path

FEATURE_COUNT = 1367
PACK = struct.Struct('<' + 'd' * FEATURE_COUNT)
csv.field_size_limit(100 * 1024 * 1024)


def build(test_stream, text_stream, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    with sqlite3.connect(target) as db:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('CREATE TABLE test_features (id TEXT PRIMARY KEY, features BLOB NOT NULL, word_count BLOB) WITHOUT ROWID')
        reader = csv.DictReader(io.TextIOWrapper(test_stream, encoding='utf-8-sig', newline=''))
        expected = ['index'] + [f'feature_{i}' for i in range(FEATURE_COUNT)]
        if reader.fieldnames != expected:
            raise ValueError('TEST.csv columns do not match the final model')
        rows = []
        for row in reader:
            rows.append((row['index'], PACK.pack(*(float(row[name]) for name in expected[1:])), None))
            if len(rows) >= 200:
                db.executemany('INSERT INTO test_features VALUES (?,?,?)', rows)
                rows.clear()
        if rows:
            db.executemany('INSERT INTO test_features VALUES (?,?,?)', rows)
        count = db.execute('SELECT count(*) FROM test_features').fetchone()[0]
        reader = csv.DictReader(io.TextIOWrapper(text_stream, encoding='utf-8-sig', newline=''), delimiter='\t')
        if reader.fieldnames != ['index', 'word_count']:
            raise ValueError('text_data.tsv columns differ')
        batch = []
        for row in reader:
            batch.append((zlib.compress(row['word_count'].encode('utf-8'), 3), row['index']))
            if len(batch) >= 200:
                db.executemany('UPDATE test_features SET word_count=? WHERE id=?', batch)
                batch.clear()
        if batch:
            db.executemany('UPDATE test_features SET word_count=? WHERE id=?', batch)
        missing = db.execute('SELECT count(*) FROM test_features WHERE word_count IS NULL').fetchone()[0]
        print(f'Built {target}: {count} TEST records; {missing} without text (treated as empty).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', dest='archive')
    parser.add_argument('--password')
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.archive:
            if len(args.paths) != 1 or not args.password:
                parser.error('--zip needs --password and one output path')
            import pyzipper
            archive = stack.enter_context(pyzipper.AESZipFile(args.archive))
            archive.setpassword(args.password.encode())
            test, text = stack.enter_context(archive.open('TEST.csv')), stack.enter_context(archive.open('text_data.tsv'))
            output = args.paths[0]
        else:
            if len(args.paths) != 3:
                parser.error('provide TEST.csv text_data.tsv output.sqlite')
            test, text = (stack.enter_context(open(path, 'rb')) for path in args.paths[:2])
            output = args.paths[2]
        build(test, text, output)


if __name__ == '__main__':
    main()
