import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ml_backend.predictor import TestFeaturesLookup
from scripts.build_test_features import build


class TestFeatureArtifactTest(unittest.TestCase):
    def test_id_preserves_float64_and_joins_text(self):
        features = [f'feature_{i}' for i in range(1367)]
        values = ['0.12345678912345678'] + ['0'] * 1366
        test_csv = (','.join(['index', *features]) + '\n194357,' + ','.join(values) + '\n').encode()
        text_tsv = 'index\tword_count\n194357\tспорт 3 музыка 2\n'.encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_features.sqlite'
            build(io.BytesIO(test_csv), io.BytesIO(text_tsv), path)
            record = TestFeaturesLookup(directory).record_by_index('194357', features)
            self.assertEqual(record['feature_0'], float(values[0]))
            self.assertEqual(record['word_count'], 'спорт 3 музыка 2')
            self.assertEqual(len(record), 1369)
            with self.assertRaises(KeyError):
                TestFeaturesLookup(directory).record_by_index('missing', features)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM test_features').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
