import tempfile
import unittest
from pathlib import Path
from lesson12_evaluate import read_cases, score_exact, summarize

class EvalTests(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(score_exact('  지도  미세조정 ', '지도 미세조정'), 1)
        self.assertEqual(score_exact('다른 답', '지도 미세조정'), 0)
    def test_summary(self):
        r = summarize([{'model':'base','exact_match':1,'seconds':2},
                       {'model':'base','exact_match':0,'seconds':4}])
        self.assertEqual(r['base']['exact_match'], .5)
        self.assertEqual(r['base']['avg_seconds'], 3)
    def test_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'a.csv'
            p.write_text('question,reference\nA,B\nA,C\n', encoding='utf-8')
            with self.assertRaises(ValueError): read_cases(p)

if __name__ == '__main__':
    unittest.main()