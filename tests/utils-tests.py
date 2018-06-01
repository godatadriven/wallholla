import unittest
from utils import make_pretrained_filenames

class MakeFileNameTestCase(unittest.TestCase):

    def test_basic_usage1(self):
        res = make_pretrained_filenames("catdog", "random", "mobilenet", 200, (100, 100))
        outcome = [
            'catdog-mobilenet-random-200-100x100-train-data.npy',
            'catdog-mobilenet-random-200-100x100-train-label.npy',
            'catdog-mobilenet-random-200-100x100-valid-data.npy',
            'catdog-mobilenet-random-200-100x100-valid-label.npy']

        for i in range(4):
            self.assertEqual(res[i], outcome[i])

    def test_basic_usage2(self):
        res = make_pretrained_filenames("dogcatz", "random", "foofoo", 200, (100, 100))
        outcome = [
            'dogcatz-foofoo-random-200-100x100-train-data.npy',
            'dogcatz-foofoo-random-200-100x100-train-label.npy',
            'dogcatz-foofoo-random-200-100x100-valid-data.npy',
            'dogcatz-foofoo-random-200-100x100-valid-label.npy']

        for i in range(4):
            self.assertEqual(res[i], outcome[i])

    def test_basic_usage3(self):
        res = make_pretrained_filenames("catdog", "random", "mobilenet", 20000, (5, 5))
        outcome = [
            'catdog-mobilenet-random-20000-5x5-train-data.npy',
            'catdog-mobilenet-random-20000-5x5-train-label.npy',
            'catdog-mobilenet-random-20000-5x5-valid-data.npy',
            'catdog-mobilenet-random-20000-5x5-valid-label.npy']

        for i in range(4):
            self.assertEqual(res[i], outcome[i])


if __name__ == '__main__':
    unittest.main()