import unittest
from utils import make_pretrained_filenames

class MakeFileNameTestCase(unittest.TestCase):

    def test_basic_usage1(self):
        res = make_pretrained_filenames(dataset="catdog",
                                        model="vgg16",
                                        generator="random",
                                        n_img=200,
                                        img_size=(224, 224),
                                        n_orig_img=100)
        x_train_fname, y_train_fname, x_valid_fname, y_valid_fname = res

        self.assertEqual(x_train_fname, "catdog/vgg16/random/224x224-200-100-train-data.npy")
        self.assertEqual(y_train_fname, "catdog/vgg16/random/224x224-200-100-train-label.npy")
        self.assertEqual(x_valid_fname, "catdog/vgg16/224x224-valid-data.npy")
        self.assertEqual(y_valid_fname, "catdog/vgg16/224x224-valid-label.npy")




if __name__ == '__main__':
    unittest.main()