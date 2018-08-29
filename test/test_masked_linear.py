import unittest

import torch
from seq2seq.models.MaskedRNN import MaskedLinear, MASK_TYPES


class TestMaskedLinear(unittest.TestCase):

    def test_wise(self):

        for wise in MASK_TYPES:
            self.assertIsInstance(MaskedLinear(10, 5, wise), MaskedLinear)

        with self.assertRaises(ValueError):
            MaskedLinear(10, 5, 'not-feat-or-elem')

    def test_mask_in_features(self):

        # check size of mask parameter matrix
        ml = MaskedLinear(10, 5, 'feat', 15)
        self.assertEqual(list(ml.W_mask.data.size()), [5, 15])

        # check size of output
        input = torch.Tensor(2, 10).fill_(1)
        mask_input = torch.Tensor(2, 15).fill_(1)
        y = ml(input, mask_input)
        self.assertEqual(list(y.size()), [2, 5])

    def test_num_parameters(self):

        def get_num_params(module):
            num_params = 0
            for p in module.parameters():
                layer_params = 1
                for s in list(p.size()):
                    layer_params *= s
                num_params += layer_params
            return num_params

        ml = MaskedLinear(10, 5, 'feat')
        self.assertEqual(get_num_params(ml), 10*5 + 10*5)

        ml = MaskedLinear(10, 5, 'input')
        self.assertEqual(get_num_params(ml), (10*5 + 10*10))

        ml = MaskedLinear(10, 5, 'elem')
        self.assertEqual(get_num_params(ml), (10*5 + 10*(10*5)))

    def test_output_size_2D(self):

        x = torch.FloatTensor(4, 10)

        ml = MaskedLinear(10, 5, 'feat')
        self.assertEqual(list(ml(x).size()), [4, 5])

    def test_output_size_3D(self):

        x = torch.FloatTensor(4, 1, 10)

        ml = MaskedLinear(10, 5, 'feat')
        self.assertEqual(list(ml(x).size()), [4, 1, 5])

        ml = MaskedLinear(10, 5, 'elem')
        self.assertEqual(list(ml(x).size()), [4, 1, 5])

        ml = MaskedLinear(10, 5, 'input')
        self.assertEqual(list(ml(x).size()), [4, 1, 5])
