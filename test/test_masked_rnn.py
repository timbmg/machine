import unittest

import torch
from seq2seq.models.MaskedRNN import MaskedRNN
from seq2seq.models.MaskedRNN import MaskedLinear


class TestMaskedRNN(unittest.TestCase):

    def test_mask_wise(self):

        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm', mask_input=None, mask_hidden=None)
        self.assertIsInstance(masked_lstm.cell.W_f, torch.nn.Linear)
        self.assertIsInstance(masked_lstm.cell.U_f, torch.nn.Linear)

        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm', mask_input='feat', mask_hidden='feat')
        self.assertIsInstance(masked_lstm.cell.W_f, MaskedLinear)
        self.assertIsInstance(masked_lstm.cell.U_f, MaskedLinear)

        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm', mask_input='elem', mask_hidden='elem')
        self.assertIsInstance(masked_lstm.cell.W_f, MaskedLinear)
        self.assertIsInstance(masked_lstm.cell.U_f, MaskedLinear)

    def test_packed_sequence(self):

        x = torch.nn.utils.rnn.pack_padded_sequence(
                torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                batch_first=True)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm', mask_input='feat', mask_hidden='elem')
        y, _ = masked_lstm(x)
        self.assertIsInstance(y, torch.nn.utils.rnn.PackedSequence)

        x = torch.FloatTensor(4, 8, 10)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm', mask_input='feat', mask_hidden='elem')
        y, _ = masked_lstm(x)
        self.assertIsInstance(y, torch.FloatTensor)

    def test_output_sizes(self):

        x = torch.nn.utils.rnn.pack_padded_sequence(
                torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                batch_first=True)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm', mask_input='feat', mask_hidden='elem')
        y, hx = masked_lstm(x)
        self.assertEqual(list(y.data.size()), [4*8, 5])
        self.assertEqual(list(hx[0].size()), [1, 4, 5])
        self.assertEqual(list(hx[1].size()), [1, 4, 5])

    def test_hx_init(self):

        x = torch.nn.utils.rnn.pack_padded_sequence(
                torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                batch_first=True)
        hx = torch.FloatTensor(1, 4, 5), torch.FloatTensor(1, 4, 5)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm', mask_input='feat', mask_hidden='elem')
        y, hx = masked_lstm(x, hx)
