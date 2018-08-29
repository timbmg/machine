import unittest

import torch
from seq2seq.models.MaskedRNN import MaskedRNN, MASK_TYPES, CELL_TYPES
from seq2seq.models.MaskedRNN import MaskedLinear


class TestMaskedRNN(unittest.TestCase):

    def test_cell_types_and_mask_types(self):

        for ct in CELL_TYPES:
            for mi in MASK_TYPES:
                for mh in MASK_TYPES:
                    MaskedRNN(10, 5, 1,
                              cell_type=ct,
                              mask_input=mi,
                              mask_hidden=mh)

        with self.assertRaises(ValueError):
            MaskedRNN(10, 5, 1,
                      cell_type='not-a-cell-type',
                      mask_input='feat',
                      mask_hidden='feat')
            MaskedRNN(10, 5, 1,
                      cell_type='srn',
                      mask_input='not-a-mask-input',
                      mask_hidden='feat')
            MaskedRNN(10, 5, 1,
                      cell_type='srn',
                      mask_input='feat',
                      mask_hidden='not-a-mask-hidden')

    def test_mask_types(self):

        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_input=None,
                                mask_hidden=None)
        self.assertIsInstance(masked_lstm.cell.W_f, torch.nn.Linear)
        self.assertIsInstance(masked_lstm.cell.U_f, torch.nn.Linear)

        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_input='feat',
                                mask_hidden='feat')
        self.assertIsInstance(masked_lstm.cell.W_f, MaskedLinear)
        self.assertIsInstance(masked_lstm.cell.U_f, MaskedLinear)

        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_input='elem',
                                mask_hidden='elem')
        self.assertIsInstance(masked_lstm.cell.W_f, MaskedLinear)
        self.assertIsInstance(masked_lstm.cell.U_f, MaskedLinear)

        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_input='input',
                                mask_hidden='input')
        self.assertIsInstance(masked_lstm.cell.W_f, MaskedLinear)
        self.assertIsInstance(masked_lstm.cell.U_f, MaskedLinear)

    def test_input_output_types(self):

        # Packed Sequence input
        x = torch.nn.utils.rnn.pack_padded_sequence(
                torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                batch_first=True)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_input='feat',
                                mask_hidden='elem')
        y, _ = masked_lstm(x)
        self.assertIsInstance(y, torch.nn.utils.rnn.PackedSequence)

        # Float Tensor input
        x = torch.FloatTensor(4, 8, 10)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_input='feat',
                                mask_hidden='elem')
        y, _ = masked_lstm(x)
        self.assertIsInstance(y, torch.FloatTensor)

    def test_output_sizes(self):

        x = torch.nn.utils.rnn.pack_padded_sequence(
                torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                batch_first=True)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_input='feat',
                                mask_hidden='elem')
        y, hx = masked_lstm(x)
        self.assertEqual(list(y.data.size()), [4*8, 5])
        self.assertEqual(list(hx[0].size()), [1, 4, 5])
        self.assertEqual(list(hx[1].size()), [1, 4, 5])

    def test_hx_init(self):

        # LSTM
        for ct in CELL_TYPES:
            x = torch.nn.utils.rnn.pack_padded_sequence(
                    torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                    batch_first=True)
            if ct == 'lstm':
                hx = torch.FloatTensor(1, 4, 5), torch.FloatTensor(1, 4, 5)
            else:
                hx = torch.FloatTensor(1, 4, 5)

            masked_rnn = MaskedRNN(10, 5, 1,
                                   cell_type=ct,
                                   mask_input='feat',
                                   mask_hidden='elem')
            y, hx = masked_rnn(x, hx)
