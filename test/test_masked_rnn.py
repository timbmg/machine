import unittest

import torch
from seq2seq.models.MaskedRNN import \
    MaskedRNN, MASK_TYPES, CELL_TYPES, CONDITIONS


class TestMaskedRNN(unittest.TestCase):

    def test_cell_types_and_mask_types(self):

        for ct in CELL_TYPES:
            for mi in MASK_TYPES:
                for mh in MASK_TYPES:
                    for condition_input in CONDITIONS:
                        for condition_hidden in CONDITIONS:
                            MaskedRNN(10, 5, 1,
                                      cell_type=ct,
                                      mask_type_input=mi,
                                      mask_type_hidden=mh,
                                      mask_condition_input=condition_input,
                                      mask_condition_hidden=condition_hidden)

        with self.assertRaises(ValueError):
            MaskedRNN(10, 5, 1,
                      cell_type='not-a-cell-type',
                      mask_type_input='feat',
                      mask_type_hidden='feat')
            MaskedRNN(10, 5, 1,
                      cell_type='srn',
                      mask_type_input='not-a-mask-input',
                      mask_type_hidden='feat')
            MaskedRNN(10, 5, 1,
                      cell_type='srn',
                      mask_type_input='feat',
                      mask_type_hidden='not-a-mask-hidden')

    def test_input_output_types(self):

        # Packed Sequence input
        x = torch.nn.utils.rnn.pack_padded_sequence(
                torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                batch_first=True)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_type_input='feat',
                                mask_type_hidden='elem')
        y, _ = masked_lstm(x)
        self.assertIsInstance(y, torch.nn.utils.rnn.PackedSequence)

        # Float Tensor input
        x = torch.FloatTensor(4, 8, 10)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_type_input='feat',
                                mask_type_hidden='elem')
        y, _ = masked_lstm(x)
        self.assertIsInstance(y, torch.FloatTensor)

    def test_output_sizes(self):

        x = torch.nn.utils.rnn.pack_padded_sequence(
                torch.FloatTensor(4, 8, 10), [8, 8, 8, 8],
                batch_first=True)
        masked_lstm = MaskedRNN(10, 5, 1,
                                cell_type='lstm',
                                mask_type_input='feat',
                                mask_type_hidden='elem')
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
                                   mask_type_input='feat',
                                   mask_type_hidden='elem')
            y, hx = masked_rnn(x, hx)

    def test_mask_weight_sizes(self):

        input_size = 10
        hidden_size = 5

        for condition_input in CONDITIONS:
            for condition_hidden in CONDITIONS:

                mr = MaskedRNN(input_size, hidden_size, 1,
                               cell_type='lstm',
                               mask_type_input='feat',
                               mask_type_hidden='elem',
                               mask_condition_input=condition_input,
                               mask_condition_hidden=condition_hidden)

                if condition_input == 'x':
                    mask_input_size = [hidden_size, input_size]
                elif condition_input == 'h':
                    mask_input_size = [hidden_size, hidden_size]
                elif condition_input == 'x_h':
                    mask_input_size = [hidden_size, input_size+hidden_size]
                self.assertEqual(list(mr.cell.W_f.W_mask.size()),
                                 mask_input_size)

                if condition_hidden == 'x':
                    mask_hidden_size = [hidden_size*hidden_size, input_size]
                elif condition_hidden == 'h':
                    mask_hidden_size = [hidden_size*hidden_size, hidden_size]
                elif condition_hidden == 'x_h':
                    mask_hidden_size = [hidden_size*hidden_size,
                                        input_size+hidden_size]
                self.assertEqual(list(mr.cell.U_f.W_mask.size()),
                                 mask_hidden_size)
