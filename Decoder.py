import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout, bias):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.bias = bias

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=self.bias))
            input_size = hidden_size

    def forward(self, input, hidden):
        # input (batch_size * input_size)
        # hidden (num_layers * hidden_size )
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))
            input = h_i
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs += [h_i]
            cs += [c_i]

        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return input, (hs, cs)


class Decoder(nn.Module):
	def __init__(self, input_size, hidden_size, n_vocab, att_type, dropout, cell_type, num_layers):
		super(Decoder, self).__init__()
		
		# embedding size
		self.input_size = input_size
		# hidden network size
		self.hidden_size = hidden_size
		# num vocab
		self.n_vocab = n_vocab

		# attn type
		self.att_type = att_type
		self.dropout = dropout
		self.cell_type = cell_type
		# num layers
		self.num_layers = num_layers

		self.bias = True

		if self.cell_type.upper() == "LSTM":
			self.rnn_cell = StackedLSTMCell(self.num_layers, self.input_size, self.hidden_size, self.dropout, self.bias)
		else :
			print(" Not Implemented ")

		# embedding layer
		self.emb = nn.Embedding(self.n_vocab, self.input_size)

		self.out2prob = FF(hidden_size, n_vocab)

	def forward(self, dec_input, enc_outputs, h_t):
		# dec_input  :  time_steps * batch_size

		# embed dec input
		emb_inp = self.emb(dec_input)

		h_o, c_o = h_t
		for time in emb_inp.shape[0]:
			inp_i = emb_inp[time]
			output, (h_o, c_o) = self.rnn_cell(inp_i, (h_o, c_o))

			# project this output 
			logits = self.out2prob(output)









