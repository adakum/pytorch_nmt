import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def sort_batch(input):
    # input will be [num_steps, batch_size]
    omask = (input != 0)
    olens = omask.sum(0)
    lens, final_indx = torch.sort(olens, descending=True)
    orig_indx = torch.sort(final_indx)[1]
    return orig_indx, final_indx, lens, omask.float()


class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_vocab, cell_type,
                 num_layers=1, bidirectional=True,
                 dropout_rnn=0, dropout_emb=0, dropout_ctx=0):
        
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = self.hidden_size * 2
        self.n_vocab = n_vocab
        self.cell_type = cell_type.upper()
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        
        self.dropout_rnn = dropout_rnn
        self.dropout_emb = dropout_emb
        self.dropout_ctx = dropout_ctx

        if self.dropout_emb > 0:
            self.drop_emb = nn.Dropout(self.dropout_emb)

        if self.dropout_ctx > 0:
            self.drop_ctx = nn.Dropout(self.dropout_ctx)

        if self.dropout_rnn > 0:
            self.drop_rnn = nn.Dropout(self.dropout_rnn)

        assert self.cell_type in ["LSTM", "GRU"], "Lord, we got a problem in Cell_Type"

        rnn_cell = getattr(nn, self.cell_type)

        # create embedding layer
        self.emb = nn.Embedding(self.n_vocab, self.input_size)

        # create encoder 
        self.enc = rnn_cell(self.input_size, self.hidden_size,
                                self.num_layers, bias=True, batch_first=False,
                                dropout=self.dropout_rnn,
                                bidirectional=self.bidirectional)


    def forward(self, input):
        # input is of dim (n_timesteps, n_samples)
        orig_indx, perm_indx, lens, mask = sort_batch(input)

        # permute accoring to len
        input = input[:,perm_indx]

        # get embeddings
        emb_input = self.emb(input)
            
        # emb droput 
        if self.dropout_emb > 0 :
            emb_input = self.drop_emb(emb_input)

        # pad sequences
        packed_input = pack_padded_sequence(emb_input, lens.data.tolist())
        
        # run RNN
        enc_outputs, h_t = self.enc(packed_input)
        
        # unpad 
        enc_outputs = pad_packed_sequence(enc_outputs)[0][:, orig_indx]

        # drop
        if self.dropout_ctx>0:
            enc_outputs = self.drop_ctx(enc_outputs)

        # if bidirectional
        if self.bidirectional:
            h_t_combined = torch.cat((h_t[0][0], h_t[0][1]),1)
            c_t_combined = torch.cat((h_t[1][0], h_t[1][1]),1)     
            #  this is the output of last time_step 
            h_t = (h_t_combined, c_t_combined)

        return enc_outputs, h_t 





