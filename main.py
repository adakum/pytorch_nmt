from torch.autograd import Variable
from Encoder import *
from Decoder import *

enc = Encoder(3, 2, 5, "lstm", 1, True, 0, 0, 0)
	# def __init__(self, input_size, hidden_size, n_vocab, att_type, dropout, cell_type, num_layers):

dec = Decoder(2, 3, 5, "luong", 0.1, "lstm", 1) 
inp = Variable(torch.LongTensor([[1,2,3,4],[2,3,4,1],[1,2,3,4],[1,0,0,0]]))

enc_outputs = Variable(torch.Tensor(4, 4, 5 ).uniform_(0, 1))
h_t = Variable(torch.Tensor(1, 4, 3).uniform_(0, 1))
c_t = Variable(torch.Tensor(1, 4, 3).uniform_(0, 1))

print(dec)
print("Loss : {}".format(dec(inp, enc_outputs,  (h_t, c_t))))