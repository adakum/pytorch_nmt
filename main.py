from torch.autograd import Variable




from Encoder import *


enc = Encoder(3, 2, 5, "lstm", 1, True, 0, 0, 0)

inp = Variable(torch.LongTensor([[1,2,3,4],[2,3,4,1],[1,2,3,4],[1,0,0,0]]))

enc(inp)