import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ff import *

class Attention(nn.Module):
	def __init__(self, query_size, key_size, atn_ctx_size ):
		
		super(Attention, self).__init__()
		# embedding size
		# query : decoder hidden state
		# key : encoder outputs


		self.query_size = query_size
		self.key_size = key_size
		self.atn_ctx_size = atn_ctx_size

		# project query to ctx size
		self.query2ctx =  nn.Linear(self.query_size, self.atn_ctx_size, bias=False)
		
		# project key to ctx size, key_size -> atn_ctx_size
		self.key2ctx   =  nn.Linear(self.key_size, self.atn_ctx_size, bias=False)

		# ctx_size -> 1 
		self.attn_layer  = nn.Linear(self.atn_ctx_size, 1)

	def forward(self, query, key):

		# query [T*B*Dim]
		# key [B*Dim]
		
		# first make the query and key of proper sizes 
		q2ctx = self.query2ctx(query)
		
		# transf
		k2ctx = self.key2ctx(key)

		# attention scores [T*B*1] -> [T*B]
		self.attn_layer = self.attn_layer(nn.Tanh(q2ctx + k2ctx)).squeeze(-1)
		 


