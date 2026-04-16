import torch 
import torch.nn as nn 
from torch.nn import LayerNorm,Module,Dropout


# Transformer Block -> Transformer Block is the Fundamental Building block of GPT and Other LLM Architecture
# The Transformer block is repeated 12 times in Gpt-2 model small (124 M parameters)

# Transformer Block


GPT_CONFIG_124M = {
    "vocab_size":50257,
    "context_length": 1024,   # max_number of inputs token which are allowed to next word prediction
    "embedding_dim" : 768,     # Embedding dimension
    "n_heads" : 12,            # Number of attention heads or every Transformer block have the  multi head attention module
    "n_layers": 12,           # Number of layers OR Number of Transformer blocks
    "drop_rate": 0.1,          # Dropout layer for prevent the overfitting  condition
    "qkv_bias": False          # Query-key-value bias
    }


# MultiHeadAttention  block
# layer Normalization

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
      super().__init__()
      self.eps = 1e-5
      self.alpha = nn.Parameter(torch.ones(embedding_dim))
      self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self,X):
      mean = X.mean(dim=-1, keepdim=True)
      var = X.var(dim=-1, keepdim=True , unbiased=False)
      norm_x = (X - mean) / torch.sqrt(var + self.eps)
      return self.alpha * norm_x + self.shift


# GELU activation function

class GELU(nn.Module):
  def __init__(self):
    super().__init__()


  def forward(self,X):
    return 0.5 * X * (1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(X + 0.044715 * torch.pow(X,3))))



class FeedForward(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layers = nn.Sequential(nn.Linear(cfg["embedding_dim"], 4*cfg["embedding_dim"]),
                                GELU(),  #  activation function
                                nn.Linear(4*cfg["embedding_dim"], cfg["embedding_dim"])

                                )
  def  forward(self,X):
    return self.layers(X)

#  Multihead attention mechanism  >>>>> here <<<<<<  

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in , d_out , context_length , num_heads , drop_out , qkv_bias):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads    # here d_out and d_in represent the input embedding and output embedding that is same

    self.W_query = nn.Linear(d_in,d_out , bias=qkv_bias)
    self.W_key = nn.Linear(d_in,d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)

    self.out_proj = nn.Linear(d_out,d_out)
    self.dropout = nn.Dropout(drop_out)

    # causal mask (register buffer so it moves with model)
    self.register_buffer("mask", torch.tril(torch.ones(context_length,context_length)))

  def forward(self,X):
    batch_size, num_token, d_in = X.shape # first  is the batch_size , num_of_tokens , embedding_dim
    query = self.W_query(X)
    key = self.W_key(X)
    value = self.W_value(X)

    # spliit heads here d_in is split into two num_heads and head_dim , number_of_token = context_length

    query = query.view(batch_size , num_token , self.num_heads , self.head_dim).transpose(1,2)
    key  = key.view(batch_size , num_token , self.num_heads , self.head_dim).transpose(1,2)
    value = value.view(batch_size , num_token , self.num_heads , self.head_dim).transpose(1,2)

    # attention score
    scores = query @ key.transpose(-2,-1)

    # scaling
    scores = scores / (self.head_dim**0.5)

    # apply mask
    mask = self.mask[:num_token, : num_token].to(scores.device)  # T is here sequence length
    scores = scores.masked_fill(mask == 0 , float("-inf"))

    # Attention weights
    weights = torch.softmax(scores , dim=-1) # difference between attention weights and score is in attention weights each row sum is to one
    weights = self.dropout(weights)

    # output
    context_vec = torch.matmul(weights,value)

    # combine heads
    context_vec = context_vec.transpose(1,2).contiguous().view(batch_size , num_token , d_in)

    # final projection
    output = self.out_proj(context_vec)

    return output







# Transformer Block full code   >>>>>>  Here  <<<<<<<<

class TransformerBlock(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.attention = MultiHeadAttention(
                                       d_in = cfg["embedding_dim"],
                                       d_out = cfg["embedding_dim"],
                                       context_length = cfg["context_length"],
                                       num_heads = cfg["n_heads"] ,
                                       drop_out = cfg["drop_rate"],
                                       qkv_bias = cfg["qkv_bias"])
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["embedding_dim"])
    self.norm2 = LayerNorm(cfg["embedding_dim"])
    self.dropout_layer = nn.Dropout(cfg["drop_rate"])

  def forward(self,X):
    # shorcut connection for attention block
    shortcut = X
    X = self.norm1(X)
    X = self.attention(X) # size is here number of ( batch_size , Number_tokens , embedding_size )
    X = self.dropout_layer(X)
    X = X + shortcut  # add the original input back

    # shorcut connection for feed forward block
    shortcut = X
    X = self.norm2(X)
    X = self.ff(X)
    X = self.dropout_layer(X)
    X = X + shortcut

    return X








# this is full architecture of gpt model   >>>>>>>> <<<<<<<<<

class DummyGPTModel(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.tok_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
    self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
    self.drop_embedding = nn.Dropout(cfg["drop_rate"])
    self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    self.final_layer_norm = LayerNorm(cfg["embedding_dim"])
    self.out_heads = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"] , bias=False)

  def forward(self,in_idx):
      batch_size, num_token = in_idx.shape
      tok_embeds = self.tok_embedding(in_idx)
      pos_embedding = self.pos_embedding(torch.arange(num_token , device=in_idx.device))
      x = tok_embeds + pos_embedding
      x = self.drop_embedding(x)
      x = self.trf_blocks(x)
      x = self.final_layer_norm(x)
      logits = self.out_heads(x)
      return logits 