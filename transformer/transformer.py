import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
# how many independent sequences will we process in parallel?
batch_size = 64 
# Context length
block_size = 256
# numbers of interation that we will use to train model 
max_iters = 5000
# at which iteration we would like to output loss
eval_interval = 500
#learning rate
learning_rate = 3e-4
# which device is avaliable 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#embedding dimension  
n_embd = 384
# Howevery self attention head we will use 
n_head = 6
# number of blocks 
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Shakespeare data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# map from charater to integer
stoi = { ch:i for i,ch in enumerate(chars) }
# map from integer to character
itos = { i:ch for i,ch in enumerate(chars) }
# functions to apply for words not just character
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

# Train and test splits 90% training and testing
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    '''
    input: 'train' or 'test'
    output: Randomly selected batch of data that are context and target

    ix = randomly selected index (start of a block)
    x = for each index choose all the character after it until the block size
    y = for each context get it's target
    '''
    data = train_data if split == 'train' else val_data
    # max=len(data) - block_size, shape=(batch_size,)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """Single head self attention"""
    # head size is the dim of projection for key, query, and also value 
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size) #what token offers
        self.query = nn.Linear(n_embd, head_size) #what token is looking for 
        self.value = nn.Linear(n_embd, head_size) # what info token contains
        # create lower triangular matrix of block_size x block size 
        # store it as a buffer (not as a parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step(block), channels (embedding dim))
        # output of size (batch, time-step (block), head size (dim of projection))
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)

        #calculate weight (attention score) by multipying query and key and dividing by sqrt(head size)
        # weight tells you how much each token attends to previous tokens 
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**(-0.5)
        # convert it into upper tri matrix
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # convert it into probablity 
        wei = F.softmax(wei, dim=-1) # B, T, T
        
        #value of x
        v = self.value(x)

        #return the enriched version of x
        out = wei @ v # (B, T, T) x (B, T, head size) = (B, T, head_size)
        return out 

class MultiHeadAttention(nn.Module):
    """multiple single Head in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # creates list with length of num_head
        # moduleList is pytorch container gets moved to GPU 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # each head produces vector of head_size 
        # We have num_head of head 
        # then finally we use these head to project to embedding size
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """neral net part of the transformer"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ a single transformer block"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # self attention block
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token is embedded into dim = vocab_size

        # takes token to embedding space
        # takes token's index to embedding space
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        # takes position to embedding space
        # takes position index to embedding sapce
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        # creates sequence of blocks
        # block has multihead attention layer
        # feedforward neural network 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embd)
        # output layer 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # decoupling the input batch
        B, T = idx.shape

        # B batch, T block -> (B, T, C) embedding the token 
        tok_emb = self.token_embedding_table(idx)

        #position embedding
        # input is array create 0...T-1 position 
        # then outputs embedding of 0...T-1 position into n_embd dim -> (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # (B,T,C) + (T, C) we just add position embedding to toke emb
        x = tok_emb + pos_emb

        # (B,T,C) feet it into blocks
        x = self.blocks(x)

        # batch layer norm
        x = self.ln_f(x)

        # output layer #(B, T, vocab_size)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # decoupling the logits to feed into cross entropy
            B, T, C = logits.shape
            # converting the dim of logits 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # calculate loss
            loss = F.cross_entropy(logits, targets)
        return logits, loss
            
    # idx is (B, T) array of indices in the current context 
    # max_new_tokens is how many we want to generate
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
#initialize the model
model = GPTLanguageModel(vocab_size)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if os.path.exists("gpt_weights.pth"):
    m.load_state_dict(torch.load("gpt_weights.pth", map_location=device))
    m.eval()
    print("Loaded existing weights, skipping training.")
else:
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # get batch
        xb, yb = get_batch('train')

        # get logit and loss (forward pass)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # backpropogation
        loss.backward()
        optimizer.step()

    # save trained weights
    torch.save(m.state_dict(), "gpt_weights.pth")
    print("Saved weights to gpt_weights.pth")
    m.eval()

#generate shakesphere 
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(torch.zeros((1, 1), dtype=torch.long, device=device).shape)
#print(torch.tensor(encode("Fortnite"), dtype=torch.long, device=device).shape)
context = torch.tensor(encode("Fortnite"), dtype=torch.long, device=device)[None, :]
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
