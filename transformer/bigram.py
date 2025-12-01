import torch
import torch.nn as nn
from torch.nn import functional as F

# how sequence of words are we gonna train 
batch_size = 32 
# context length (time dim)
block_size = 8 
max_iters = 300000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token is embedded into dim = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B=batch size,T=Block_size,C=channel length or embedding size)

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
            # forward pass
            logits, loss = self(idx)
            # for each seq in the block focus only on the last time step
            logits = logits[:, -1, :] #(B, C)
            # convert it into probablities 
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
#initialize the model
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

for iter in range(max_iters):

    if iter % eval_interval == 0:
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

#generate shakesphere 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))