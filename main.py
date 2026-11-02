import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sentencepiece as spm

from transformer_block import Block


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# corpus = [
#     "my name is  abhinav krishna p",
#     "i am working as a software engineer",
#     "working for the last 1.5 years in the industry",
#     "started as a mern stack developer",
#     "deployed 3  shopify  apps and a chrome extension",
#     "participated and  qualified in  SERP api hackathon  ",
#     "now learning RAG , FINETUNING and prompt engineering",
#     "looking for a job in the field of ai and ml",
#     "now learnig LLM Models",
# ]

# corpus = [s + " <END>" for s in corpus]
# text = " ".join(corpus)
# print (text)

# words = list(set(text.split()))
# print(words)

# vocab_size = len(words)
# print(vocab_size)

# word2idx = {w: i for i, w in enumerate(words)}
# print("word2idx : ", word2idx)

# idx2word = {i: w for w, i in word2idx.items()} 
# print("idx2word : ", idx2word) 

# data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
# print("data : ", data) 
# print(len(data))

with  open("corpus.txt", "r",encoding="utf-8") as f:
    text = f.read()
    print(text)

spm.SentencePieceTrainer.train(input='corpus.txt', model_prefix='tokenizer', vocab_size=70, model_type='bpe')  

sp=spm.SentencePieceProcessor()
sp.load('tokenizer.model')

ids=sp.encode(text, out_type=int)
print("ids : ", ids)
data=torch.tensor(ids, dtype=torch.long)
print("data : ", data)

vocab_size = sp.get_piece_size()
print("vocab_size : ", vocab_size)

block_size = 6 
embedding_dim = 32
n_heads = 2  
n_layers = 2 
lr = 1e-3 
epochs = 1500  


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y




class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx



model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch() 
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


context= torch.tensor([sp.encode("name", out_type=int)], dtype=torch.long)
out = model.generate(context, max_new_tokens=20)[0].tolist()
decoded=sp.decode(out)
print("Generated text:", decoded)