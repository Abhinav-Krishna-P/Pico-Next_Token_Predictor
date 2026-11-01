import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import hashlib
from  transformer_block import Block

CHECKPOINT_PATH = "pico.pt"

print(torch.__version__)
print(torch.cuda.is_available())

corpus = [
    "my name is  abhinav krishna p",
    "i am working as a software engineer",
    "working for the last 1.5 years in the industry",
    "started as a mern stack developer",
    "deployed 3  shopify  apps and a chrome extension",
    "participated and  qualified in  SERP api hackathon  ",
    "now learning RAG , FINETUNING and prompt engineering",
    "looking for a job in the field of ai and ml",
    "now learnig LLM Models",
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
print (text)


words=sorted(set(text.split()))  # sorted -> same word order (token ids) on every run
print(words)

vocab_size=len(words)
print(vocab_size)

## building tokenizer  only for the  words  in the corpus
word2index={word:i for i,word in enumerate(words)}
index2word={i:word for word,i in word2index.items()}

print(word2index)
print(index2word)


data=torch.tensor([word2index[w] for w in text.split()],dtype=torch.long)
print(data)

block_size=6
embedding_dim=32
n_heads=2
n_layers=2
lr= 1e-3
epochs=2700

def get_batch(batch_size=16):
    ix=torch.randint(len(data)-block_size, (batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

class Pico(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding=nn.Embedding(block_size,embedding_dim)
        self.blocks=nn.Sequential(*[Block(embedding_dim,block_size,n_heads) for _ in range(n_layers)])
        self.ln_f=nn.LayerNorm(embedding_dim)
        self.head=nn.Linear(embedding_dim,vocab_size)
    
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


config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "embedding_dim": embedding_dim,
    "n_heads": n_heads,
    "n_layers": n_layers,
}
# fingerprint of the training text: if the corpus changes, the saved model is stale
corpus_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_checkpoint():
    """Return the saved checkpoint if it exists and matches the current corpus/config, else None."""
    if not os.path.exists(CHECKPOINT_PATH):
        print("No saved model found -> training.")
        return None
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if ckpt["corpus_hash"] != corpus_hash or ckpt["config"] != config:
        print("Corpus or hyperparameters changed since the model was saved -> retraining.")
        return None
    return ckpt


model=Pico()
force_retrain = "--retrain" in sys.argv
ckpt = None if force_retrain else load_checkpoint()

if ckpt is not None:
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded trained Pico from {CHECKPOINT_PATH} (final loss {ckpt['final_loss']:.4f})")
else:
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)

    for step in range(epochs):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "word2index": word2index,
        "corpus_hash": corpus_hash,
        "final_loss": loss.item(),
    }, CHECKPOINT_PATH)
    print(f"Saved trained Pico to {CHECKPOINT_PATH}")


model.eval()
context = torch.tensor([[word2index["abhinav"]]], dtype=torch.long)
with torch.no_grad():
    out = model.generate(context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(index2word[int(i)] for i in out[0]))