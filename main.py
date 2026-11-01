# =============================================================================
#  Pico - a tiny GPT-style "next token predictor" built from scratch in PyTorch
# -----------------------------------------------------------------------------
#  The whole idea of a language model in one sentence:
#     "Given some words, predict the word that comes next."
#  We train on a few sentences, then let the model write text by repeatedly
#  predicting the next word and appending it.
#
#  Pipeline:  text -> tokenizer (word -> number) -> embeddings -> transformer
#             blocks -> scores for every word in the vocabulary -> next word
#
#  Run:   python main.py            (trains once, then reuses pico.pt)
#         python main.py --retrain  (ignore pico.pt and train again)
# =============================================================================

import torch                      # tensors + automatic gradients
import torch.nn as nn             # neural network layers (Linear, Embedding, ...)
import torch.nn.functional as F   # functions without weights (softmax, cross_entropy)
import os                         # check whether the saved model file exists
import sys                        # read command-line flags like --retrain
import hashlib                    # fingerprint the training text
from  transformer_block import Block   # one transformer layer (attention + feed-forward)

# File where the trained weights are stored, so we don't retrain on every run.
CHECKPOINT_PATH = "pico.pt"

# Sanity check of the environment: PyTorch version and whether a GPU is visible.
print(torch.__version__)
print(torch.cuda.is_available())

# -----------------------------------------------------------------------------
# 1. TRAINING DATA
# -----------------------------------------------------------------------------
# A tiny corpus. Real LLMs train on billions of words; with 9 sentences the
# model will simply memorise them - which is perfect for learning how it works.
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

# Mark the end of every sentence with a special <END> token, so the model can
# learn where a sentence stops. Then join everything into one long string.
corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
print (text)


# -----------------------------------------------------------------------------
# 2. TOKENIZER  (words <-> numbers)
# -----------------------------------------------------------------------------
# Neural networks only understand numbers, so every unique word gets an id.
# This is a "word-level" tokenizer: the vocabulary is every distinct word in
# the corpus. (GPT uses sub-word BPE tokens instead, so it can handle new words.)
words=sorted(set(text.split()))  # sorted -> same word order (token ids) on every run
print(words)

# How many different words the model can read and predict.
vocab_size=len(words)
print(vocab_size)

## building tokenizer  only for the  words  in the corpus
word2index={word:i for i,word in enumerate(words)}   # encode: "abhinav" -> 10
index2word={i:word for word,i in word2index.items()}  # decode: 10 -> "abhinav"

print(word2index)
print(index2word)


# The entire corpus turned into one long list of token ids.
data=torch.tensor([word2index[w] for w in text.split()],dtype=torch.long)
print(data)

# -----------------------------------------------------------------------------
# 3. HYPERPARAMETERS  (the knobs that decide the model's size and training)
# -----------------------------------------------------------------------------
block_size=6       # context length: the model looks at most 6 words back
embedding_dim=32   # each word is represented by a vector of 32 numbers
n_heads=2          # attention heads per block (each looks for different patterns)
n_layers=2         # how many transformer blocks are stacked
lr= 1e-3           # learning rate: how big each weight update is
epochs=2700        # number of training steps (one random batch per step)


# -----------------------------------------------------------------------------
# 4. BATCHING  (making training examples)
# -----------------------------------------------------------------------------
def get_batch(batch_size=16):
    """Pick `batch_size` random windows of the text and return (inputs, targets).

    The target is the input shifted one word to the left, so at every position
    the model learns "given the words so far, predict the next one".
        x = [my, name, is, abhinav, krishna, p]
        y = [name, is, abhinav, krishna, p, <END>]
    One window therefore gives 6 training examples at once.
    """
    # random start positions (leave room for block_size + 1 words)
    ix=torch.randint(len(data)-block_size, (batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])        # inputs  (B, T)
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])    # targets (B, T)
    return x,y


# -----------------------------------------------------------------------------
# 5. THE MODEL
# -----------------------------------------------------------------------------
# Shape names used in the comments:
#   B = batch size, T = number of words in the context (<= block_size),
#   C = embedding_dim (32), V = vocab_size (53)
class Pico(nn.Module):
    def __init__(self):
        """Create all the layers (and therefore all the learnable weights)."""
        super().__init__()
        # Lookup table: word id -> 32-number vector that captures its "meaning".
        self.token_embedding=nn.Embedding(vocab_size,embedding_dim)
        # Lookup table: position 0..5 -> vector, so the model knows word ORDER
        # (attention by itself has no idea which word came first).
        self.position_embedding=nn.Embedding(block_size,embedding_dim)
        # The stack of transformer blocks - this is where words "talk to" each
        # other through attention and get processed by a small MLP.
        self.blocks=nn.Sequential(*[Block(embedding_dim,block_size,n_heads) for _ in range(n_layers)])
        # Final normalisation, keeps the numbers in a stable range.
        self.ln_f=nn.LayerNorm(embedding_dim)
        # Turns each 32-number vector into one score (logit) per vocabulary word.
        self.head=nn.Linear(embedding_dim,vocab_size)

    def forward(self, idx, targets=None):
        """Run the model.

        idx:     (B, T) word ids
        targets: (B, T) the correct next word ids (only given during training)
        returns: logits (B, T, V) - a score for every possible next word,
                 loss   - how wrong the predictions are (None if no targets)
        """
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                                   # (B, T, C) what each word is
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, C)    where each word is
        x = tok_emb + pos_emb   # (B, T, C) word meaning + position, combined
        x = self.blocks(x)      # (B, T, C) words exchange information via attention
        x = self.ln_f(x)        # (B, T, C) normalise
        logits = self.head(x)   # (B, T, V) score for every word in the vocabulary
        loss = None
        if targets is not None:
            # cross_entropy wants a flat list of predictions and answers,
            # so we merge the batch and time dimensions: (B*T, V) and (B*T,)
            B, T, C = logits.shape
            # Cross-entropy = -log(probability given to the correct word).
            # An untrained model guessing randomly scores about ln(53) = 3.97.
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Write text: predict a word, append it, and repeat.

        idx: (B, T) starting word ids (the "prompt").
        This loop is exactly how ChatGPT produces text - one token at a time.
        """
        for _ in range(max_new_tokens):
            # The model can only see block_size words, so keep the last 6.
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # We only care about the prediction after the LAST word.
            logits = logits[:, -1, :]                     # (B, V)
            # Turn scores into probabilities that add up to 1.
            probs = F.softmax(logits, dim=-1)             # (B, V)
            # Sample a word at random according to those probabilities
            # (so the output can differ from run to run).
            next_idx = torch.multinomial(probs, 1)        # (B, 1)
            # Append it and feed the longer sequence back in.
            idx = torch.cat((idx, next_idx), dim=1)       # (B, T+1)
        return idx


# -----------------------------------------------------------------------------
# 6. SAVE / LOAD  (train once, reuse afterwards)
# -----------------------------------------------------------------------------
# Settings that define the model's shape. Saved with the weights so we can
# detect when they no longer match the code.
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


# Build the model with fresh random weights.
model=Pico()
# `python main.py --retrain` forces training even if pico.pt exists.
force_retrain = "--retrain" in sys.argv
ckpt = None if force_retrain else load_checkpoint()

if ckpt is not None:
    # A matching saved model exists: copy its trained weights into the model.
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded trained Pico from {CHECKPOINT_PATH} (final loss {ckpt['final_loss']:.4f})")
else:
    # -------------------------------------------------------------------------
    # 7. TRAINING LOOP
    # -------------------------------------------------------------------------
    # AdamW is the optimizer: it uses the gradients to nudge every weight in
    # the direction that lowers the loss.
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)

    for step in range(epochs):
        xb, yb = get_batch()             # 1. get a random batch of examples
        logits, loss = model(xb, yb)     # 2. forward pass: predict + measure error
        optimizer.zero_grad()            # 3. clear gradients from the previous step
        loss.backward()                  # 4. backpropagation: compute gradients
        optimizer.step()                 # 5. update the weights
        if step % 300 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

    # Save the weights plus everything needed to use them again later.
    torch.save({
        "model_state": model.state_dict(),   # all learned weights
        "config": config,                    # model shape
        "word2index": word2index,            # the tokenizer's vocabulary
        "corpus_hash": corpus_hash,          # which text it was trained on
        "final_loss": loss.item(),
    }, CHECKPOINT_PATH)
    print(f"Saved trained Pico to {CHECKPOINT_PATH}")


# -----------------------------------------------------------------------------
# 8. GENERATION  (let Pico write)
# -----------------------------------------------------------------------------
model.eval()   # switch to inference mode
# The prompt: a single word, shaped (1 batch, 1 word).
context = torch.tensor([[word2index["abhinav"]]], dtype=torch.long)
with torch.no_grad():   # no gradients needed when we're not training
    out = model.generate(context, max_new_tokens=15)

# Turn the generated ids back into words.
print("\nGenerated text:\n")
print(" ".join(index2word[int(i)] for i in out[0]))
