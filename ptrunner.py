# idek if this works twinazini 

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

# ─────────────────────────────
# Flask setup
# ─────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────
# Tokenizer (CHAR LEVEL)
# ─────────────────────────────
class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def fit(self, text):
        chars = sorted(set(text))
        self.stoi = {c:i for i,c in enumerate(chars)}
        self.itos = {i:c for i,c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos.get(i,'') for i in ids)


# ─────────────────────────────
# MODEL (Mini GPT)
# ─────────────────────────────
class ModelConfig:
    vocab_size = 256
    block_size = 512
    n_embd = 512
    n_heads = 8
    n_layers = 6


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embd // cfg.n_heads

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = nn.Sequential(
            nn.Linear(cfg.n_embd, 4*cfg.n_embd),
            nn.GELU(),
            nn.Linear(4*cfg.n_embd, cfg.n_embd)
        )

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.tok(idx) + self.pos(pos)
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200):
        for _ in range(max_new_tokens):
            logits = self(idx[:,-self.cfg.block_size:])
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ─────────────────────────────
# GLOBAL STATE
# ─────────────────────────────
model = None
tokenizer = None


def load_model(model_path, text_sample="hello world"):
    global model, tokenizer

    tokenizer = CharTokenizer()
    tokenizer.fit(text_sample)

    cfg = ModelConfig()
    cfg.vocab_size = tokenizer.vocab_size

    model = MiniGPT(cfg).to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()


# ─────────────────────────────
# HTML UI
# ─────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>PT AI Runner</title>
<style>
body { font-family: Arial; background:#111; color:#eee; display:flex; flex-direction:column; align-items:center; }
.box { width:600px; background:#222; padding:15px; margin-top:20px; border-radius:10px; }
#chat { height:300px; overflow-y:auto; background:#000; padding:10px; }
.msg { margin:5px 0; }
.user { color:#4fc3f7; }
.bot { color:#81c784; }
</style>
</head>
<body>

<h2>Single File AI Runner</h2>

<div class="box">
<input type="file" id="file">
<button onclick="upload()">Upload .pt</button>
</div>

<div class="box">
<div id="chat"></div>
<input id="input" style="width:80%">
<button onclick="send()">Send</button>
</div>

<script>
let ready=false;

async function upload(){
    let f=document.getElementById("file").files[0];
    let fd=new FormData();
    fd.append("file",f);

    let r=await fetch("/upload",{method:"POST",body:fd});
    let d=await r.json();
    alert(d.status);
    ready=true;
}

async function send(){
    if(!ready){alert("upload model first");return;}

    let t=document.getElementById("input");
    let text=t.value;
    t.value="";

    add("You",text,"user");

    let r=await fetch("/chat",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({prompt:text})
    });

    let d=await r.json();
    add("Bot",d.response,"bot");
}

function add(r,m,c){
    let div=document.createElement("div");
    div.className="msg "+c;
    div.innerText=r+": "+m;
    document.getElementById("chat").appendChild(div);
}
</script>

</body>
</html>
"""


# ─────────────────────────────
# Routes
# ─────────────────────────────
@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    global model

    f = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(path)

    load_model(path)

    return jsonify({"status": "model loaded"})


@app.route("/chat", methods=["POST"])
def chat():
    global model, tokenizer

    prompt = request.json["prompt"]

    idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(idx, 200)

    text = tokenizer.decode(out[0].tolist())

    return jsonify({"response": text})


# ─────────────────────────────
# RUN
# ─────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
