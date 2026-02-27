
import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
import math
import os

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="LLM Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS CUSTOM — thème sombre élégant
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

.stApp {
    background: #0d0f14;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: #6b7280;
    font-size: 0.95rem;
    margin-top: 0.6rem;
}
.stTabs [data-baseweb="tab-list"] {
    background: #13161f;
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid #1e2130;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b7280;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 10px 22px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e2130 !important;
    color: #e8eaf0 !important;
}
.model-card {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.8rem;
}
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 13px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 0.9rem;
}
.badge-blue  { background: rgba(96,165,250,0.12); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }
.badge-purple{ background: rgba(167,139,250,0.12); color: #a78bfa; border: 1px solid rgba(167,139,250,0.25); }
.badge-pink  { background: rgba(244,114,182,0.12); color: #f472b6; border: 1px solid rgba(244,114,182,0.25); }
.model-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8eaf0;
    margin: 0 0 0.3rem 0;
}
.model-subtitle {
    color: #6b7280;
    font-size: 0.88rem;
    margin-bottom: 1.3rem;
    line-height: 1.5;
}
.stats-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.stat-chip {
    background: #0d0f14;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.78rem;
    line-height: 1.6;
}
.stat-chip span { color: #6b7280; display: block; }
.stat-chip strong { color: #e8eaf0; font-family: 'Space Mono', monospace; font-size: 0.82rem; }
.stTextArea textarea, .stTextInput input {
    background: #0d0f14 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.3) !important;
}
.coming-soon {
    background: #13161f;
    border: 1px dashed #2d3148;
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.coming-soon h3 {
    font-family: 'Space Mono', monospace;
    color: #a78bfa;
    font-size: 1.15rem;
    margin-bottom: 0.8rem;
}
.coming-soon p { color: #4b5563; font-size: 0.88rem; line-height: 1.7; }
label { color: #9ca3af !important; font-size: 0.85rem !important; font-weight: 500 !important; }
#MainMenu, footer, header { visibility: hidden; }
.footer-bar {
    text-align: center;
    color: #374151;
    font-size: 0.78rem;
    border-top: 1px solid #1e2130;
    padding-top: 1.5rem;
    margin-top: 3rem;
    font-family: 'Space Mono', monospace;
}
blockquote {
    border-left: 3px solid #3b82f6;
    background: #13161f;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.2rem;
    color: #9ca3af;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# ARCHITECTURE MODÈLE
# ============================================================
VOCAB_SIZE  = 16000
D_MODEL     = 512
BLOCK_SIZE  = 256
N_LAYER     = 10
N_HEAD      = 8
DROPOUT     = 0.1
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head    = n_head
        self.head_size = d_model // n_head
        self.qkv       = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj      = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.sa   = MultiHeadAttention(d_model, n_head)
        self.ffwd = FeedForward(d_model)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyLanguageModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.drop   = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block(D_MODEL, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f   = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.token_embedding_table(idx) +
                      self.position_embedding_table(torch.arange(T, device=idx.device)))
        x = self.blocks(x)
        return self.lm_head(self.ln_f(x))


# ============================================================
# GÉNÉRATION UNIVERSELLE
# ============================================================
def generate_stream(model, tokenizer, prompt, max_tokens, temperature=0.75, top_p=0.9):
    model.eval()
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(idx[:, -BLOCK_SIZE:])[0, -1, :] / max(temperature, 1e-5)
            probs  = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            sp[torch.cumsum(sp, 0) - sp > top_p] = 0.0
            sp /= sp.sum()
            nid = si[torch.multinomial(sp, 1)].item()
            idx = torch.cat((idx, torch.tensor([[nid]], device=DEVICE)), dim=1)
            yield tokenizer.decode([nid])
            if nid == tokenizer.token_to_id("</s>"):
                break

def generate_beam_search(model, tokenizer, prompt, max_tokens, beam_size=3, temperature=0.7):
    model.eval()
    
    # 1. Encodage du prompt initial
    encoded = tokenizer.encode(prompt)
    # ids est l'attribut contenant la liste d'entiers pour ByteLevelBPETokenizer
    idx = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(DEVICE) 
    
    # Structure pour stocker les faisceaux (beams)
    # Chaque faisceau : (séquence d'indices, score cumulé)
    beams = [(idx, 0.0)]
    
    for _ in range(max_tokens):
        candidates = []
        
        for seq, score in beams:
            # On ne génère pas plus si le token de fin est atteint
            if seq[0, -1].item() == tokenizer.token_to_id("</s>"):
                candidates.append((seq, score))
                continue
                
            with torch.no_grad():
                # Forward pass pour obtenir les logits du dernier token
                logits = model(seq[:, -BLOCK_SIZE:])
                logits = logits[0, -1, :] / max(temperature, 1e-5)
                probs = F.log_softmax(logits, dim=-1) # Log-probs pour additionner les scores
                
                # On prend les 'beam_size' meilleurs tokens suivants
                top_probs, top_ids = torch.topk(probs, beam_size)
                
                for i in range(beam_size):
                    next_id = top_ids[i].unsqueeze(0).unsqueeze(0)
                    next_seq = torch.cat((seq, next_id), dim=1)
                    next_score = score + top_probs[i].item()
                    candidates.append((next_seq, next_score))
        
        # 2. Sélection des 'beam_size' meilleurs parmi tous les candidats
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Pour le streaming dans Streamlit, on affiche le meilleur token actuel
        best_seq = beams[0][0]
        yield tokenizer.decode([best_seq[0, -1].item()])
        
        # Arrêt si le meilleur faisceau a fini
        if best_seq[0, -1].item() == tokenizer.token_to_id("</s>"):
            break
        



# ============================================================
# CHARGEMENT MODÈLES (mis en cache)
# ============================================================
@st.cache_resource
def load_tiny_gpt():
    tok = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    mod = TinyLanguageModelV3()
    mod.load_state_dict(torch.load("tiny_gpt_v4_10layers.pth", map_location=DEVICE))
    return tok, mod.to(DEVICE).eval()

@st.cache_resource
def load_finetuned():
    tok = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    mod = TinyLanguageModelV3()
    mod.load_state_dict(torch.load("tiny_gpt_v4_wikipedia.pth", map_location=DEVICE))
    # 🔧 Rétablir le weight tying (indispensable !)
    mod.lm_head.weight = mod.token_embedding_table.weight
    return tok, mod.to(DEVICE).eval()

@st.cache_resource
def load_finetuned2():
    tok = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    mod = TinyLanguageModelV3()
    mod.load_state_dict(torch.load("best_model_finetuned.pth", map_location=DEVICE))
    # 🔧 Rétablir le weight tying (indispensable !)
    mod.lm_head.weight = mod.token_embedding_table.weight
    return tok, mod.to(DEVICE).eval()




# ============================================================
# WIDGET RÉGLAGES (3 colonnes inline)
# ============================================================
def settings_row(prefix, default_len=20):
    c1, c2, c3 = st.columns(3)
    with c1:
        length = st.slider("Longueur (tokens)", 10, 100, default_len, key=f"{prefix}_len")
    with c2:
        temp = st.slider("Température", 0.1, 1.5, 0.75, step=0.05, key=f"{prefix}_temp")
    with c3:
        top_p = st.slider("Top-p", 0.5, 1.0, 0.90, step=0.05, key=f"{prefix}_topp")
    return length, temp, top_p


# ============================================================
# HEADER PRINCIPAL
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧠 LLM Playground</h1>
    <p>Exploration et comparaison de modèles de langage &nbsp;·&nbsp;</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📖   TinyGPT — Histoires enfants",
    "🌍   TinyGPT — Wikipedia",
    "📚   TinyGPT — TextBook",
])


# ─────────────────────────────────────
# ONGLET 1 — TinyGPT original
# ─────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="model-card">
        <div class="model-badge badge-blue">✦ CUSTOM · SCRATCH</div>
        <p class="model-title">TinyGPT V4</p>
        <p class="model-subtitle">
            Transformer entraîné entièrement from scratch · 10 couches · 8 têtes d'attention · 
            Cosine LR scheduler · Weight tying · Flash Attention
        </p>
        <div class="stats-row">
            <div class="stat-chip"><span>Paramètres</span><strong>39.8M</strong></div>
            <div class="stat-chip"><span>Loss finale</span><strong>~1.47</strong></div>
            <div class="stat-chip"><span>Contexte</span><strong>256 tokens</strong></div>
            <div class="stat-chip"><span>Vocabulaire</span><strong>16 000</strong></div>
            <div class="stat-chip"><span>Dataset</span><strong>TinyStories 500k</strong></div>
        </div>
        <div style="margin-top:1.2rem; padding:1rem 1.2rem; background:#0d0f14; border-left:3px solid #60a5fa; border-radius:0 10px 10px 0;">
            <p style="color:#6b7280; font-size:0.75rem; font-family:'Space Mono',monospace; margin:0 0 0.5rem 0;">EXEMPLE D'HISTOIRE DU DATASET D'ENTRAÎNEMENT</p>
            <p style="color:#9ca3af; font-size:0.82rem; line-height:1.7; margin:0; font-style:italic;">
                "One day, a fast driver named Tim went for a ride in his loud car. He loved to speed down the street 
                and feel the wind in his hair. As he drove, he saw his friend, Sam, standing by the road. 
                'Hi, Sam!' Tim called out. 'Do you want to go for a ride?' 'Yes, please!' Sam said, and he got in 
                the car. They drove around the town, going fast and having fun. The car was very loud, and everyone 
                could hear them coming. At last, they stopped at the park to play. They ran and laughed until it 
                was time to go home. Tim and Sam had a great day together, speeding in the loud car and playing 
                in the park."
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        tokenizer1, model1 = load_tiny_gpt()

        length1, temp1, top_p1 = settings_row("t1")

        prompt1 = st.text_input(
            "✍️ Début de l'histoire :",
            "Once upon a time, a small robot named Leo",
            key="p1"
        )

        if st.button("🚀 Générer l'histoire", key="g1"):
            def s1():
                yield prompt1 + " "
                yield from generate_stream(model1, tokenizer1, prompt1, length1, temp1, top_p1)
            st.write_stream(s1())

    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant : {e}")
        st.code("tiny_gpt_v3_optimized.pth\ntokenizer_v2/vocab.json\ntokenizer_v2/merges.txt")
    except Exception as e:
        st.error(f"❌ {e}")


# ─────────────────────────────────────
# ONGLET 2 — TinyGPT fine-tuné Wikipedia
# ─────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="model-card">
        <div class="model-badge badge-purple">✦ FINE-TUNED · WIKIPEDIA</div>
        <p class="model-title">TinyGPT V4 — Spécialisé Wikipedia</p>
        <p class="model-subtitle">
            Même architecture que TinyGPT V4, repris depuis le checkpoint entraîné et 
            spécialisé via fine-tuning sur Simple English Wikipedia avec un LR réduit à 3e-5.
            Le modèle adopte un style encyclopédique et factuel.
        </p>
        <div class="stats-row">
            <div class="stat-chip"><span>Paramètres</span><strong>39.8M</strong></div>
            <div class="stat-chip"><span>Base</span><strong>TinyGPT V4</strong></div>
            <div class="stat-chip"><span>Dataset</span><strong>Wikipedia Simple EN</strong></div>
            <div class="stat-chip"><span>LR</span><strong>3e-5</strong></div>
            <div class="stat-chip"><span>Steps</span><strong>7 000</strong></div>
            <div class="stat-chip"><span>Loss sur Wikipedia</span><strong>~3.27</strong></div>
        </div>
        <div style="margin-top:1.2rem; padding:1rem 1.2rem; background:#0d0f14; border-left:3px solid #a78bfa; border-radius:0 10px 10px 0;">
            <p style="color:#6b7280; font-size:0.75rem; font-family:'Space Mono',monospace; margin:0 0 0.5rem 0;">EXEMPLE D'ARTICLE DU DATASET D'ENTRAÎNEMENT</p>
            <p style="color:#9ca3af; font-size:0.82rem; line-height:1.7; margin:0; font-style:italic;">
                "Biology  is the science that studies life, living things, and the evolution of life. Living things include animals, plants, fungi (such as mushrooms), and microorganisms such as bacteria and archaea.
                The term 'biology' is relatively modern. It was introduced in 1799 by a physician, Thomas Beddoes.
                People who study biology are called biologists. Biology looks at how animals and other living things behave and work, and what they are like. Biology also studies how organisms react with each other and the environment. It has existed as a science for about 200 years, and before that it was called "natural history". Biology has many research fields and branches. Like all sciences, biology uses the scientific method. This means that biologists must be able to show evidence for their ideas and that other biologists must be able to test the ideas for themselves.
                Biology attempts to answer questions such as:
                "What are the characteristics of this living thing?" (comparative anatomy)
                "How do the parts work?" (physiology)
                "How should we group living things?" (classification, taxonomy)
                "What does this living thing do?" (behaviour, growth)
                "How does inheritance work?" (genetics)
                "What is the history of life?" (palaeontology)
                "How do living things relate to their environment?" (ecology)
                Modern biology is influenced by evolution, which answers the question: "How has the living world come to be as it is?"
                History
                The word biology comes from the Greek word βίος (bios), "life", and the suffix -λογία (logia), "study of" : 
                Branches
                Algalogy
                Anatomy
                Arachnology
                Bacteriology
                Biochemistry
                Biogeography
                Biophysics
                Botany
                Bryology
                Cell biology
                Cytology
                Dendrology
                Developmental biology
                Ecology
                Endocrinology
                Entomology
                Embryology
                Ethology
                Evolution / Evolutionary biology
                Genetics / Genomics
                Herpetology
                Histology
                Human biology / Anthropology / Primatology
                Ichthyology
                Limnology
                Mammalology
                Marine biology
                Microbiology / Bacteriology
                Molecular biology
                Morphology
                Mycology / Lichenology
                Ornithology
                Palaeontology
                Parasitology
                Phycology
                Phylogenetics
                Physiology
                Taxonomy
                Virology
                Zoology
                References
                Science-related lists"
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists("tiny_gpt_v4_wikipedia.pth"):
        try:
            tokenizer2, model2 = load_finetuned()  # charge tiny_gpt_v4_wikipedia.pth
            length2, temp2, top_p2 = settings_row("t2")
            prompt2 = st.text_input("✍️ Prompt :", "Biology is", key="p2")
            if st.button("🚀 Générer", key="g2"):
                def s2():
                    yield prompt2 + " "
                    yield from generate_stream(model2, tokenizer2, prompt2, length2, temp2, top_p2)
                st.write_stream(s2())
        except Exception as e:
            st.error(f"❌ {e}")
    else:
        st.warning("Fichier 'tiny_gpt_v4_wikipedia.pth' non trouvé. Placez-le dans le dossier.")

# ─────────────────────────────────────
# ONGLET 3 — TinyGPT fine-tuné TextBook (Cosmopedia / fineweb-edu)
# ─────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="model-card">
        <div class="model-badge badge-purple">✦ FINE-TUNED · TEXTBOOK</div>
        <p class="model-title">TinyGPT V4 — Spécialisé TextBook</p>
        <p class="model-subtitle">
            Fine-tuning sur un corpus éducatif Cosmopedia avec 20% de TinyStories 
            pour préserver la génération d'histoires. Loss finale ~3.77.
        </p>
        <div class="stats-row">
            <div class="stat-chip"><span>Paramètres</span><strong>39.8M</strong></div>
            <div class="stat-chip"><span>Base</span><strong>TinyGPT V4</strong></div>
            <div class="stat-chip"><span>Dataset</span><strong>TextBook + TinyStories</strong></div>
            <div class="stat-chip"><span>LR</span><strong>5e-5</strong></div>
            <div class="stat-chip"><span>Steps</span><strong>5 000</strong></div>
            <div class="stat-chip"><span>Loss</span><strong>~3.77</strong></div>
        </div>
        <div style="margin-top:1.2rem; padding:1rem 1.2rem; background:#0d0f14; border-left:3px solid #f472b6; border-radius:0 10px 10px 0;">
            <p style="color:#6b7280; font-size:0.75rem; font-family:'Space Mono',monospace; margin:0 0 0.5rem 0;">EXEMPLE DE TEXTE DU DATASET D'ENTRAÎNEMENT</p>
            <p style="color:#9ca3af; font-size:0.82rem; line-height:1.7; margin:0; font-style:italic;">
                "A random variable is a mathematical concept used in probability theory to represent 
                the possible outcomes of a random phenomenon or experiment, along with their associated 
                probabilities. For instance, if we consider the example of rolling a fair six-sided die, 
                the random variable X could take on any value from 1 to 6, with equal likelihoods of 1/6."
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists("best_model_finetuned.pth"):
        try:
            tokenizer3, model3 = load_finetuned2()  # charge best_model_finetuned.pth
            length3, temp3, top_p3 = settings_row("t3")
            prompt3 = st.text_input("✍️ Prompt :", "The sky is blue because", key="p3")
            if st.button("🚀 Générer", key="g3"):
                def s3():
                    yield prompt3 + " "
                    yield from generate_stream(model3, tokenizer3, prompt3, length3, temp3, top_p3)
                st.write_stream(s3())
        except Exception as e:
            st.error(f"❌ {e}")
    else:
        st.warning("Fichier 'best_model_finetuned.pth' non trouvé. Placez-le dans le dossier.")



