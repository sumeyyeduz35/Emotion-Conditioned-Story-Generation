# generate.py
import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from model import StoryTransformer

# === PARAMETRELER ===
MODEL_PATH = "./best_transformer_checkpoint.pth"
TOKENIZER_PATH = "./tokenizer.pickle"
VOCAB_PATH = "./vocab.json"
MAX_SEQUENCE_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Tokenizer ve Vocab ===
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

inv_vocab = {v: k for k, v in vocab.items()}

# === Model Yükle ===
vocab_size = len(tokenizer.word_index) + 1
model = StoryTransformer(
    vocab_size=vocab_size,
    emotion_size=1,
    embed_size=256,
    hidden_size=512,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    max_len=MAX_SEQUENCE_LEN
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def nucleus_sampling(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # Sort in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_keep = cumulative_probs <= top_p
    sorted_indices_to_keep[..., 0] = True  # Always keep at least one
    indices_to_keep = sorted_indices[sorted_indices_to_keep]

    # Renormalize
    new_probs = probs[indices_to_keep]
    new_probs = new_probs / new_probs.sum()

    # Sample
    next_token_id = indices_to_keep[torch.multinomial(new_probs, num_samples=1)].item()
    return next_token_id

def generate_text(emotion, start_text, max_gen_len=50, temperature=1.0, top_p=0.9):
    prefix = f"{emotion.lower()}: {start_text.lower()}"
    seq = tokenizer.texts_to_sequences([prefix])[0]

    # Başlangıç tokenları ekle
    tokens = [vocab["<bos>"]] + seq

    for _ in range(max_gen_len):
        seq_padded = pad_sequences([tokens], maxlen=MAX_SEQUENCE_LEN, padding='post')
        input_ids = torch.tensor(seq_padded, dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output = model(input_ids, torch.zeros(1, dtype=torch.long).to(DEVICE))
            next_token_logits = output[0, len(tokens)-1, :]

            # Top-p nucleus sampling
            next_token_id = nucleus_sampling(next_token_logits, top_p=top_p, temperature=temperature)

        if next_token_id == vocab["<eos>"] or len(tokens) >= MAX_SEQUENCE_LEN:
            break
        tokens.append(next_token_id)

    # Tokenları kelimelere dönüştür
    generated_words = [inv_vocab.get(tid, "<unk>") for tid in tokens[1:]]
    story = " ".join(generated_words).replace("<eos>", "").strip()
    return story

if __name__ == "__main__":
    emotion = input("🔹 Duygu girin (örn: fear, happy, sad): ")
    start = input("🔹 Başlangıç cümlesi girin: ")
    story = generate_text(emotion, start)
    print("\n🎉 Üretilen Hikaye:\n", story)
