# gpt2_generate.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# === Model ve Tokenizer'ı yükle ===
MODEL_NAME = "gpt2-medium"  # daha büyük model istersen: "gpt2-large" / "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_story(emotion, start_text, max_length=100, temperature=0.8, top_p=0.9):
    prompt = f"{emotion.lower()}: {start_text.lower()}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    while True:
        emotion = input("\n🔹 Duygu girin (örn: fear, happy, sad) (çıkmak için 'exit' yaz): ").strip()
        if emotion.lower() == "exit":
            print("👋 Çıkış yapılıyor. Görüşmek üzere!")
            break
        start = input("🔹 Başlangıç cümlesi girin: ").strip()
        if start.lower() == "exit":
            print("👋 Çıkış yapılıyor. Görüşmek üzere!")
            break

        story = generate_story(emotion, start)
        print("\n🎉 Üretilen Hikaye:\n", story)

        next_action = input("\n🔍 Başka bir hikaye üretmek ister misin? (yes/no): ").strip().lower()
        if next_action != "yes":
            print("👋 Çıkış yapılıyor. Görüşmek üzere!")
            break

if __name__ == "__main__":
    main()
