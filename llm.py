from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("Qwen loaded successfully!")


def generate_answer(prompt):

    messages = [
        {
            "role": "system",
            "content": "You are a strict assistant. Use only given context."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.2,
            top_p=0.9
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()