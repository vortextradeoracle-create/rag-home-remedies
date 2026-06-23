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
            "content": "You are a helpful assistant. Use provided sources and include citations like [1], [2]."
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
            max_new_tokens=350,
            temperature=0.05,
            top_p=0.85,
            repetition_penalty=1.15,
            do_sample=True
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()