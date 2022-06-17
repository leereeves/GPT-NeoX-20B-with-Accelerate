import gptneox20b
import torch
model = gptneox20b.create_model(
    "./20B_checkpoints/global_step150000",
    use_cache=True,
    device="cuda:0",
)
tokenizer = gptneox20b.create_tokenizer(
    "./20B_checkpoints/20B_tokenizer.json",
)
with torch.inference_mode():
    text = gptneox20b.greedy_generate_text(
        model, tokenizer,
        #"GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI.",
        "How do you feel?",
        max_seq_len=100,
    )

print(text)