from huggingface_hub import login
import transformers
import torch

transformers.logging.set_verbosity_debug()

model_id = "meta-llama/Llama-3.3-70B-Instruct"
token ="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"
login(token = token)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])