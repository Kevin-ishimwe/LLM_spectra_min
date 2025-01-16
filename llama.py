import logging
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
system_prompt="""You are an expert in NMR spectroscopy analysis and organic chemistry."""


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def test():
  print("test")
  
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_id="meta-llama/Llama-3.3-70B-Instruct"):
    try:
        start_time = time.time()
        clear_gpu_memory()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info(f"Tokenizer loaded successfully in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        logging.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        model.eval()
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def move_to_device(input_ids, model):
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    return input_ids.to(device)

def run_inference(model, tokenizer, prompt,temperature):  # Changed parameter name from task to prompt
    try:
        start_time = time.time()
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs["input_ids"] = move_to_device(inputs["input_ids"], model)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100000,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.75,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )
            
            txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
            txt = txt[len(formatted_prompt):].strip()
            return txt, txt.split("### Start answer ###")[1].split("### End answer ###")[0].replace('\n','').strip().lower()
    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise
