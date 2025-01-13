import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch,Accelerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
accelerator = Accelerator(even_batches=True)

def load_model(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    try:
        start_time = time.time()
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        with init_empty_weights():
          model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        path_to_index=""
        model = load_checkpoint_and_dispatch(
            model,path_to_index,
            device_map='auto',
            offload_folder="offload",
            offload_state_dict=True,
            dtype = "float16",
            no_split_module_classes=["LlamaDecoderLayer"]
        )
        logging.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.\x1b[0m")

    except Exception as e:
        logging.error(f"\033[31mError loading model: {e}\033[0m")
        raise

    try:
        tokenizer_start_time = time.time()
        logging.info(f"\033[94mLoading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        logging.info(f"Tokenizer loaded successfully in {time.time() - tokenizer_start_time:.2f} seconds.\x1b[0m")
    except Exception as e:
        logging.error(f"\033[31mError loading tokenizer: {e}\033[0m")
        raise

    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, input_text):
    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=50000,
            temperature=1,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    end_time = time.time()
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    logging.info(f"\033[94m--------------------------------Output generated in {end_time - start_time:.2f} seconds.\n \x1b[0m")
    print("\n\033[94m LLama: \x1b[0m", decoded_output, "\n")
    return decoded_output

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    input_text ="""
        Here is HNMR spectrum data:
        {
        "formula": "C9H10O",
        "nmr_challenge_id": 143,
        "peaks": [
          {
            "name": "A (t)",
            "shift": 1.26,
            "range": [1.28, 1.23],
            "hydrogens": 3,
            "integral": 3.0,
            "class": "t",
            "j_values": [7.21, 7.21]
          },
          {
            "name": "B (q)",
            "shift": 3.04,
            "range": [3.07, 3.01],
            "hydrogens": 2,
            "integral": 2.02,
            "class": "q",
            "j_values": [7.23, 7.23, 7.22]
          },
          {
            "name": "C (m)",
            "shift": 8.0,
            "range": [8.02, 7.96],
            "hydrogens": 2,
            "integral": 1.98,
            "class": "m",
            "j_values": []
          },
          {
            "name": "D (m)",
            "shift": 7.58,
            "range": [7.61, 7.55],
            "hydrogens": 1,
            "integral": 1.03,
            "class": "m",
            "j_values": []
          },
          {
            "name": "E (m)",
            "shift": 7.48,
            "range": [7.52, 7.45],
            "hydrogens": 2,
            "integral": 2.0,
            "class": "m",
            "j_values": []
          }
        ]
        }
                 
        What's the molecule name? 
        Think step-by-step, making extensive use of a scratchpad to record your thoughts. Consider finding ways to group related peaks 
        together, and keep track of the stoichiometry and the amount of unassigned H atoms as you make provisional assignments.
        Please solely look at the HNMR to determine the molecule.
        Format the final answer like this - 
        ### Scratchpad ### <scratchpad> ### Scratchpad ###
        ### Start answer ### <prediction> ### End answer ###
        The prediction should only contain the name of the molecule and no other text
        """

    model, tokenizer = load_model(model_id)
    generated_text = run_inference(model, tokenizer, input_text)
    
    logging.info(f"Llama: Generated text: {generated_text}")