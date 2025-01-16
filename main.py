import os
from prompts import system_prompt, base_prompt, cot_prompt, logic_tips_prompt, expert_tips_prompt, expert_logic_tips_prompt
from helper import json_extractor,csv_extractor,write_benchmark_result,get_output_path,token_formater
from llms import call_openAI,call_gemini
from tanimoto_similarity import calculate_tanimoto
from crosscheck import cross_check_molecule,normalize_string
from smile import generate_smiles_string
import logging
from pathlib import Path
from llama import load_model,run_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# cross reference the prediction
def cross_check(references,challenge,prediction,outfile,scratchpad):
    smiles_llm=generate_smiles_string(prediction)
    logging.info(f"llm prediction smile: {smiles_llm}")
    for molecule in references:
        if(molecule['id']==(challenge['nmr_challenge_id']) and molecule['formula']==challenge['formula']):
            smiles_correct=molecule['smile']
            logging.info(f"correct smile: {smiles_correct}")
            takimono=str(calculate_tanimoto(smiles_correct,smiles_llm))
            logging.info(f" Tanimoto coefficient: {takimono}")
            reference=str(molecule['id'])+','+challenge['formula']+','+f"\"{molecule['name']}\""+","+f"\"{smiles_correct}\""
            if(smiles_correct!=None and smiles_correct==smiles_llm ):
                logging.info("\033[32m \nMOLECULE PREDICTION TRUE \x1b[0m")
                write_benchmark_result(outfile,reference,prediction,'1',smiles_llm,takimono,scratchpad)
                return True
            else:
                logging.info("\033[91m \nMOLECULE PREDICTION FALSE \x1b[0m")
                write_benchmark_result(outfile,reference,prediction,'0',smiles_llm,takimono,scratchpad)
                return False        


def run_batch(temperature,id_file,challenge_file,outputfile,model,tokenizer,prompt,self_aug,formula):
    correct_reference=csv_extractor(id_file)
    molecules= json_extractor(challenge_file)
    for index,molecule in enumerate (molecules):
        data=token_formater(molecule, formula)
        formula = molecule.get('formula') if formula else None
        response,model_prediction=run_single(temperature,prompt(data,formula), self_aug=self_aug,model=model,tokenizer=tokenizer)
        scratch_pad=response.split("### Start answer ###")[0].replace("\n"," ")
        cross_check(references=correct_reference,
                    challenge=molecule,
                    prediction=model_prediction,
                    outfile=outputfile,
                    scratchpad=scratch_pad

                    )


def run_single(temperature,prompt, model,self_aug=False): 
    txt=run_inference(model=model,tokenizer=tokenizer,prompt=prompt,temperature=temperature)
    
    if (not self_aug):
        return response,model_prediction

    # self augmentation test
    response,constitent=call_openAI(conversation=conversation,prompt=consistency_reprompt(),temperature=temperature)
    if (constitent.lower()=="yes"):
        logging.info("\033[34m \nMOLECULAR FORMULA CONSISTENT WITH PREDICTION \x1b[0m")
    else:
        logging.info("\033[41m \nMOLECULAR FORMULA NOT CONSISTENT WITH PREDICTION \nRE-PROMPTING ....\x1b[0m")
        response,model_prediction=call_openAI(conversation=conversation,prompt=regeneration_prompt(),temperature=temperature)
    return response,model_prediction





if (__name__=="__main__"):
 try:
    MODEL = "Llama-3.3-70B-Instruct"
    
    #paths 
    IDS_ROOT="./H1 NMR Datasets/IDS/"
    CHALLENGE_ROOT="./H1 NMR Datasets/Challenges/"
    BENCHMARKS_ROOT = Path("./BENCHMARK")
    # pay attention to correct paths

    CHALLENGE_FILES=[
        f"{CHALLENGE_ROOT}Hnmr_spectra_easy.json",  
        f"{CHALLENGE_ROOT}Hnmr_spectra_medium.json",
        f"{CHALLENGE_ROOT}Hnmr_spectra_hard.json",
        ]
    CHALLENGE_IDS = [
        f"{IDS_ROOT}Hnmr_challenge_easy_ids.csv",
        f"{IDS_ROOT}Hnmr_challenge_med_ids.csv",
        f"{IDS_ROOT}Hnmr_challenge_har_ids.csv",
    ]
    # model params 
    TEMPERATURES=[0,0.5,0.8,1]
    REPROMPT=False
    
    PROMPT_TYPES = {
    "base": base_prompt,
    "cot": cot_prompt,
    "logic": logic_tips_prompt,
    "expert": expert_tips_prompt,
    "expert_logic": expert_logic_tips_prompt
    }
    FORMULA_OPTIONS=[True,False]
    # load llama

    model,tokenizer=load_model()


    for ID,PROMPT in PROMPT_TYPES.items():
    #run entire benchmark
        for FORMULA in FORMULA_OPTIONS:
            for TEMP in TEMPERATURES:
                for index,file in enumerate(CHALLENGE_FILES):
                    CATEGORY=(CHALLENGE_FILES[index].split("_")[2].split(".json")[0]).upper()
                    OUTPUTFILE=get_output_path(BENCHMARKS_ROOT,CATEGORY, MODEL, ID, TEMP, FORMULA)

                    run_batch(
                            temperature=TEMP,
                            id_file=CHALLENGE_IDS[index],
                            challenge_file=CHALLENGE_FILES[index],
                            outputfile=OUTPUTFILE,
                            model=model,
                            tokenizer=tokenizer,
                            prompt=PROMPT,
                            self_aug=REPROMPT,
                            formula=FORMULA
                        )

 except Exception as e:
        print("[ERROR]\n",str(e))