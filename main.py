
import os
from prompts import system_prompt,user_prompt,consistency_reprompt,regeneration_prompt
from extractor import json_extractor,csv_extractor
from llms import call_openAI
from crosscheck import cross_check_molecule,normalize_string
from cas import generate_smiles_string
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_benchmark_result(file,reference,prediction,verdict,smiles_llm):
    csv_file=open(file,"a")
    csv_file.write('\n'+f"{reference}"+','+f"\"{prediction}\""+","+f"\"{smiles_llm}\""+','+verdict+","+"N/A")

def prepare_output_file(path):
    try:
        csv_file=open(path,"a")
        csv_file.write("Id,Formula,TrueName,SMILE-correct,Prediction,SMILE-LLM,Verdict,Isomers")
    except Exception as e:
        logging.error(str(e))


# cross reference the prediction
def cross_check(references,challenge,prediction,outfile):
    smiles_llm=generate_smiles_string(prediction)
    logging.info(f"llm prediction smile: {smiles_llm}")
    for molecule in references:
        if(molecule[0]==(challenge['nmr_challenge_id']) and molecule[1]==challenge['formula'].lower()):
            smiles_correct=generate_smiles_string(molecule[2])
            logging.info(f"correct smile: {smiles_correct}")
            if(smiles_correct!=None and smiles_correct==smiles_llm ):
                logging.info("\033[32m \nMOLECULE PREDICTION TRUE \x1b[0m")
                reference=str(molecule[0])+','+challenge['formula']+','+f"\"{molecule[2]}\""+","+f"\"{smiles_correct}\""
                write_benchmark_result(outfile,reference,prediction,'1',smiles_llm)
                return True
            else:
                logging.info("\033[91m \nMOLECULE PREDICTION FALSE \x1b[0m")
                reference=str(molecule[0])+','+challenge['formula']+','+f"\"{molecule[2]}\""+","+f"\"{smiles_correct}\""
                write_benchmark_result(outfile,reference,prediction,'0',smiles_llm)
                return False        


def run_batch(temperature=0.8,id_file="N/A",challenge_file="N/A",outputfile="N/A"):
    correct_reference=csv_extractor(id_file)
    molecules= json_extractor(challenge_file)
    for index,molecule in enumerate (molecules):
        model_prediction=run_single(temperature,molecule)
        cross_check(references=correct_reference,
                    challenge=molecule,
                    prediction=model_prediction,
                    outfile=outputfile)


def run_single(temperature,molecule): 
    conversation=[]
    conversation.append
    ( {
        "role": "system", 
        "content":system_prompt
        })
    prompt=f"""{user_prompt(molecule)}"""
    response,model_prediction=call_openAI(conversation=conversation,prompt=prompt,temperature=temperature)
    # check molecular formula constency
    response,constitent=call_openAI(conversation=conversation,prompt=consistency_reprompt(),temperature=temperature)
    if (constitent.lower()=="yes"):
        logging.info("\033[34m \nMOLECULAR FORMULA CONSISTENT WITH PREDICTION \x1b[0m")
    else:
        logging.info("\033[41m \nMOLECULAR FORMULA NOT CONSISTENT WITH PREDICTION \nRE-PROMPTING ....\x1b[0m")
        response,model_prediction=call_openAI(conversation=conversation,prompt=regeneration_prompt(),temperature=temperature)
    return model_prediction

if (__name__=="__main__"):
 try:
    MODEL = "gpt-4o1"
    TEMPERATURES=[0.5,0]
    
    ROOT="./NMR datasets/"
    BENCHMARKS_ROOT = "./BENCHMARKS/"

    CHALLENGE_FILES=[
        f"{ROOT}nmr_spectra_easy.json",  
        f"{ROOT}nmr_spectra_medium.json",
        f"{ROOT}nmr_spectra_hard.json",
        ]
    CHALLENGE_IDS = [
        f"{ROOT}nmr_challenge_ids_easy.csv",
        f"{ROOT}nmr_challenge_med_ids.csv",
        f"{ROOT}nmr_challenge_har_ids.csv",
    ]

    #run entire benchmark
    for temp in TEMPERATURES:
        OUTPUTFILES=[
            f"{BENCHMARKS_ROOT}EASY/EASY-results-T{temp}.csv",
            f"{BENCHMARKS_ROOT}MEDIUM/MEDIUM-results-T{temp}.csv",
            f"{BENCHMARKS_ROOT}HARD/HARD-results-T{temp}.csv"
        ]
        prepare_output_file(OUTPUTFILES[2])
        run_batch(
                temperature=temp,
                id_file=CHALLENGE_IDS[2],
                challenge_file=CHALLENGE_FILES[2],
                outputfile=OUTPUTFILES[2]
            )

 except Exception as e:
        print("[ERROR]\n",str(e))
