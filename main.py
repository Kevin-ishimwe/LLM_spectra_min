
import os
from prompts import system_prompt,user_prompt_COT,consistency_reprompt,regeneration_prompt
from extractor import json_extractor,csv_extractor
from llms import call_openAI,call_gemini
from tanimoto_similarity import calculate_tanimoto
from crosscheck import cross_check_molecule,normalize_string
from smile import generate_smiles_string
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_benchmark_result(file,reference,prediction,verdict,smiles_llm,takimono,scratchpad):
    csv_file=open(file,"a")
    csv_file.write('\n'+f"{reference}"+','+f"\"{prediction}\""+","+f"\"{smiles_llm}\""+','+verdict+","+takimono+','+"N/A"+","+f"\"{scratchpad}\"")

def prepare_output_file(path):
    try:
        if (os.path.exists(path)):
            return
        else :
            csv_file=open(path,"a")
            csv_file.write("Id,Formula,TrueName,SMILE-correct,Prediction,SMILE-LLM,Verdict,TanimotoCoefficient,Isomers,ScratchPad")
    except Exception as e:
        logging.error(str(e))


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


def run_batch(temperature=0.8,id_file="N/A",challenge_file="N/A",outputfile="N/A",reprompt=False):
    correct_reference=csv_extractor(id_file)
    molecules= json_extractor(challenge_file)
    for index,molecule in enumerate (molecules):
        response,model_prediction=run_single(temperature,molecule, reprompt=reprompt)
        scratch_pad=response.split("### Start answer ###")[0].replace("\n"," ")
        cross_check(references=correct_reference,
                    challenge=molecule,
                    prediction=model_prediction,
                    outfile=outputfile,
                    scratchpad=scratch_pad

                    )


def run_single(temperature,molecule, reprompt=False): 
    conversation=[]
    conversation.append
    ( {
        "role": "system", 
        "content":system_prompt
        })
    prompt=f"""{user_prompt_COT(molecule)}"""
    response,model_prediction=call_gemini(conversation=conversation,prompt=prompt,temperature=temperature)

    if (not reprompt):
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
    MODEL = "gpt-4o1"
    TEMPERATURES=[0,0.5,0.8,1]
    REPROMPT=False
    IDS_ROOT="./NMR Datasets/IDS/"
    CHALLENGE_ROOT="./NMR Datasets/CHALLENGES/"
    BENCHMARKS_ROOT = "./BENCHMARKS/GEMINI/COT/"

    CHALLENGE_FILES=[
        f"{CHALLENGE_ROOT}nmr_spectra_easy.json",  
        f"{CHALLENGE_ROOT}nmr_spectra_medium.json",
        f"{CHALLENGE_ROOT}nmr_spectra_hard.json",
        ]
    CHALLENGE_IDS = [
        f"{IDS_ROOT}nmr_challenge_ids_easy.csv",
        f"{IDS_ROOT}nmr_challenge_med_ids.csv",
        f"{IDS_ROOT}nmr_challenge_har_ids.csv",
    ]

    #run entire benchmark
    for temp in TEMPERATURES:
        OUTPUTFILES=[
            f"{BENCHMARKS_ROOT}EASY/EASY-results-T{temp}.csv",
            f"{BENCHMARKS_ROOT}MEDIUM/MEDIUM-results-T{temp}.csv",
            f"{BENCHMARKS_ROOT}HARD/HARD-results-T{temp}.csv"
        ]

        for index,file in enumerate(CHALLENGE_FILES):
            prepare_output_file(OUTPUTFILES[index])
            run_batch(
                    temperature=temp,
                    id_file=CHALLENGE_IDS[index],
                    challenge_file=CHALLENGE_FILES[index],
                    outputfile=OUTPUTFILES[index],
                    reprompt=REPROMPT
                )

 except Exception as e:
        print("[ERROR]\n",str(e))