#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import logging
import pandas as pd
from pathlib import Path
import os
import time
from tqdm import tqdm 
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [EXTRACTOR] %(message)s'
)

# System prompt
SYSTEM_PROMPT = """You are an expert in NMR spectroscopy analysis and organic chemistry."""

def json_extractor(filename):
    """Extract and validate NMR data from JSON file"""
    try:
        logging.info(f"Reading file: {filename}")
        with open(filename, "r") as file:
            json_data = json.load(file)
            
        # Validate data structure
        if not isinstance(json_data, list):
            logging.error("JSON data is not a list of molecules")
            return []
            
        # Validate each molecule
        valid_molecules = []
        for mol in json_data:
            if validate_molecule_data(mol):
                valid_molecules.append(mol)
            else:
                logging.warning(f"Invalid molecule data found: {mol.get('nmr_challenge_id', 'unknown ID')}")
                
        logging.info(f"Successfully extracted {len(valid_molecules)} valid molecules")
        return valid_molecules
        
    except Exception as e:
        logging.error(f"Error extracting JSON data: {str(e)}")
        return []

def validate_molecule_data(molecule):
    """Validate molecule data structure"""
    required_fields = ['formula', 'nmr_challenge_id', 'peaks']
    return all(field in molecule for field in required_fields) and            isinstance(molecule['peaks'], list) and            all(validate_peak_data(peak) for peak in molecule['peaks'])

def validate_peak_data(peak):
    """Validate peak data structure"""
    required_peak_fields = [
        'name', 'shift', 'range', 'hydrogens',
        'integral', 'class', 'j_values'
    ]
    return all(field in peak for field in required_peak_fields)

def token_formater(data, include_formula=True):
    """Format NMR data for model input"""
    try:
        peaks = []
        for peak in data["peaks"]:
            peak_str = "--------\n"
            peak_str += f"Name: {peak['name']}\n"
            peak_str += f"Shift: {peak['shift']}\n"
            peak_str += f"Range: {' '.join(str(value) for value in peak['range'])}\n"
            if include_formula:  # Only include hydrogens if formula is included
                peak_str += f"Hydrogens: {peak['hydrogens']}\n"
            peak_str += f"Integral: {peak['integral']}\n"
            peak_str += f"Class: {peak['class']}\n"
            peak_str += f"J values: {' '.join(str(value) for value in peak['j_values'])}\n"
            
            if 'method' in peak:
                peak_str += f"Method: {peak['method']}\n"
                
            peaks.append(peak_str)
        
        if include_formula:
            formatted_data = f"\nFormula: {data.get('formula', 'N/A')}\nPeaks:\n{''.join(peaks)}"
        else:
            formatted_data = f"\nPeaks:\n{''.join(peaks)}"
        return formatted_data
    

def call_openai(prompt, model="gpt-4-turbo-2024-04-09", temperature=1):
    """Make calls to OpenAI API with error handling"""
    client = OpenAI()  # Will use OPENAI_API_KEY environment variable
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        txt = completion.choices[0].message.content
        
        # Parse response
        try:
            scratchpad = txt.split("### Scratchpad ###")[1].split("### Scratchpad ###")[0].strip()
            answer = txt.split("### Start answer ###")[1].split("### End answer ###")[0].strip().lower()
        except IndexError:
            logging.error("Response format incorrect")
            scratchpad = txt
            answer = txt.lower()
            
        time.sleep(1)  # Rate limiting
        return scratchpad, answer
        
    except Exception as e:
        logging.error(f"OpenAI API Error: {str(e)}")
        raise

def base_prompt(nmr_data, formula=None):
    """Zero-shot prompt"""
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else f"with chemical formula {formula}:"}
            {nmr_data}
            What's the molecule's name?
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
        """

def cot_prompt(nmr_data, formula=None):
    """Chain of thought prompt"""
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else f"with chemical formula {formula}:"}
            {nmr_data}
            What's the molecule's name?
            Think step-by-step, making extensive use of a scratchpad to record your thoughts.
            Consider finding ways to group related peaks together, and keep track of the stoichiometry 
            and the amount of unassigned H atoms as you make provisional assignments.
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
        """

def logic_tips_prompt(nmr_data, formula=None):
    """Chain of thought prompt with logic tips"""
    return f"""
           Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else f"with chemical formula {formula}:"}
            {nmr_data}
            What's the molecule's name?
            Let's think step-by-step. 
            Use a scratchpad to record your thoughts. First, identify all groups of peaks and the corresponding number of hydrogens, and ensure the stoichiometry of the formula matches that in the spectrum Then, for each group of peaks, hypothesize one or more molecule fragments that could explain the signals in that peak.
            Rank the hypotheses according to which is most plausible
            Occasionally synthesize a candidate molecule from the provisionally-assigned fragments
            Be mindful of connectivity and ensure consistency with all hypothesized fragments
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Continue using the scratchpad until you're confident about the answer.
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
        """

def expert_tips_prompt(nmr_data, formula=None):
    """Chain of thought prompt with expert tips"""
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else f"with chemical formula {formula}:"}
            {nmr_data} 
            What's the molecule's name? Let's think step-by-step. 
            Use a scratchpad to record your thoughts, and continuing using it until your confident about your answer. 
            
            Before the analysis of NMR spectra, think about the molecular formula. Is the molecule fully saturated, or are there some multiple bonds or rings?
            Consider the solvent used. Do you expect the appearance of signals from exchangeable hydrogens (e.g., in CDCl3or DMSO), or are they exchanged to deuterium (e.g. in D2O or methanol-d4)? 
            Identify solvent signals and ignore them in the structural analysis.
            Consider the number of signals, their positions (chemical shifts), intensity, shape and splitting.
            The number of signals corresponds to the number of non-equivalent atoms in the molecule.
            Typical regions in proton NMR spectra are: carboxylic acids: 10–13 ppm, aldehydes: 9–11 ppm, aromatic hydrogens: 6.5–8 ppm, hydrogens attached to a double-bond carbon: 4–7.5 ppm, O–CH fragments (alcohols, ethers): 3–5 ppm, hydrogens attached to a triple-bond carbon: 2–3.5 ppm, hydrogens attached to a single-bond carbon without an electronegative substituent: 0.5–2.5 ppm.
            
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }  
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
    """

def expert_logic_tips_prompt(nmr_data, formula=None):
    """Chain of thought prompt with both expert and logic tips"""
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else f"with chemical formula {formula}:"}
            {nmr_data} 
            Let's think step-by-step using both logical analysis and expert knowledge.
            
            Use a scratchpad to:
            1. Identify all groups of peaks and corresponding hydrogens
            2. Ensure stoichiometry matches the formula
            3. For each peak group, hypothesize possible molecular fragments
            4. Rank hypotheses by plausibility
            5. Build candidate molecules from fragments
            
            Consider these expert guidelines:
            - Check if the molecule is fully saturated or has multiple bonds/rings
            - Look for exchangeable hydrogens based on solvent
            - Note typical chemical shift regions:
              * 10-13 ppm: carboxylic acids
              * 9-11 ppm: aldehydes
              * 6.5-8 ppm: aromatic hydrogens
              * 4-7.5 ppm: alkene hydrogens
              * 3-5 ppm: O-CH fragments
              * 2-3.5 ppm: alkyne hydrogens
              * 0.5-2.5 ppm: alkyl hydrogens
            
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
    """

PROMPT_TYPES = {
    "base": base_prompt,
    "cot": cot_prompt,
    "logic": logic_tips_prompt,
    "expert": expert_tips_prompt,
    "expert_logic": expert_logic_tips_prompt
}

# Set paths
DATA_DIR = Path("./NMR_Set")
BENCHMARK_BASE = Path("./BENCHMARK_LLM")
BENCHMARKS_DIR = BENCHMARK_BASE / "GPT" / "RESULTS"

CHALLENGE_FILES = [
    DATA_DIR / "NMR spectra_EASY(1).json",
    DATA_DIR / "nmr_spectra_medium.json",
    DATA_DIR / "nmr_spectra_hard.json"
]

def get_output_path(difficulty, model, prompt_type, temperature, with_formula=True):
    """Generate output file path based on parameters"""
    form_type = "WITH_FORMULA" if with_formula else "NO_FORMULA"
    base_path = BENCHMARKS_DIR / difficulty / form_type
    base_path.mkdir(parents=True, exist_ok=True)
    filename = f"{difficulty}-results-T{temperature}-{prompt_type}-{model}.csv"
    return base_path / filename

def run_analysis(difficulty, prompt_type, temperature=0, include_formula=True):
    """Run analysis for a specific configuration"""
    # Get corresponding challenge file
    file_idx = {"EASY": 0, "MEDIUM": 1, "HARD": 2}[difficulty]
    challenge_file = CHALLENGE_FILES[file_idx]
    
    # Get output path
    output_file = get_output_path(
        difficulty=difficulty,
        model="gpt_4_turbo_2024_04_09",
        prompt_type=prompt_type,
        temperature=temperature,
        with_formula=include_formula
    )
    
    # Load molecules
    molecules = json_extractor(challenge_file)
    results = []
    
    for molecule in tqdm(molecules, desc=f"Processing {difficulty} dataset with {prompt_type} prompt"):
        try:
            # Format NMR data
            nmr_data = token_formater(molecule)
            
            # Prepare prompt
            prompt_func = PROMPT_TYPES[prompt_type]
            formula = molecule.get('formula') if include_formula else None
            prompt = prompt_func(nmr_data, formula)
            
            # Make API call
            scratchpad, prediction = call_openai(prompt, temperature=temperature)
            
            results.append({
                "Id": molecule['nmr_challenge_id'],
                "Formula": molecule.get('formula', 'N/A'),
                "Prediction": prediction,
                "ScratchPad": scratchpad,
                "PromptType": prompt_type
            })
            
        except Exception as e:
            logging.error(f"Error processing molecule {molecule.get('nmr_challenge_id')}: {str(e)}")
            results.append({
                "Id": molecule['nmr_challenge_id'],
                "Formula": molecule.get('formula', 'N/A'),
                "Prediction": "ERROR",
                "ScratchPad": str(e),
                "PromptType": prompt_type
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
     # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = "your_own_api_key"
    
    difficulties = ["EASY", "MEDIUM", "HARD"]
    prompt_types = ["base", "cot", "logic", "expert", "expert_logic"]
    temperatures = [0, 0.5, 0.8, 1.0]
    formula_conditions = [True, False]  # With and without formula
    
    # Calculate total runs
    total_runs = len(difficulties) * len(prompt_types) * len(temperatures) * len(formula_conditions)
    print(f"\nStarting full NMR analysis:")
    print(f"Total configurations to run: {total_runs}")
    
    # Run all combinations
    for difficulty in difficulties:
        for prompt_type in prompt_types:
            for temp in temperatures:
                for include_formula in formula_conditions:
                    print(f"\nRunning analysis:")
                    print(f"Difficulty: {difficulty}")
                    print(f"Prompt type: {prompt_type}")
                    print(f"Temperature: {temp}")
                    print(f"Formula included: {include_formula}")
                    
                    try:
                        results = run_analysis(
                            difficulty=difficulty,
                            prompt_type=prompt_type,
                            temperature=temp,
                            include_formula=include_formula
                        )
                        print(f"Successfully processed {len(results)} molecules")
                    except Exception as e:
                        logging.error(f"Failed to complete run: {str(e)}")
                        print(f"Error encountered, check logs for details")
    
    print("\nAnalysis complete! Results are saved in the BENCHMARK directory structure.")


# In[ ]:




