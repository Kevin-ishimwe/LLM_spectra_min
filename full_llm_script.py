import json
import logging
import pandas as pd
from pathlib import Path
import os
import time
from tqdm import tqdm 
from openai import OpenAI
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [EXTRACTOR] %(message)s'
)


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Prompts
#Zero-shot prompt
def base_prompt(nmr_data, formula=None):
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else f"with chemical formula {formula}:"}
            {nmr_data}
            What's the molecule's name?
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Format the final answer like this - 
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
        """
# Chain of thought prompt
def cot_prompt(nmr_data, formula=None):
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
# Chain of thought prompt with logic tips
def logic_tips_prompt(nmr_data, formula=None):
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
# Chain of thought prompt with expert tips
def expert_tips_prompt(nmr_data, formula=None):
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

# Chain of thought prompt with both expert and logic tips
def expert_logic_tips_prompt(nmr_data, formula=None):
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

# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #

# System prompt for all conversations
SYSTEM_PROMPT = """You are an expert in NMR spectroscopy analysis and organic chemistry."""
# Config 
MODEL="gpt-4-turbo-2024-04-09"
os.environ['OPENAI_API_KEY'] = "your_own_api_key"  # Replace with your actual API key

# zip file handling
def setup_files():
    """Extract zip file if needed and ensure files are accessible"""
    zip_name = "H1 NMR SPECTRA TASKS OAI TEAM_N.zip"
    extract_dir = "H1 NMR SPECTRA TASKS OAI TEAM_N"
    
    # Clean up any existing extracted files
    if os.path.exists(extract_dir):
        print("Removing existing extracted files...")
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(extract_dir)
    
    # Extract fresh copy
    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Files extracted successfully")

# Then modify your paths section to:
setup_files()  # Call this before defining paths

DATA_DIR = Path("H1 NMR SPECTRA TASKS OAI TEAM_N/normalized_files")
BENCHMARK_BASE = Path("./BENCHMARK")
BENCHMARKS_DIR = BENCHMARK_BASE / "GPT" / "RESULTS"

CHALLENGE_FILES = [
    DATA_DIR / "NNMR_spectra_EASY.json",
    DATA_DIR / "NNMR_spectra_MEDIUM.json",
    DATA_DIR / "NNMR_spectra_HARD.json"
]

PROMPT_TYPES = {
    "base": base_prompt,
    "cot": cot_prompt,
    "logic": logic_tips_prompt,
    "expert": expert_tips_prompt,
    "expert_logic": expert_logic_tips_prompt
}
TEMPERATURE=[0, 0.5,0.8, 1.0]
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #



# Extract and validate NMR data from JSON file
def json_extractor(filename):
    try:
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

# Validate molecule data structure
def validate_molecule_data(molecule):
    required_fields = ['formula', 'nmr_challenge_id', 'peaks']
    return all(field in molecule for field in required_fields) and isinstance(molecule['peaks'], list) and all(validate_peak_data(peak) for peak in molecule['peaks'])

# Validate peak data structure
def validate_peak_data(peak):
    required_peak_fields = [
        'name', 'shift', 'range', 'hydrogens',
        'integral', 'class', 'j_values'
    ]
    return all(field in peak for field in required_peak_fields)

# Format NMR data for model input
def token_formater(data, include_formula=True):
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
        
    except Exception as e:
        logging.error(f"Error formatting data: {str(e)}")
        return ""

def call_openai(prompt, model=MODEL, temperature=1):
    try:
        client = OpenAI()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return completion.choices[0].message.content.replace("\n","")
        
    except Exception as e:
        logging.error(f"OpenAI API Error: {str(e)}")
        raise

# Generate output file path based on parameters
def get_output_path(difficulty, model, prompt_type, temperature, with_formula=True):
    form_type = "WITH_FORMULA" if with_formula else "NO_FORMULA"
    base_path = BENCHMARKS_DIR / MODEL / difficulty / form_type
    filename = f"{difficulty}-results-T{temperature}-{prompt_type}-{model}.csv"
    return base_path / filename

# Create benchmark directories if they don't exist
def ensure_benchmark_dirs():
    for difficulty in ["EASY", "MEDIUM", "HARD"]:
        for form_type in ["WITH_FORMULA", "NO_FORMULA"]:
            path = BENCHMARKS_DIR/ MODEL / difficulty / form_type
            path.mkdir(parents=True, exist_ok=True)


# Run batch processing of NMR spectra analysis
def run_batch(temperature, challenge_file, outputfile, prompt_type, include_formula=True, model=MODEL):
    molecules = json_extractor(challenge_file)
    results = []
    
    for molecule in tqdm(molecules, desc=f"Processing {challenge_file.stem}"):
        try:
            # Format NMR data with include_formula parameter
            nmr_data = token_formater(molecule, include_formula=include_formula)
            # Prepare prompt
            prompt_func = PROMPT_TYPES[prompt_type]
            formula = molecule.get('formula') if include_formula else None
            prompt = prompt_func(nmr_data, formula)
            
            # Make API call
            prediction = call_openai(prompt, model=model, temperature=temperature)
            
            results.append({
                "Id": molecule['nmr_challenge_id'],
                "Formula": molecule.get('formula', 'N/A'),
                "PromptType": prompt_type,
                "Prediction": f"{prediction}"
 
            })
            
        except Exception as e:
            logging.error(f"Error processing molecule {molecule.get('nmr_challenge_id')}: {str(e)}")
            raise
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(outputfile, index=False)
    logging.info(f"Results saved to {outputfile}")
    return results

def test_loop_systems(
    n_molecules,
    difficulties,
    temperatures,
    prompt_types,
    include_formula,
    model=MODEL
):
    """Test all looping systems with specified parameters"""
    
    # Create directories
    ensure_benchmark_dirs()
    
    # Calculate total API calls
    total_molecules = 0
    if n_molecules is None:
        for difficulty in difficulties:
            idx = 0 if difficulty == "EASY" else (1 if difficulty == "MEDIUM" else 2)
            molecules = json_extractor(CHALLENGE_FILES[idx])
            total_molecules += len(molecules)
    else:
        total_molecules = n_molecules * len(difficulties)
    
    total_calls = (
        total_molecules * 
        len(temperatures) * 
        len(prompt_types) *
        len(include_formula)
    )
    
    print(f"\nTest Configuration:")
    print(f"Number of molecules per file: {'All' if n_molecules is None else n_molecules}")
    print(f"Difficulties: {difficulties}")
    print(f"Temperatures: {temperatures}")
    print(f"Prompt types: {prompt_types}")
    print(f"Formula conditions: {['With Formula' if f else 'No Formula' for f in include_formula]}")
    print(f"Total API calls needed: {total_calls}")
    
    # Ask for confirmation
    print(f"\nThis will make {total_calls} API calls.\n\n\n")
    results_summary = []
    try:
        for difficulty in difficulties:
            # Get corresponding challenge file
            if difficulty == "EASY":
                challenge_file = CHALLENGE_FILES[0]
            elif difficulty == "MEDIUM":
                challenge_file = CHALLENGE_FILES[1]
            else:
                challenge_file = CHALLENGE_FILES[2]
            
            # Load molecules
            all_molecules = json_extractor(challenge_file)
            test_molecules = all_molecules[:n_molecules] if n_molecules else all_molecules
            
            # Save test molecules to temporary file
            test_file = Path(f"./temp_{difficulty}_test.json")
            with open(test_file, "w") as f:
                json.dump(test_molecules, f)
            
            for formula in include_formula:
                for temp in temperatures:
                    for prompt_type in prompt_types:
                        try:
                            print(f"\nProcessing: {difficulty}, {'With' if formula else 'No'} Formula, T={temp}, Prompt={prompt_type}")
                            
                            # Get output path
                            output_file = get_output_path(
                                difficulty=difficulty,
                                model=model.replace("-", "_"),
                                prompt_type=prompt_type,
                                temperature=temp,
                                with_formula=formula
                            )
                            
                            # Run batch
                            results = run_batch(
                                temperature=temp,
                                challenge_file=test_file,
                                outputfile=output_file,
                                prompt_type=prompt_type,
                                include_formula=formula,
                                model=model
                            )
                            
                            # Store summary
                            results_summary.append({
                                'difficulty': difficulty,
                                'with_formula': formula,
                                'temperature': temp,
                                'prompt_type': prompt_type,
                                'n_processed': len(results),
                                'n_errors': sum(1 for r in results if r['Prediction'] == 'ERROR'),
                                'output_file': str(output_file)
                            })
                            
                        except Exception as e:
                            logging.error(f"Error in combination - Difficulty: {difficulty}, "
                                        f"Formula: {formula}, Temp: {temp}, Prompt: {prompt_type}")
                            logging.error(str(e))
            
            # Clean up temporary file
            test_file.unlink()
    
    except Exception as e:
        logging.error(f"Test loop error: {str(e)}")
        raise
    
    finally:
        # Print summary
        print("\n=== Test Summary ===")
        for result in results_summary:
            print(f"Formula: {'With' if result['with_formula'] else 'Without'}")
            print(f"\nDifficulty: {result['difficulty']}")
            print(f"Temperature: {result['temperature']}")
            print(f"Prompt Type: {result['prompt_type']}")
            print(f"Molecules Processed: {result['n_processed']}")
            print(f"Errors: {result['n_errors']}")
            print(f"Output File: {result['output_file']}")
        
        return results_summary


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    try:
        test_loop_systems(
        n_molecules=None,
        difficulties=["EASY", "MEDIUM", "HARD"],
        temperatures=TEMPERATURE,
        prompt_types=["base", "cot", "logic", "expert", "expert_logic"],
        include_formula=[True, False]
    )
    except Exception as e:
        logging.error(f"error: {str(e)}")
