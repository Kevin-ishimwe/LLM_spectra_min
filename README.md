# NMR Analysis Script

This script automates the analysis of ¹H NMR spectra using OpenAI's API. It processes NMR data provided in a structured format, applies different prompts for molecule identification, and generates benchmarks for the results.


## Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed.
2. **Required Libraries**: Install the following packages before running the script:
   ```zsh
    pip install pandas==2.2.2 tqdm==4.66.5 openai==1.52.0
   ```
3. **API Key**: Obtain your OpenAI API key and keep it ready.


## How to Use

### Step 1: Setup
Place the zip file named `H1 NMR Datasets` in the same directory as the script. This is the file structure  containing the task JSON files:
```
H1 NMR Datasets/
└── Challenges/
    ├── Hnmr_spectra_easy.json
    ├── Hnmr_spectra_medium.json
    └── Hnmr_spectra_hard.json
```

### Step 2: Configuration
Before running the script, update the following:

1. **API Key**: Replace the placeholder in the script with your OpenAI API key:
   ```python
    os.environ['OPENAI_API_KEY'] = "your-api-key-here"
   ```

2. **Model Selection**: Change the GPT model to use by updating this line for every experiment:
   ```python
   MODEL = "gpt-4-turbo-2024-04-09"
   ```

### Step 3: Execution
Run the script in your terminal:
```bash
    python3 llm_script.py
```

## Features

1. **Automated Processing**:
   - Extracts the dataset file and organizes its contents.
   - Creates a `BENCHMARK` folder to store results.

2. **Prompts for Analysis**:
   The script includes multiple prompt styles to test NMR data interpretation:
   - **Base Prompt**: Direct identification of the molecule.
   - **Chain of Thought (CoT)**: Encourages step-by-step reasoning.
   - **Logic Prompt**: Focuses on logical peak and fragment analysis.
   - **Expert Prompt**: Utilizes expert knowledge of NMR regions.
   - **Expert + Logic**: Combines expert knowledge and logical analysis.

3. **Testing Configurations**:
   Each prompt is tested under the following conditions:
   - **Temperatures**: 0, 0.5, 0.8, 1.0.
   - **Molecular Formula**: Included and excluded scenarios.


## Example Output

- The script formats predictions as:
  ```csv
    Id,Formula,PromptType,Prediction
    143,C9H10O,base,### Start answer ### Ethyl benzoate ### End answer###
    ```


## Logging
The script provides detailed logging for:
- Data extraction and validation.
- Benchmarking progress.
- Errors during execution.

## File Structure

- **Script Directory**:
  - `H1 NMR Datasets` 
  - `nmr_analysis_script.py`

- **Generated Folders**:
  - `BENCHMARK` 

## Notes

1. **Ensure Compatibility**: Use the exact library versions mentioned in the prerequisites.
2. **Model Selection**: The default model is `gpt-4-turbo-2024-04-09`. Experiment with  newer models as available.
3. **API Limits**: Ensure your OpenAI account has sufficient quota to run multiple API requests.

