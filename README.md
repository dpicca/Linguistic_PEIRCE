# PEIRCE + LMM Pipeline Runner

This module implements the complete pipeline for the PEIRCE + LMM project, integrating all components into a unified workflow for transforming factual statements into creative urban-style text.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Configuration Options](#configuration-options)
- [Output Format](#output-format)
- [Components in Detail](#components-in-detail)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

## Pipeline Overview

The pipeline consists of the following steps:

1. **Input**: Factual statement (id, text, topic)
2. **Prompt**: Generated rap / urban poem prompt, linked to the id
3. **Generation**: LLM produces creative version (with access to the knowledge base)
4. **Hard Critique**:
   - Mapping of terms used in the creative output to the knowledge base
   - Check for required elements and semantic fidelity to the fact
5. **Soft Critique**:
   - Calculation of parsimony, coherence, uncertainty, fluency, cultural fit
6. **Feedback & Refinement**:
   - If any critique fails, a new generation → critique → feedback iteration is launched
   - Process continues until preset thresholds are met or maximum iterations reached
7. **Output**:
   - Final creative version
   - Critique scores
   - Log of iterations
   - Symbolic mappings (for subsequent analysis)

## Project Structure

The PEIRCE + LMM project is organized into several key directories:

```
peirce/
├── pipeline/           # Main pipeline implementation
├── generation/         # Text generation components
├── critique/           # Critique components for evaluation
├── refinement/         # Refinement components for improvement
├── prompt/             # Prompt templates and models
├── data/               # Input data and output results
│   └── pipeline_outputs/  # Pipeline output files
└── test/               # Test scripts and documentation
```

### Key Files

- `pipeline/urban_lmm_pipeline.py`: Main pipeline implementation
- `generation/urban_lmm.py`: Text generation with urban slang
- `critique/urban_lmm_hard_critique.py`: Formal/symbolic checks
- `critique/urban_lmm_soft.py`: Linguistic/epistemic checks
- `refinement/urban_lmm_refinement.py`: Iterative refinement mechanism
- `config.yaml`: Configuration for LLM providers and models

## Pipeline Architecture

The pipeline integrates several components that work together to transform factual statements into creative urban-style text:

1. **UrbanLMMPipeline**: The main pipeline class that orchestrates the entire process
2. **UrbanLMMGenerator**: Generates creative text with urban slang
3. **UrbanLMMHardCritique**: Performs formal/symbolic checks on the generated text
4. **UrbanLMMSoftCritique**: Performs linguistic/epistemic checks on the generated text
5. **UrbanLMMRefinement**: Refines the generated text based on critique feedback

### Pipeline Diagram

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Factual         |     |  Creative        |     |  Generated       |
|  Statements      +---->+  Prompts         +---->+  Output          |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +--------+---------+
                                                           |
                                                           v
+------------------+     +------------------+     +--------+---------+
|                  |     |                  |     |                  |
|  Final           |     |  Refined         |     |  Critique        |
|  Output          +<----+  Output          +<----+  Results         |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +--------+---------+
                                 ^                         |
                                 |                         |
                                 |    +------------------+ |
                                 |    |                  | |
                                 +----+  Refinement      +-+
                                      |  Feedback        |
                                      |                  |
                                      +------------------+
```

### Data Flow

1. Factual statements are loaded or provided as input
2. Creative prompts are generated from the factual statements
3. The generator produces creative text based on the prompts
4. The hard critique evaluates the text for urban terms and semantic fidelity
5. The soft critique evaluates the text for parsimony, coherence, uncertainty, etc.
6. If any critique fails, the refinement component generates a new prompt with feedback
7. The generator produces refined text based on the new prompt
8. The process repeats until both critiques pass or maximum iterations reached
9. The final output is saved to the output directory

## Installation and Setup

### Requirements

- Python 3.6+
- OpenAI API key (or other LLM provider credentials)
- spaCy with the `en_core_web_sm` model
- Transformers library (for soft critique components)
- Other dependencies as specified in the project's requirements.txt

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/peirce.git
   cd peirce
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Configure your API keys in `config.yaml`:
   ```yaml
   gpt-4o:
     engine: gpt-4o
     api_key: your_openai_api_key

   ollama:
     api_url: 'http://localhost:11434'
     available_models:
       - llama3
       - mistral
       - gemma
   ```

## Usage

### Basic Usage

```python
from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

# Initialize the pipeline
pipeline = UrbanLMMPipeline(
    model_name="llama3",       # Specify the model to use
    provider="ollama",         # Specify the provider
    max_iterations=3,          # Maximum number of refinement iterations
    min_urban_terms=2,         # Minimum number of urban terms required
    output_dir=None            # Use default output directory
)

# Run the pipeline with default settings (processes first 3 factual statements)
pipeline_results = pipeline.run_pipeline(num_samples=3)
```

### Command Line Usage

You can also run the pipeline from the command line:

```bash
# Run with default settings (processes first 3 factual statements)
python run_pipeline.py

# Run with custom settings
python run_pipeline.py --model gpt-4o --provider openai --max-iterations 5 --min-urban-terms 3 --num-samples 10

# Get help
python run_pipeline.py --help
```

### Custom Factual Statements and Prompts

You can provide your own factual statements and creative prompts:

```python
from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

# Initialize the pipeline
pipeline = UrbanLMMPipeline(
    model_name="llama3",
    provider="ollama"
)

# Create custom factual statements
custom_statements = [
    {
        "id": "CUSTOM001",
        "factual_statement": "Your factual statement here.",
        "topic": "your_topic"
    }
]

# Run the pipeline with custom factual statements
pipeline_results = pipeline.run_pipeline(factual_statements=custom_statements)

# Or generate creative prompts first and then run the pipeline
creative_prompts = pipeline.generate_creative_prompts(custom_statements)
pipeline_results = pipeline.run_pipeline(
    factual_statements=custom_statements,
    creative_prompts=creative_prompts
)
```

### Advanced Usage: Custom Input and Output Files

```bash
# Run the pipeline with custom input and output files
python run_pipeline.py --input-file my_facts.json --output-dir my_outputs --prompts-file my_prompts.json
```

### Advanced Usage: Verbose Mode

```bash
# Run the pipeline in verbose mode to see detailed progress
python run_pipeline.py --verbose
```

## Configuration Options

### Model Selection

The pipeline supports different LLM models:

- **OpenAI models**: gpt-3.5-turbo, gpt-4o, gpt-4o-mini
- **Ollama models**: llama3, mistral, gemma

You can specify the model using the `model_name` parameter:

```python
from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

pipeline = UrbanLMMPipeline(model_name="gpt-4o")
```

Or from the command line:

```bash
python run_pipeline.py --model gpt-4o
```

### Provider Selection

The pipeline supports different LLM providers:

- **OpenAI**: For OpenAI models
- **Ollama**: For local models using Ollama

You can specify the provider using the `provider` parameter:

```python
from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

pipeline = UrbanLMMPipeline(provider="openai")
```

Or from the command line:

```bash
python run_pipeline.py --provider openai
```

### Refinement Parameters

You can customize the refinement process:

- **max_iterations**: Maximum number of refinement iterations
- **min_urban_terms**: Minimum number of urban terms required in the output

```python
from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

pipeline = UrbanLMMPipeline(max_iterations=5, min_urban_terms=3)
```

Or from the command line:

```bash
python run_pipeline.py --max-iterations 5 --min-urban-terms 3
```

### Output Customization

You can specify a custom output directory:

```python
from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

pipeline = UrbanLMMPipeline(output_dir="my_outputs")
```

Or from the command line:

```bash
python run_pipeline.py --output-dir my_outputs
```

## Output Format

The pipeline produces the following outputs:

1. **Factual Statements**: The input factual statements
2. **Creative Prompts**: The prompts generated from the factual statements
3. **Outputs**: The final creative versions after refinement
4. **Iterations**: Detailed logs of the refinement process

Each output includes:
- Hard critique results (term mapping, semantic fidelity)
- Soft critique results (parsimony, coherence, uncertainty, etc.)
- Feedback provided for improvement
- Final refined output

### Output Files

The pipeline saves the following files to the output directory:

- `factual_statements_[timestamp].json`: Input factual statements
- `creative_prompts_[timestamp].json`: Generated creative prompts
- `outputs_[timestamp].json`: Final creative versions
- `iterations_[timestamp].json`: Detailed logs of the refinement process
- `pipeline_results_[timestamp].json`: Complete pipeline results

### JSON Structure Example

```json
{
  "factual_statements": [
    {
      "id": "S001",
      "factual_statement": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
      "topic": "science"
    }
  ],
  "creative_prompts": [
    {
      "id": "S001_RAP",
      "factual_statement_id": "S001",
      "prompt_type": "rap",
      "prompt": "Rewrite this fact as a rap using urban slang and hip-hop terminology: 'Water freezes at 0 degrees Celsius at standard atmospheric pressure.'",
      "topic": "science"
    }
  ],
  "outputs": [
    {
      "id": "S001_RAP_OUTPUT",
      "prompt_id": "S001_RAP",
      "factual_statement_id": "S001",
      "prompt_type": "rap",
      "topic": "science",
      "output": "Yo, listen up, this is no cap,\nWhen the temp hits zero C, that H2O gets trapped,\nAt standard pressure, it's a scientific fact,\nWater freezes solid, that's the exact impact,\nIce forming crystal clear, that's straight fire,\nFreezing point knowledge, let me take you higher,\nZero degrees Celsius, that's the magic number,\nStandard atmospheric pressure, no need to wonder,\nThis is basic science, but I'm making it lit,\nH2O transformation, that's some cold hard drip!"
    }
  ],
  "iterations": [
    {
      "original_prompt": "Rewrite this fact as a rap using urban slang and hip-hop terminology: 'Water freezes at 0 degrees Celsius at standard atmospheric pressure.'",
      "factual_statement": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
      "final_output": "Yo, listen up, this is no cap,\nWhen the temp hits zero C, that H2O gets trapped,\nAt standard pressure, it's a scientific fact,\nWater freezes solid, that's the exact impact,\nIce forming crystal clear, that's straight fire,\nFreezing point knowledge, let me take you higher,\nZero degrees Celsius, that's the magic number,\nStandard atmospheric pressure, no need to wonder,\nThis is basic science, but I'm making it lit,\nH2O transformation, that's some cold hard drip!",
      "best_output": "Yo, listen up, this is no cap,\nWhen the temp hits zero C, that H2O gets trapped,\nAt standard pressure, it's a scientific fact,\nWater freezes solid, that's the exact impact,\nIce forming crystal clear, that's straight fire,\nFreezing point knowledge, let me take you higher,\nZero degrees Celsius, that's the magic number,\nStandard atmospheric pressure, no need to wonder,\nThis is basic science, but I'm making it lit,\nH2O transformation, that's some cold hard drip!",
      "best_score": 0.85,
      "iterations": [
        {
          "iteration": 1,
          "output": "Yo, listen up, this is no cap,\nWhen the temp hits zero C, that H2O gets trapped,\nAt standard pressure, it's a scientific fact,\nWater freezes solid, that's the exact impact,\nIce forming crystal clear, that's straight fire,\nFreezing point knowledge, let me take you higher,\nZero degrees Celsius, that's the magic number,\nStandard atmospheric pressure, no need to wonder,\nThis is basic science, but I'm making it lit,\nH2O transformation, that's some cold hard drip!",
          "hard_critique": {
            "passed": true,
            "urban_terms_pass": true,
            "urban_term_count": 4,
            "terms_found": ["no cap", "fire", "lit", "drip"],
            "structure_preserved": true,
            "missing_entities": [],
            "feedback": []
          },
          "soft_critique": {
            "passed": true,
            "overall_score": 0.85,
            "feedback": []
          },
          "combined_score": 0.85
        }
      ]
    }
  ]
}
```

## Components in Detail

### Generation Component

The generation component (`UrbanLMMGenerator`) is responsible for generating creative text with urban slang. It:

1. Loads urban dictionary terms from a file or uses fallback terms
2. Enhances prompts with urban terms
3. Generates text using the specified LLM model and provider
4. Handles different model providers (OpenAI, Ollama)

### Hard Critique Component

The hard critique component (`UrbanLMMHardCritique`) performs formal/symbolic checks on the generated text. It:

1. Maps creative terms to concepts in the knowledge base
2. Verifies that the creative text includes at least a minimum number of urban terms
3. Checks that the logical structure of the factual statement is respected
4. Provides feedback if the verification fails

### Soft Critique Component

The soft critique component (`UrbanLMMSoftCritique`) performs linguistic/epistemic checks on the generated text. It:

1. Assesses semantic density (parsimony)
2. Evaluates coherence using textual entailment
3. Analyzes uncertainty (hedging expressions)
4. Evaluates fluency (readability, grammatical correctness)
5. Assesses cultural fit with respect to urban slang
6. Provides feedback for improvement

### Refinement Component

The refinement component (`UrbanLMMRefinement`) refines the generated text based on critique feedback. It:

1. Generates initial output
2. Critiques the output using hard and soft critiques
3. Calculates combined score
4. Generates refinement prompt based on feedback
5. Generates refined output
6. Repeats until both critiques pass or max iterations reached

## Troubleshooting

### Common Issues

#### API Key Issues

If you encounter API key issues:

1. Make sure your API keys are correctly set in `config.yaml`
2. Check that you have the necessary permissions for the API
3. Verify that your API key is not expired

#### Model Not Found

If you encounter "Model not found" errors:

1. Check that the model is correctly specified in `config.yaml`
2. For Ollama models, make sure the model is pulled:
   ```bash
   ollama pull llama3
   ```
3. Verify that the model name is spelled correctly

#### Critique Component Initialization Failures

If critique components fail to initialize:

1. Make sure all dependencies are installed
2. Check that the spaCy model is installed:
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. For coherence critique, make sure you have access to the required Hugging Face models

#### Output Directory Issues

If you encounter output directory issues:

1. Make sure the output directory exists
2. Check that you have write permissions for the directory
3. Verify that there is enough disk space

### Error Handling

The pipeline includes comprehensive error handling to ensure that it can still provide useful results even if some components fail. If a critical error occurs, the pipeline will:

1. Log the error
2. Try to continue with fallback mechanisms
3. Provide as much output as possible

## Testing

A test script is provided to demonstrate the usage of the pipeline:

```bash
python -m pipeline.test_pipeline
```

This script creates a small sample of factual statements, generates creative prompts, and explains how to run the full pipeline.

For more comprehensive testing, you can use the test scripts in the `test` directory:

```bash
python -m unittest discover -s test
```

These tests verify that the pipeline works correctly with different configurations and inputs.