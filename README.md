# PEIRCE + LMM Pipeline Tests

This folder contains tests for the PEIRCE + LMM pipeline. The tests verify that the pipeline works correctly with different configurations and inputs.

## Test Files

- `test_model_factory.py`: Tests the model_factory module to ensure it correctly handles different model configurations.
- `test_pipeline.py`: Tests the basic functionality of the pipeline without running the full pipeline.
- `test_pipeline_e2e.py`: End-to-end test that runs the pipeline with a small sample to verify that it works correctly.

## Running the Tests

You can run the tests using the Python unittest framework:

```bash
# Run all tests
python -m unittest discover -s test

# Run a specific test file
python -m unittest test.test_model_factory
python -m unittest test.test_pipeline
python -m unittest test.test_pipeline_e2e

# Run a specific test case
python -m unittest test.test_model_factory.TestModelFactory.test_load_config
```

## Test Requirements

The tests require the following:

- Python 3.6+
- The same dependencies as the main pipeline (see the project's requirements.txt)
- For the end-to-end test, you need a working Ollama installation with the llama3 model pulled

## Test Environment

The tests are designed to work in a test environment without proper credentials. If the pipeline fails due to missing models or API keys, that's expected in a test environment and the tests will handle it gracefully.

## Test Coverage

The tests cover the following aspects of the pipeline:

### Model Factory Tests

- Loading the configuration
- Creating an Ollama instance with a valid model
- Creating an Ollama instance with an invalid model (should raise an error)
- Creating a GPT instance with a valid model
- Creating a GPT instance with an invalid model (should raise an error)
- Creating an instance with an invalid provider (should raise an error)

### Pipeline Tests

- Generating creative prompts
- Loading factual statements
- Pipeline initialization with different configurations
- Saving pipeline results

### End-to-End Tests

- Running the pipeline with a small sample
- Verifying that the results contain the expected keys
- Verifying that the output files are created

## Adding New Tests

To add a new test, create a new test file in this folder and follow the unittest framework conventions. Make sure to import the necessary modules and add the project root to the Python path:

```python
import os
import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the modules to test
from module.to.test import ClassToTest

class TestNewFeature(unittest.TestCase):
    def test_something(self):
        # Test code here
        pass
```

## Test Data

The tests use sample data created within the test files. If you need to add more test data, you can create it in the setUp method of your test case or add it to the test/data folder.