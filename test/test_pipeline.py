"""
Test script for the PEIRCE + LMM pipeline.

This script tests the pipeline to ensure it works correctly with different configurations.
"""

import os
import sys
import unittest
import json
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

class TestPipeline(unittest.TestCase):
    """
    Test case for the PEIRCE + LMM pipeline.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create a sample factual statement
        self.sample_statements = [
            {
                "id": "TEST001",
                "factual_statement": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
                "topic": "science"
            }
        ]
    
    def tearDown(self):
        """
        Clean up after the test case.
        """
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_generate_creative_prompts(self):
        """
        Test that the pipeline correctly generates creative prompts.
        """
        # Initialize the pipeline
        pipeline = UrbanLMMPipeline(
            model_name="llama3",
            provider="ollama",
            max_iterations=1,
            min_urban_terms=2,
            output_dir=self.test_dir
        )
        
        # Generate creative prompts
        prompts = pipeline.generate_creative_prompts(self.sample_statements)
        
        # Verify that the prompts were generated correctly
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]["id"], "TEST001_RAP")
        self.assertEqual(prompts[0]["factual_statement_id"], "TEST001")
        self.assertEqual(prompts[0]["prompt_type"], "rap")
        self.assertEqual(prompts[1]["id"], "TEST001_POEM")
        self.assertEqual(prompts[1]["factual_statement_id"], "TEST001")
        self.assertEqual(prompts[1]["prompt_type"], "urban_poem")
    
    def test_load_factual_statements(self):
        """
        Test that the pipeline correctly loads factual statements.
        """
        # Create a temporary file with factual statements
        statements_file = os.path.join(self.test_dir, "test_statements.json")
        with open(statements_file, "w") as f:
            json.dump(self.sample_statements, f)
        
        # Initialize the pipeline
        pipeline = UrbanLMMPipeline(
            model_name="llama3",
            provider="ollama",
            max_iterations=1,
            min_urban_terms=2,
            output_dir=self.test_dir
        )
        
        # Load factual statements
        statements = pipeline.load_factual_statements(statements_file)
        
        # Verify that the statements were loaded correctly
        self.assertEqual(len(statements), 1)
        self.assertEqual(statements[0]["id"], "TEST001")
        self.assertEqual(statements[0]["factual_statement"], "Water freezes at 0 degrees Celsius at standard atmospheric pressure.")
        self.assertEqual(statements[0]["topic"], "science")
    
    def test_pipeline_initialization(self):
        """
        Test that the pipeline initializes correctly with different configurations.
        """
        # Test with default settings
        pipeline = UrbanLMMPipeline()
        self.assertEqual(pipeline.model_name, "gpt-oss")
        self.assertEqual(pipeline.provider, "ollama")
        self.assertEqual(pipeline.max_iterations, 3)
        self.assertEqual(pipeline.min_urban_terms, 2)
        
        # Test with custom settings
        pipeline = UrbanLMMPipeline(
            model_name="llama3",
            provider="ollama",
            max_iterations=5,
            min_urban_terms=3,
            output_dir=self.test_dir
        )
        self.assertEqual(pipeline.model_name, "llama3")
        self.assertEqual(pipeline.provider, "ollama")
        self.assertEqual(pipeline.max_iterations, 5)
        self.assertEqual(pipeline.min_urban_terms, 3)
        self.assertEqual(pipeline.output_dir, self.test_dir)
    
    def test_save_pipeline_results(self):
        """
        Test that the pipeline correctly saves results.
        """
        # Initialize the pipeline
        pipeline = UrbanLMMPipeline(
            model_name="llama3",
            provider="ollama",
            max_iterations=1,
            min_urban_terms=2,
            output_dir=self.test_dir
        )
        
        # Create sample results
        results = {
            "factual_statements": self.sample_statements,
            "creative_prompts": pipeline.generate_creative_prompts(self.sample_statements),
            "outputs": [],
            "iterations": []
        }
        
        # Save results
        pipeline.save_pipeline_results(results)
        
        # Verify that the files were created
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith("factual_statements_") for f in files))
        self.assertTrue(any(f.startswith("creative_prompts_") for f in files))
        self.assertTrue(any(f.startswith("outputs_") for f in files))
        self.assertTrue(any(f.startswith("iterations_") for f in files))
        self.assertTrue(any(f.startswith("pipeline_results_") for f in files))

if __name__ == "__main__":
    unittest.main()
