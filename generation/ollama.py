"""
Ollama integration module.
This module provides a class for interacting with the Ollama API using the official ollama package.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Union, Iterator

import ollama
from prompt.prompt_model import PromptModel

# Set up logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

class Ollama:
    """
    A class for interacting with the Ollama API using the official ollama package.
    """

    def __init__(self, model_name: str, api_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama class.
        
        Args:
            model_name (str): The name of the model to use (e.g., "llama3").
            api_url (str, optional): The URL of the Ollama API. Defaults to "http://localhost:11434".
        """
        self.model_name = model_name
        self.api_url = api_url
        self.client = ollama.Client(host=api_url)
        self.prompt_model = PromptModel()
        logger.info(f"Initialized Ollama with model: {model_name}, API URL: {api_url}")

    def generate(self, model_prompt_dir: str, prompt_name: str, **kwargs) -> str:
        """
        Generate text using the Ollama API.
        
        Args:
            model_prompt_dir (str): The directory containing the prompt templates.
            prompt_name (str): The name of the prompt template to use.
            **kwargs: Additional arguments to pass to the prompt template.
            
        Returns:
            str: The generated text.
        """
        logger.info(f"Generating text with model: {self.model_name}, prompt: {prompt_name}")
        
        # Process the prompt using the prompt model
        system_prompt, user_prompt = self.prompt_model.process_prompt(
            model_prompt_dir=model_prompt_dir,
            prompt_name=prompt_name,
            **kwargs
        )
        
        # Print raw result for testing if requested
        if kwargs.get('test', False):
            logger.info(f"System prompt: {system_prompt}")
            logger.info(f"User prompt: {user_prompt}")
        
        try:
            # Generate text using the ollama package
            response = self.client.generate(
                model=self.model_name,
                prompt=user_prompt,
                system=system_prompt,
                stream=False
            )
            
            # Extract the content from the response
            content = response.response
            
            # Print raw result for testing if requested
            if kwargs.get('test', False):
                logger.info(f"Raw response: {response}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion using the Ollama API.
        
        Args:
            messages (List[Dict[str, str]]): A list of messages in the format:
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            **kwargs: Additional arguments to pass to the Ollama API.
            
        Returns:
            Dict[str, Any]: The chat completion response in the format:
                {"message": {"role": "assistant", "content": "..."}}
        """
        logger.info(f"Generating chat completion with model: {self.model_name}")
        
        try:
            # Convert messages to the format expected by the ollama package
            ollama_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            
            # Generate chat completion using the ollama package
            response = self.client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False
            )
            
            # Format the response to match the expected format
            result = {
                "message": {
                    "role": "assistant",
                    "content": response.message["content"]
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            
            # If there's an error, try to parse the response as a string
            if isinstance(e, ollama.RequestError) and hasattr(e, 'response') and hasattr(e.response, 'text'):
                response_text = e.response.text
                content = self._extract_content_from_response(response_text)
                return {"message": {"role": "assistant", "content": content}}
            
            raise

    def _extract_content_from_response(self, response_text: str) -> str:
        """
        Extract the content from a response text.
        This method handles different response formats, including JSON and plain text.
        
        Args:
            response_text (str): The response text to parse.
            
        Returns:
            str: The extracted content.
        """
        if not response_text:
            return ""
        
        response_text = response_text.strip()
        
        # Try to parse the entire response as a single JSON object
        try:
            result = json.loads(response_text)
            if isinstance(result, dict) and "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            return response_text
        except json.JSONDecodeError:
            # If that fails, try to extract a valid JSON object
            try:
                # Look for a complete JSON object pattern
                json_pattern = re.compile(r'(\{.*\})', re.DOTALL)
                match = json_pattern.search(response_text)
                
                if match:
                    # Try to parse the matched JSON object
                    json_str = match.group(1)
                    result = json.loads(json_str)
                    if isinstance(result, dict) and "message" in result and "content" in result["message"]:
                        return result["message"]["content"]
                
                # If we have multiple JSON objects (streaming response), concatenate their contents
                json_objects = re.findall(r'(\{.*?\})', response_text, re.DOTALL)
                if json_objects:
                    content = ""
                    for json_obj in json_objects:
                        try:
                            obj = json.loads(json_obj)
                            if isinstance(obj, dict) and "message" in obj and "content" in obj["message"]:
                                content += obj["message"]["content"]
                        except json.JSONDecodeError:
                            pass
                    if content:
                        return content
                
                # If no JSON object is found or parsing fails, return the response as is
                return response_text
            except Exception:
                # If all parsing attempts fail, return the response as is
                return response_text