import os
import json
import time
import logging
import glob
import pandas as pd
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DarkWebLocalAnalyzer:
    """Analyze dark web data from local directory using open-source models."""
    
    def __init__(self):
        """Initialize the analyzer with publicly available models."""
        self.models_config = [
            {
                "name": "tiny-llama",
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "tasks": ["text_analysis", "transaction_detection"]
            },
            {
                "name": "phi-2",
                "model_id": "microsoft/phi-2",
                "tasks": ["listing_detection", "price_extraction"]
            }
        ]
        
        self.clip_model_id = "openai/clip-vit-base-patch32"
        self.models = {}
        self.tokenizers = {}
        self.image_model = None
        self.image_processor = None
        self.transaction_history = []
        logger.info(f"Initialized DarkWebLocalAnalyzer with {len(self.models_config)} model configurations")
        
    def load_text_model(self, model_name: str) -> bool:
        """Load a specific text analysis model by name."""
        # Find the model config
        model_config = next((m for m in self.models_config if m["name"] == model_name), None)
        if not model_config:
            logger.error(f"Model {model_name} not found in configuration")
            return False
        
        try:
            logger.info(f"Loading model: {model_name} ({model_config['model_id']})")
            # Load tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_config["model_id"])
            
            # Load model
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_config["model_id"],
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Successfully loaded {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def load_image_model(self) -> bool:
        """Load the CLIP model for image analysis."""
        try:
            logger.info(f"Loading CLIP model for image analysis: {self.clip_model_id}")
            self.image_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
            self.image_model = CLIPModel.from_pretrained(self.clip_model_id)
            logger.info("Successfully loaded CLIP model")
            return True
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            return False
    
    def analyze_text(self, text: str, model_name: str, task: str, 
                     max_length: int = 1024) -> Dict[str, Any]:
        """
        Analyze text using the specified model and task.
        
        Args:
            text: Text content to analyze
            model_name: Name of the model to use
            task: Type of analysis to perform
            max_length: Maximum token length for generation
            
        Returns:
            Dict containing analysis results
        """
        if model_name not in self.models:
            if not self.load_text_model(model_name):
                return {"error": f"Could not load model {model_name}"}
        
        # Get the model and tokenizer
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Prepare the prompt based on the task
        prompt = self._create_prompt_for_task(text, task)
        
        try:
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.3,  # Lower temperature for more deterministic results
                    do_sample=True
                )
                
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process the response
            result = self._post_process_response(response, task, prompt)
            
            return {
                "task": task,
                "model": model_name,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text with {model_name} for task {task}: {str(e)}")
            return {"error": str(e), "task": task, "model": model_name}
    
    def analyze_image(self, image_path: str, categories: List[str] = None) -> Dict[str, Any]:
        """
        Analyze an image using CLIP model to detect content categories.
        
        Args:
            image_path: Path to the image file
            categories: List of categories to check for (defaults to predefined list)
            
        Returns:
            Dict containing analysis results
        """
        if not self.image_model:
            if not self.load_image_model():
                return {"error": "Could not load image model"}
                
        if not categories:
            # Default categories for dark web marketplace analysis
            categories = [
                "drugs", "medication", "pills", "weapons", "counterfeit", 
                "financial", "hacking tools", "cryptocurrency", "ID document",
                "credit card", "electronics", "digital goods"
            ]
            
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare text descriptions
            texts = [f"an image of {category}" for category in categories]
            
            # Process inputs
            inputs = self.image_processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.image_model(**inputs)
                
            # Get similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze().tolist()
            
            # Prepare results
            results = []
            for category, score in zip(categories, probs):
                results.append({
                    "category": category,
                    "confidence": score
                })
                
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Get the top categories (over 20% confidence)
            top_categories = [r for r in results if r["confidence"] > 0.2]
            
            return {
                "file": image_path,
                "all_categories": results,
                "detected_categories": top_categories,
                "primary_category": results[0] if results else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return {"error": str(e), "file": image_path}
    
    def _create_prompt_for_task(self, text: str, task: str) -> str:
        """Create an appropriate prompt based on the task."""
        prompts = {
            "text_analysis": f"""
            Analyze the following dark web marketplace content:

            {text}

            Provide:
            1. Type of marketplace
            2. Main categories of items
            3. Any notable patterns
            4. Any sensitive or illegal content indicators
            """,
            
            "transaction_detection": f"""
            Analyze the following marketplace content and identify any evidence of transactions or purchases:

            {text}

            List any completed transactions that you can identify, including:
            - What item was purchased
            - The price and currency
            - Buyer and seller identifiers if available
            - When the transaction occurred
            """,
            
            "listing_detection": f"""
            Extract all product or service listings from this marketplace data:

            {text}

            For each listing, provide:
            - Product title
            - Price
            - Seller name
            - Brief description
            """,
            
            "price_extraction": f"""
            Extract all prices and payment methods from the following content:

            {text}

            List all prices found with their corresponding items and accepted payment methods.
            """
        }
        
        return prompts.get(task, f"Analyze the following text:\n\n{text}\n\nAnalysis:")
    
    def _post_process_response(self, response: str, task: str, prompt: str) -> Dict[str, Any]:
        """Process the model's response based on the task type."""
        # Remove the prompt from the response if it's included
        if prompt in response:
            response = response.split(prompt, 1)[1].strip()
        
        # Basic processing - return the raw response for all tasks
        return {"raw_response": response}
    
    def analyze_directory(self, data_dir: str, output_dir: str = "./analysis_results") -> Dict[str, Any]:
        """
        Analyze all data in a directory including HTML files, text files, and images.
        
        Args:
            data_dir: Directory containing dark web data
            output_dir: Directory to save analysis results
            
        Returns:
            Dict containing summary of analysis
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Start time
        start_time = datetime.now()
        logger.info(f"Starting analysis of directory: {data_dir}")
        
        # Initialize counters and results storage
        file_counts = {
            "html": 0,
            "text": 0,
            "image": 0,
            "other": 0
        }
        
        listings = []
        transactions = []
        detected_categories = set()
        image_analysis_results = []
        
        # Process all files in directory and subdirectories
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = file.split('.')[-1].lower()
                
                try:
                    # Process based on file type
                    if file_extension in ['html', 'htm']:
                        file_counts['html'] += 1
                        html_results = self._analyze_html_file(file_path)
                        
                        # Track listings and transactions
                        if 'listings' in html_results:
                            listings.extend(html_results['listings'])
                        if 'transactions' in html_results:
                            transactions.extend(html_results['transactions'])
                            
                        # Save individual results
                        result_file = os.path.join(output_dir, f"html_{os.path.basename(file_path)}.json")
                        with open(result_file, 'w') as f:
                            json.dump(html_results, f, indent=2)
                            
                    elif file_extension in ['txt', 'md', 'csv']:
                        file_counts['text'] += 1
                        text_results = self._analyze_text_file(file_path)
                        
                        # Save individual results
                        result_file = os.path.join(output_dir, f"text_{os.path.basename(file_path)}.json")
                        with open(result_file, 'w') as f:
                            json.dump(text_results, f, indent=2)
                            
                    elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                        file_counts['image'] += 1
                        image_results = self.analyze_image(file_path)
                        image_analysis_results.append(image_results)
                        
                        # Track detected categories
                        if 'detected_categories' in image_results:
                            for category in image_results['detected_categories']:
                                detected_categories.add(category['category'])
                                
                        # Save individual results
                        result_file = os.path.join(output_dir, f"img_{os.path.basename(file_path)}.json")
                        with open(result_file, 'w') as f:
                            json.dump(image_results, f, indent=2)
                            
                    else:
                        file_counts['other'] += 1
                        logger.info(f"Skipping unsupported file type: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Generate summary report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "analysis_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "files_processed": file_counts,
            "listings_found": len(listings),
            "transactions_detected": len(transactions),
            "image_categories_detected": list(detected_categories),
            "listings_sample": listings[:10] if listings else [],
            "transactions_sample": transactions[:10] if transactions else []
        }
        
        # Save summary
        with open(os.path.join(output_dir, "analysis_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate markdown report
        self._generate_markdown_report(summary, output_dir)
        
        logger.info(f"Analysis complete. Results saved to {output_dir}")
        return summary
    
    def _analyze_html_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze HTML file for marketplace content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
                
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text(separator=" ", strip=True)
            
            # Analyze for listings
            listing_results = self.analyze_text(text_content, "phi-2", "listing_detection")
            
            # Analyze for transactions
            transaction_results = self.analyze_text(text_content, "tiny-llama", "transaction_detection")
            
            # Extract basic page info
            title = soup.title.string if soup.title else "No title"
            
            # Get all links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http') or '.onion' in href:
                    links.append({
                        "text": link.get_text(strip=True),
                        "url": href
                    })
            
            results = {
                "file": file_path,
                "title": title,
                "content_length": len(text_content),
                "links_count": len(links),
                "links_sample": links[:10],
                "listings": listing_results.get("result", {}).get("raw_response", ""),
                "transactions": transaction_results.get("result", {}).get("raw_response", "")
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing HTML file {file_path}: {str(e)}")
            return {"error": str(e), "file": file_path}
    
    def _analyze_text_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze text file for marketplace content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
                
            # Skip if file is too small
            if len(text_content) < 50:
                return {"file": file_path, "error": "Content too small for analysis"}
                
            # General analysis
            analysis_results = self.analyze_text(text_content, "tiny-llama", "text_analysis")
            
            # Look for pricing
            price_results = self.analyze_text(text_content, "phi-2", "price_extraction")
            
            results = {
                "file": file_path,
                "content_length": len(text_content),
                "general_analysis": analysis_results.get("result", {}).get("raw_response", ""),
                "price_information": price_results.get("result", {}).get("raw_response", "")
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing text file {file_path}: {str(e)}")
            return {"error": str(e), "file": file_path}
    
    def _generate_markdown_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        """Generate a markdown report from the analysis summary."""
        report = f"""# Dark Web Data Analysis Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Analysis Summary
- **Start Time**: {summary['analysis_start']}
- **End Time**: {summary['analysis_end']}
- **Duration**: {summary['duration_seconds'] / 60:.2f} minutes

## Files Processed
- HTML Files: {summary['files_processed']['html']}
- Text Files: {summary['files_processed']['text']}
- Image Files: {summary['files_processed']['image']}
- Other Files: {summary['files_processed']['other']}
- **Total**: {sum(summary['files_processed'].values())}

## Content Overview
- **Detected Listings**: {summary['listings_found']}
- **Detected Transactions**: {summary['transactions_detected']}
- **Categories Detected in Images**: {', '.join(summary['image_categories_detected'])}

## Sample Findings
"""
        
        if summary['listings_sample']:
            report += """
### Sample Listings
"""
            for listing in summary['listings_sample']:
                report += f"- {listing}\n"
                
        if summary['transactions_sample']:
            report += """
### Sample Transactions
"""
            for transaction in summary['transactions_sample']:
                report += f"- {transaction}\n"
        
        report += """
## Conclusion
This report provides an overview of the analyzed dark web data. For detailed results, refer to the individual JSON files in the output directory.
"""
        
        # Save report
        with open(os.path.join(output_dir, "analysis_report.md"), 'w') as f:
            f.write(report)


# Jupyter Notebook Usage Example
def jupyter_notebook_demo():
    """
    Demo function for using the DarkWebLocalAnalyzer in a Jupyter notebook.
    This analyzes files in a local data directory.
    """
    print("Initializing Dark Web Data Analyzer...")
    analyzer = DarkWebLocalAnalyzer()
    
    # Test loading a model
    print("Loading TinyLlama model (this is open source and doesn't require login)...")
    result = analyzer.load_text_model("tiny-llama")
    print(f"Model loaded successfully: {result}")
    
    # Create a sample text file for testing if it doesn't exist
    sample_dir = "./sample_data"
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_text = """
    Marketplace Updates - 2023-03-15
    
    New Listings:
    - Premium Product A - 0.05 BTC - Seller: Vendor123
    - Digital Item B - 0.015 BTC - Seller: DigitalDealer
    - Package C - 0.12 BTC - Seller: TopVendor
    
    Recent Transactions:
    - Premium Product A - 0.05 BTC - Buyer: User456 - Status: Complete
    - Digital Item B - Payment Pending - Buyer: Anon789
    
    Feedback:
    - "Fast shipping, product as described" - User456 for Premium Product A
    - "Good communication" - User567 for Package X
    """
    
    sample_file = os.path.join(sample_dir, "sample_marketplace.txt")
    with open(sample_file, 'w') as f:
        f.write(sample_text)
    
    # Single file analysis test
    print("\nAnalyzing sample text file...")
    result = analyzer._analyze_text_file(sample_file)
    print(json.dumps(result, indent=2))
    
    # For analyzing your existing data directory:
    print("\nTo analyze your existing data directory, run the following code:")
    print("results = analyzer.analyze_directory('/path/to/your/data/directory')")
    print("This will process all text, HTML and image files in the directory and subdirectories")

    # Example of analyzing a directory (uncomment and modify path as needed)
    # data_directory = "./data"  # Change to your data directory path
    # print(f"\nAnalyzing data directory: {data_directory}")
    # results = analyzer.analyze_directory(data_directory)
    # print(f"Analysis complete. Summary: {len(results['files_processed'])} files processed")
    
    print("\nFor analyzing a specific directory, run:")
    print("analyzer.analyze_directory('/home/nexulean/Projects/NarcoTakedownBackup/data')")