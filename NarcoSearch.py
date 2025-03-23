#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import pickle
from PIL import Image
import io
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class ImageAnalyzer:
    """Class to handle image analysis for narcotic content detection"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load a pre-trained image classification model
        try:
            print("Loading image classification model...")
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)
            print("Image model loaded")
        except Exception as e:
            print(f"Error loading image model: {e}")
            # Fallback to a basic model
            self.model = None
            self.processor = None

    def predict(self, image_data):
        """Analyze an image for suspicious content

        Args:
            image_data: PIL Image or image bytes

        Returns:
            dict: Dictionary with analysis results
        """
        try:
            if self.model is None or self.processor is None:
                return {'suspicious': False, 'confidence': 0.1}

            # Convert to PIL image if it's bytes
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str) and os.path.exists(image_data):
                image = Image.open(image_data)
            else:
                image = image_data

            # Check if image has suspicious dimensions or colors
            # This is a simple heuristic - you would replace with your own logic
            width, height = image.size
            if width < 50 or height < 50:
                return {'suspicious': False, 'confidence': 0.05}

            # Pre-process image for the model
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Process outputs
            # For demonstration, we're using a placeholder approach here
            # You would typically look at specific classes relevant to narcotic content
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            # For demonstration, we'll check if certain "suspicious" classes are high probability
            # You would adapt this logic to your specific model and task
            # This is just a placeholder - you'd use actual indices of relevant classes
            suspicious_class_indices = [67, 401, 463]  # Placeholder indices
            suspicious_probs = probabilities[0, suspicious_class_indices].sum().item()

            return {
                'suspicious': suspicious_probs > 0.3,  # Threshold
                'confidence': suspicious_probs
            }

        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {'suspicious': False, 'confidence': 0.0, 'error': str(e)}

class NarcoticWebsiteClassifier:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.text_model = None
        self.image_model = None
        self.combined_model = None
        self.tokenizer = None
        self.image_analyzer = ImageAnalyzer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self):
        """Load and preprocess all data from the data directory"""
        print("Loading data from:", self.data_dir)
        self.data = {
            "text": [],
            "images": [],
            "urls": [],
            "labels": []
        }

        # Recursively walk through all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # Process based on file type
                if file_ext in ['.txt', '.csv', '.json']:
                    self._process_text_file(file_path, file_ext)
                elif file_ext in ['.jpg', '.jpeg', '.png']:
                    self._process_image_file(file_path)

        print(f"Loaded {len(self.data['labels'])} samples")
        return self.data

    def _process_text_file(self, file_path, file_ext):
        """Process text files based on their extension"""
        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Assuming each text file has content and a label (narcotic or not)
                    # You'll need to adapt this based on your actual data structure
                    is_narcotic = self._check_narcotic_keywords(content)
                    self.data["text"].append(content)
                    self.data["labels"].append(is_narcotic)
                    self.data["urls"].append(self._extract_url(content))
                    self.data["images"].append(None)

            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                # Adjust column names based on your CSV structure
                if all(col in df.columns for col in ['content', 'url', 'is_narcotic']):
                    for _, row in df.iterrows():
                        self.data["text"].append(row['content'])
                        self.data["urls"].append(row['url'])
                        self.data["labels"].append(row['is_narcotic'])
                        self.data["images"].append(None)

            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    json_data = json.load(f)
                    # Process JSON based on your structure
                    if isinstance(json_data, list):
                        for item in json_data:
                            if all(key in item for key in ['content', 'url', 'is_narcotic']):
                                self.data["text"].append(item['content'])
                                self.data["urls"].append(item['url'])
                                self.data["labels"].append(item['is_narcotic'])
                                self.data["images"].append(None)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def _process_image_file(self, file_path):
        """Process image files"""
        try:
            # For images, we'll need labels from a separate source or from the file path
            # This is a placeholder - adapt to your data organization
            parent_dir = os.path.basename(os.path.dirname(file_path))
            is_narcotic = 'narcotic' in parent_dir.lower()

            self.data["images"].append(file_path)
            self.data["text"].append(None)
            self.data["urls"].append(None)
            self.data["labels"].append(is_narcotic)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")

    def _check_narcotic_keywords(self, text):
        """Simple keyword check - replace with your actual logic"""
        keywords = ['narcotic', 'drug', 'cocaine', 'heroin', 'mdma', 'lsd', 
                   'marijuana', 'cannabis', 'buy drugs', 'pills', 'opioid'
                    'heroin', 'fentanyl', 'carfentanil', 'morphine', 'codeine', 'oxycodone', 'hydrocodone', 'oxycontin',
                    'lisdexamfetamine', 'benzedrine', 'dexosyn', 'desoxyn', 'white', 'snow', 'blow', 'flake',
                    'yayo', 'yeyo', 'rock', 'hard', 'nose candy', 'tina', 'crank', 'go', 'go fast', 'uppers',
                    'stimulants', 'stims', 'tweak', 'tweaking', 'gak', 'yay', 'yola', 'cola', 'powder', 'base',
                    'freebase', 'addy', 'addies', 'study drug', 'study aid', 'bennies', 'pep pills', 'dexies',
                    'methylone', 'bath salts', 'cathinones', 'khat', 'qat', 'modafinil', 'provigil', 'armodafinil',
                    'nuvigil', 'ephedrine', 'pseudoephedrine', 'sudafed', 'shards', 'christina', 'shabu', 
                    'jane', 'green', 'trees', 'flower', 'nugs', 'buds', '420', 'four twenty', 'four-twenty']


        return any(keyword in text.lower() for keyword in keywords)

    def _extract_url(self, text):
        """Extract URL from text if present"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|(?:[-\w.]|(?:%[\da-fA-F]{2}))+\.onion'
        match = re.search(url_pattern, text)
        return match.group(0) if match else None

    def build_models(self):
        """Build and train the models"""
        print("Building models...")
        # 1. Text model using BERT
        self._build_text_model()

        # 2. Image model
        self._build_image_model()

        # 3. Combined model
        self._build_combined_model()

    def _build_text_model(self):
        """Build and train the text classification model"""
        print("Building text model...")
        # Filter data to include only text samples
        text_data = [(text, label) for text, label in zip(self.data["text"], self.data["labels"]) if text is not None]

        if not text_data:
            print("No text data available to train text model")
            return

        texts, labels = zip(*text_data)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        text_model = AutoModel.from_pretrained("distilbert-base-uncased")

        # Extract features
        features = []
        for text in texts:
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = text_model(**inputs)
            # Use CLS token as feature vector
            features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten())

        # Train a classifier
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        self.text_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.text_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.text_model.predict(X_test)
        print("Text model performance:")
        print(classification_report(y_test, y_pred))

    def _build_image_model(self):
        """Build and train the image classification model"""
        print("Building image model...")
        # Filter data to include only image samples
        image_data = [(img, label) for img, label in zip(self.data["images"], self.data["labels"]) if img is not None]

        if not image_data:
            print("No image data available to train image model")
            return

        # Here we simply use the ImageAnalyzer class
        print("Using ImageAnalyzer for image classification")
        self.image_model = self.image_analyzer

    def _build_combined_model(self):
        """Build a model that combines text and image features"""
        print("Building combined model...")
        # This would combine the outputs of the text and image models
        # For now, just use the text model
        self.combined_model = self.text_model

    def classify_website(self, url):
        """Classify a website as narcotic or not"""
        print(f"Analyzing website: {url}")

        if self.text_model is None:
            raise ValueError("Model has not been trained. Call build_models() first.")

        # Check if it's an onion URL
        is_onion = '.onion' in url

        try:
            # For .onion URLs, we would need a Tor setup
            if is_onion:
                print("Onion URL detected. Using pre-configured proxy for Tor access...")
                # This is where you'd implement Tor proxy access
                # For now, we'll use features that suggest it's likely narcotic
                content = "This is a placeholder for Tor hidden service content"
                images = []
            else:
                # For regular URLs, fetch the content
                print("Fetching website content...")
                response = requests.get(url, timeout=10)
                content = response.text

                # Extract images from the website
                print("Extracting images from website...")
                images = self._extract_images(url, content)
                print(f"Found {len(images)} images")

            # Extract text features
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text()

            # Text analysis
            print("Analyzing text content...")
            text_result = self._analyze_text(text_content)

            # Image analysis
            print("Analyzing images...")
            image_results = self._analyze_images(images)

            # Combined analysis
            combined_result = self._combine_analyses(text_result, image_results, url, is_onion)

            return combined_result

        except Exception as e:
            print(f"Error analyzing {url}: {e}")
            return {
                "url": url,
                "is_narcotic": None,
                "error": str(e)
            }

    def _extract_images(self, base_url, html_content):
        """Extract image URLs from HTML content"""
        images = []
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all img tags
        img_tags = soup.find_all('img')

        for img in img_tags:
            # Get the image URL
            img_url = img.get('src')
            if img_url:
                # Make URL absolute if it's relative
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(base_url, img_url)

                # Add to list
                images.append(img_url)

        return images

    def _analyze_text(self, text_content):
        """Analyze text content using the text model"""
        # Tokenize and get features
        inputs = self.tokenizer(text_content, truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            text_model = AutoModel.from_pretrained("distilbert-base-uncased")
            outputs = text_model(**inputs)

        # Use CLS token as feature vector
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().reshape(1, -1)

        # Make prediction
        prediction = self.text_model.predict(features)[0]
        probability = self.text_model.predict_proba(features)[0][1]  # Probability of being narcotic

        # Additional signals
        keyword_match = self._check_narcotic_keywords(text_content)
        suspicious_patterns = self._check_suspicious_patterns(text_content)

        return {
            "is_narcotic": prediction,
            "confidence": probability,
            "keyword_match": keyword_match,
            "suspicious_patterns": suspicious_patterns
        }

    def _analyze_images(self, image_urls):
        """Analyze images from the website"""
        results = []

        for img_url in image_urls:
            try:
                # Fetch the image
                response = requests.get(img_url, timeout=5)
                if response.status_code == 200:
                    img_data = response.content

                    # Analyze the image
                    analysis = self.image_analyzer.predict(img_data)

                    # Add to results
                    results.append({
                        "url": img_url,
                        "suspicious": analysis.get("suspicious", False),
                        "confidence": analysis.get("confidence", 0.0)
                    })
            except Exception as e:
                print(f"Error analyzing image {img_url}: {e}")

        return results

    def _combine_analyses(self, text_result, image_results, url, is_onion):
        """Combine text and image analyses for final decision"""
        # Calculate the percentage of suspicious images
        total_images = len(image_results)
        suspicious_images = sum(1 for img in image_results if img.get("suspicious", False))
        suspicious_image_ratio = suspicious_images / max(1, total_images)

        # Get text analysis results
        text_is_narcotic = text_result.get("is_narcotic", False)
        text_confidence = text_result.get("confidence", 0.0)
        keyword_match = text_result.get("keyword_match", False)
        suspicious_patterns = text_result.get("suspicious_patterns", False)

        # Determine overall probability
        # This is a simple weighted approach - you can make this more sophisticated
        overall_confidence = (
            0.6 * text_confidence + 
            0.3 * suspicious_image_ratio + 
            0.1 * (1.0 if is_onion else 0.0)
        )

        # Make final decision
        # Considered narcotic if any of these are true
        is_narcotic = (
            text_is_narcotic or
            suspicious_image_ratio > 0.3 or
            (is_onion and (keyword_match or suspicious_patterns))
        )

        return {
            "url": url,
            "is_narcotic": is_narcotic,
            "confidence": overall_confidence,
            "additional_signals": {
                "is_onion": is_onion,
                "keyword_match": keyword_match,
                "suspicious_patterns": suspicious_patterns,
                "total_images": total_images,
                "suspicious_images": suspicious_images,
                "suspicious_image_ratio": suspicious_image_ratio,
                "text_confidence": text_confidence
            },
            "image_analysis": image_results if total_images > 0 else "No images found"
        }

    def _check_suspicious_patterns(self, content, url=None):
        """Check for patterns common in narcotic websites"""
        patterns = [
            r'bitcoin|btc|monero|xmr|cryptocurrency',  # Payment methods
            r'escrow|vendor|marketplace',              # Marketplace terminology
            r'anonymous|encrypted|secure',             # Security terms
            r'shipping|delivery|tracking',             # Shipping terms
            r'telegram|wickr|signal' 
             # Payment methods
            r'bitcoin|btc|monero|xmr|cryptocurrency|crypto|eth|ethereum|ltc|litecoin|xrp|ripple|dash|zcash|zec',
            r'wallet|transaction|blockchain|mixer|tumbler|mixing|tumbling|atomic swap|p2p|peer[\s-]to[\s-]peer',
            r'coinbase|binance|kraken|kucoin|localbitcoins|paxful|bisq|exodus|metamask|cold storage',
            r'lightning network|segwit|unconfirmed|confirmations|block explorer|private key|public key',
            r'non[\s-]custodial|seed phrase|recovery phrase|backup|2fa|two[\s-]factor|authentication',
            r'escrow|multisig|multi[\s-]signature|finalize early|fe|auto[\s-]finalize|pgp|payment processor',
            r'cash[\s-]app|venmo|paypal|western union|money order|gift card|prepaid|voucher|vanilla|paysafecard',
            r'cashless|anonymous payment|secure payment|untraceable|no kyc|no verification',

            # Marketplace terminology
            r'escrow|vendor|marketplace|market|bazaar|shop|store|mall|emporium|exchange|darknet|darkweb',
            r'verified vendor|trusted seller|trusted vendor|vendor rating|feedback|review|rating system',
            r'verified buyer|market admin|dispute|resolution|support ticket|refund policy|reship policy',
            r'listing|product page|description|category|subcategory|inventory|stock|restock|out of stock',
            r'vendor shop|vendor page|vendor profile|pgp key|pgp verification|vendor level|trust level',
            r'featured listing|hot listing|new arrival|best seller|top rated|highly rated|verified purchase',
            r'market rules|terms of service|tos|faq|frequently asked|vendor bond|vendor fee|buyer protection',
            r'market mirror|alternative link|official link|phishing|scam|exit scam|selective scam|legit check',

            # Security terms
            r'anonymous|encrypted|secure|private|hidden|concealed|stealth|covert|discreet|clandestine|secret',
            r'tor|onion|tails|whonix|i2p|freenet|vpn|proxy|bridge|relay|node|encryption|decryption|cipher',
            r'pgp|gpg|public key|private key|aes|rsa|elliptic curve|end[\s-]to[\s-]end|e2e|opsec|persec',
            r'security|privacy|anonymity|pseudonym|alias|burner|throwaway|temporary|compartmentalization',
            r'key verification|signature verification|hash verification|checksum|sha256|md5|fingerprint',
            r'2fa|two[\s-]factor|authentication|captcha|verification|authorization|secure login|session',
            r'no[\s-]logs?|zero[\s-]logs?|no[\s-]records?|no[\s-]tracking|surveillance|monitoring|compromise',
            r'secure connection|encrypted connection|secure channel|secure communication|secure messaging',

            # Shipping terms
            r'shipping|delivery|tracking|package|parcel|mail|post|courier|dispatch|shipment|shipping method',
            r'express|priority|standard|economy|overnight|next[\s-]day|same[\s-]day|2[\s-]day|3[\s-]day',
            r'domestic|international|worldwide|global|regional|local|cross[\s-]border|customs|border control',
            r'stealth|decoy|vacuum sealed|moisture barrier|mylar|visual barrier|odor proof|smell proof',
            r'drop|drop off|drop location|safe location|safe address|pickup|collection point|p\.o\. box',
            r'shipping time|delivery time|transit time|estimated arrival|eta|shipping delay|processing time',
            r'carrier|usps|ups|fedex|dhl|royal mail|canada post|auspost|deutsche post|postal service',
            r'signature|signature required|no signature|shipping label|tracking number|customs declaration',

            # Communication apps
            r'telegram|wickr|signal|session|element|matrix|jabber|xmpp|riot|wire|threema|status|briar',
            r'encrypted chat|secure messaging|private messaging|self[\s-]destructing|disappearing messages',
            r'otr|off[\s-]the[\s-]record|e2ee|end[\s-]to[\s-]end encryption|forward secrecy|perfect forward',
            r'burner phone|burner account|throwaway account|temporary email|temp mail|protonmail|tutanota',
            r'secure email|encrypted email|private email|anonymous email|email service|email provider',
            r'messaging app|chat app|communication platform|secure platform|anonymous platform|private app',
            r'contact method|contact details|contact information|reach out|get in touch|direct message',
            r'username|handle|contact id|address|dm|pm|private message|direct message|secure channel',

        ]

        return any(re.search(pattern, content.lower()) for pattern in patterns)

    def save_model(self, path="narcotic_classifier.pkl"):
        """Save the trained model"""
        if self.text_model is None:
            raise ValueError("No model to save. Train the model first.")

        with open(path, 'wb') as f:
            pickle.dump({
                'text_model': self.text_model,
                'combined_model': self.combined_model
            }, f)
        print(f"Model saved to {path}")

    def load_model(self, path="narcotic_classifier.pkl"):
        """Load a saved model"""
        with open(path, 'rb') as f:
            models = pickle.load(f)
            self.text_model = models['text_model']
            self.combined_model = models['combined_model']
        print(f"Model loaded from {path}")

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Initialize image analyzer
        self.image_analyzer = ImageAnalyzer()
        self.image_model = self.image_analyzer

        print("Model loaded successfully")

# Interactive URL testing function
def test_url(url):
    """Test a specific URL with the trained classifier"""
    try:
        # Load the trained classifier
        classifier = NarcoticWebsiteClassifier()
        classifier.load_model("narcotic_classifier.pkl")

        # Analyze the URL
        print(f"Analyzing: {url}")
        result = classifier.classify_website(url)

        # Display results
        print("\nResults:")
        print(f"  Is narcotic: {result['is_narcotic']}")
        if 'confidence' in result:
            print(f"  Confidence: {result['confidence']:.2f}")
        if 'additional_signals' in result:
            print(f"  Additional signals:")
            for key, value in result['additional_signals'].items():
                print(f"    - {key}: {value}")

        # Display image analysis if available
        if 'image_analysis' in result and result['image_analysis'] != "No images found":
            print("\n  Image Analysis:")
            for i, img_result in enumerate(result['image_analysis']):
                print(f"    Image {i+1}: {img_result['url']}")
                print(f"      - Suspicious: {img_result['suspicious']}")
                print(f"      - Confidence: {img_result['confidence']:.2f}")

        if 'error' in result:
            print(f"  Error: {result['error']}")

        return result
    except Exception as e:
        print(f"Error testing URL: {e}")
        return None

# Create a widget-based interface
def create_interactive_interface():
    """Create an interactive widget-based interface"""
    try:
        import ipywidgets as widgets
        from IPython.display import display

        # Create input widget
        url_input = widgets.Text(
            value='https://example.com',
            placeholder='Enter URL to check',
            description='URL:',
            disabled=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )

        # Create button
        check_button = widgets.Button(
            description='Check Website',
            disabled=False,
            button_style='primary',
            tooltip='Click to analyze the URL'
        )

        # Create output area
        output = widgets.Output()

        # Button click handler
        def on_button_clicked(b):
            with output:
                output.clear_output()
                url = url_input.value
                if url:
                    test_url(url)
                else:
                    print("Please enter a URL")

        # Connect button to function
        check_button.on_click(on_button_clicked)

        # Display the UI
        display(widgets.VBox([widgets.Label("Narcotic Website Analyzer"), url_input, check_button, output]))
    except ImportError:
        print("ipywidgets not available. Use test_url() function directly.")

# Example usage
def main():
    # Initialize the classifier
    classifier = NarcoticWebsiteClassifier(data_dir="data")

    # Load and preprocess the data
    classifier.load_data()

    # Build and train the models
    classifier.build_models()

    try:
        # Save the model
        classifier.save_model()



        for url in test_urls:
            result = classifier.classify_website(url)
            print(f"\nResults for {url}:")
            print(f"  Is narcotic: {result['is_narcotic']}")
            if 'confidence' in result:
                print(f"  Confidence: {result['confidence']:.2f}")
            if 'additional_signals' in result:
                print(f"  Additional signals: {result['additional_signals']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()

create_interactive_interface()


# In[ ]:




