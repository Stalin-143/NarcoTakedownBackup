import os
import json
import re
import requests
import base64
import socket
import logging
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from flask import Flask, request, render_template, jsonify
from PIL import Image, UnidentifiedImageError
import io
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from sklearn.ensemble import RandomForestClassifier
import pickle
import socks  # Make sure you have PySocks installed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("darkweb_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("darkweb_scanner")

app = Flask(__name__)

# Load the pre-trained classifier model
classifier = None
tokenizer = None
text_model = None

def load_model():
    """Load the pre-trained text classification model and tokenizer."""
    global classifier, tokenizer, text_model
    try:
        with open("narcotic_classifier.pkl", "rb") as f:
            models = pickle.load(f)
            classifier = models['text_model']
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Check Tor connectivity
def check_tor_connection():
    """Check if Tor is running and properly configured."""
    try:
        # Try both common Tor SOCKS ports
        ports_to_try = [9150, 9050]
        
        for port in ports_to_try:
            sock = socks.socksocket()
            sock.set_proxy(socks.SOCKS5, "127.0.0.1", port)
            sock.settimeout(5)
            
            try:
                # Try to connect to a known .onion site to verify Tor connectivity
                sock.connect(("duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion", 80))
                sock.close()
                logger.info(f"Tor connection successful on port {port}")
                return port  # Return the working port
            except Exception as e:
                logger.warning(f"Failed to connect to Tor on port {port}: {e}")
                continue
        
        logger.error("Tor connection failed on all ports. Please ensure Tor is running.")
        return None
    except Exception as e:
        logger.error(f"Error checking Tor connection: {e}")
        return None

# Load the model when the app starts
load_model()

class ImageAnalyzer:
    """Class to handle image analysis for narcotic content detection."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)
            logger.info(f"Image analyzer initialized on {self.device}")
        except Exception as e:
            logger.error(f"Error loading image model: {e}")
            self.model = None
            self.processor = None

    def predict(self, image_data):
        """Analyze an image for suspicious content."""
        try:
            if self.model is None or self.processor is None:
                return {'suspicious': False, 'confidence': 0.1}

            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str) and os.path.exists(image_data):
                image = Image.open(image_data)
            else:
                image = image_data

            width, height = image.size
            if width < 50 or height < 50:
                return {'suspicious': False, 'confidence': 0.05}

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # These indices should be properly determined based on your model's classes
            # Default ViT model doesn't have specific narcotic classes, so this is likely causing false positives
            suspicious_class_indices = [67, 401, 463]  # These should be validated
            suspicious_probs = probabilities[0, suspicious_class_indices].sum().item()

            # More conservative threshold to reduce false positives
            return {
                'suspicious': suspicious_probs > 0.5,  # Increased threshold from 0.3 to 0.5
                'confidence': suspicious_probs
            }
        except UnidentifiedImageError:
            return {'suspicious': False, 'confidence': 0.0, 'error': 'Invalid image format'}
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {'suspicious': False, 'confidence': 0.0, 'error': str(e)}

image_analyzer = ImageAnalyzer()

def validate_onion_url(url):
    """Validate if the URL is a proper .onion URL."""
    try:
        parsed = urlparse(url)
        # Check if it's a valid .onion domain (v3 onion addresses are 56 chars + .onion)
        if parsed.netloc.endswith('.onion'):
            # V3 addresses are 56 characters before .onion
            if len(parsed.netloc) == 62 and parsed.netloc[-6:] == '.onion':
                return True
            # V2 addresses (legacy) are 16 characters before .onion
            elif len(parsed.netloc) == 22 and parsed.netloc[-6:] == '.onion':
                return True
            else:
                logger.warning(f"Invalid .onion URL format: {url}")
                return False
        return False
    except Exception as e:
        logger.error(f"Error validating onion URL: {e}")
        return False

def get_tor_session(tor_port):
    """Create a requests session that routes through Tor."""
    session = requests.session()
    # Specify a default timeout for all requests
    session.timeout = 30
    
    # Configure the session to use Tor
    session.proxies = {
        'http': f'socks5h://127.0.0.1:{tor_port}',
        'https': f'socks5h://127.0.0.1:{tor_port}'
    }
    
    return session

def classify_website(url):
    """Classify a website as narcotic or not."""
    try:
        is_onion = '.onion' in url.lower()
        
        if is_onion:
            # Validate the .onion URL
            if not validate_onion_url(url):
                return {
                    "url": url,
                    "is_narcotic": "False",
                    "error": "Invalid .onion URL format"
                }
            
            # Check Tor connectivity
            tor_port = check_tor_connection()
            if tor_port is None:
                return {
                    "url": url,
                    "is_narcotic": "False",
                    "error": "Tor is not running or properly configured. Please start Tor and try again."
                }
            
            # Use Tor session for .onion URLs
            logger.info(f"Accessing .onion site: {url} via Tor on port {tor_port}")
            session = get_tor_session(tor_port)
            
            # Add User-Agent to avoid some Tor blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0'
            }
            
            # Try multiple times with increasing timeouts
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    timeout = 10 * (attempt + 1)  # Increase timeout with each attempt
                    logger.info(f"Attempt {attempt+1}/{max_retries} with timeout {timeout}s")
                    response = session.get(url, headers=headers, timeout=timeout)
                    break
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout on attempt {attempt+1}, retrying...")
                        time.sleep(2)  # Wait before retry
                    else:
                        raise
            
            logger.info(f"Successfully connected to {url}")
        else:
            # Regular HTTP request for clearnet sites
            response = requests.get(url, timeout=20)
        
        content = response.text
        images = extract_images(url, content)

        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text()

        text_result = analyze_text(text_content)
        
        # For .onion sites, be more careful with image analysis
        if is_onion:
            # Only analyze a limited number of images to avoid overloading Tor
            max_images = min(5, len(images))
            if max_images < len(images):
                logger.info(f"Limiting image analysis to {max_images} out of {len(images)} images for Tor performance")
            image_results = analyze_images(images[:max_images], is_onion=True, tor_port=tor_port)
        else:
            image_results = analyze_images(images)
            
        combined_result = combine_analyses(text_result, image_results, url, is_onion)

        # Ensure all values in the result are JSON-serializable
        combined_result["is_narcotic"] = str(combined_result["is_narcotic"])
        return combined_result
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error accessing {url}: {str(e)}"
        logger.error(error_msg)
        return {
            "url": url,
            "is_narcotic": "False",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Error analyzing {url}: {str(e)}"
        logger.error(error_msg)
        return {
            "url": url,
            "is_narcotic": "False",
            "error": error_msg
        }

def extract_images(base_url, html_content):
    """Extract image URLs from HTML content."""
    images = []
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        img_tags = soup.find_all('img')

        for img in img_tags:
            img_url = img.get('src')
            if img_url:
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(base_url, img_url)
                images.append(img_url)
        
        logger.info(f"Extracted {len(images)} images from {base_url}")
        return images
    except Exception as e:
        logger.error(f"Error extracting images from {base_url}: {e}")
        return images

def analyze_text(text_content):
    """Analyze text content using the text model."""
    try:
        global text_model
        if text_model is None:  # Ensure model is loaded
            text_model = AutoModel.from_pretrained("distilbert-base-uncased")
            
        inputs = tokenizer(text_content, truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = text_model(**inputs)

        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().reshape(1, -1)
        prediction = classifier.predict(features)[0]
        probability = classifier.predict_proba(features)[0][1]

        # Check for narcotic-related content with improved logic
        keyword_match = check_narcotic_keywords(text_content)
        suspicious_patterns = check_suspicious_patterns(text_content)

        # Only consider keyword match significant if there are multiple matches
        keyword_significance = keyword_count(text_content)
        significant_keyword_match = keyword_significance > 2  # Require multiple keyword matches

        logger.info(f"Text analysis results - prediction: {prediction}, confidence: {probability:.4f}, keyword_match: {keyword_match}, keyword_significance: {keyword_significance}, suspicious_patterns: {suspicious_patterns}")
        
        return {
            "is_narcotic": prediction and (significant_keyword_match or suspicious_patterns),  # More conservative
            "confidence": probability,
            "keyword_match": keyword_match,
            "keyword_significance": keyword_significance,
            "suspicious_patterns": suspicious_patterns
        }
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return {
            "is_narcotic": False,
            "confidence": 0.0,
            "keyword_match": False,
            "keyword_significance": 0,
            "suspicious_patterns": False,
            "error": str(e)
        }

def analyze_images(image_urls, is_onion=False, tor_port=None):
    """Analyze images from the website and return analysis with image data."""
    results = []

    session = requests.session()
    if is_onion and tor_port:
        session.proxies = {
            'http': f'socks5h://127.0.0.1:{tor_port}',
            'https': f'socks5h://127.0.0.1:{tor_port}'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0'
        }
        session.headers.update(headers)

    for img_url in image_urls:
        try:
            logger.info(f"Analyzing image: {img_url}")
            
            timeout = 20 if not is_onion else 30
            response = session.get(img_url, timeout=timeout)
            
            if response.status_code == 200:
                img_data = response.content
                
                # Only analyze images over a certain size to avoid analyzing tiny icons
                if len(img_data) < 1000:  # Less than 1KB
                    logger.info(f"Skipping very small image ({len(img_data)} bytes): {img_url}")
                    continue
                
                # Convert image to base64 for displaying in HTML
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                img_type = response.headers.get('Content-Type', 'image/jpeg')
                img_data_uri = f"data:{img_type};base64,{img_base64}"
                
                # Analyze the image
                analysis = image_analyzer.predict(img_data)
                
                logger.info(f"Image analysis for {img_url}: suspicious={analysis.get('suspicious', False)}, confidence={analysis.get('confidence', 0.0):.4f}")
                
                results.append({
                    "url": img_url,
                    "suspicious": analysis.get("suspicious", False),
                    "confidence": analysis.get("confidence", 0.0),
                    "image_data": img_data_uri
                })
            else:
                logger.warning(f"Failed to retrieve image {img_url}: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error analyzing image {img_url}: {e}")
            results.append({
                "url": img_url,
                "suspicious": False,
                "confidence": 0.0,
                "error": str(e)
            })

    return results

def combine_analyses(text_result, image_results, url, is_onion):
    """Combine text and image analyses for final decision with more conservative thresholds."""
    total_images = len(image_results)
    suspicious_images = sum(1 for img in image_results if img.get("suspicious", False))
    suspicious_image_ratio = suspicious_images / max(1, total_images) if total_images > 0 else 0

    text_is_narcotic = text_result.get("is_narcotic", False)
    text_confidence = text_result.get("confidence", 0.0)
    keyword_match = text_result.get("keyword_match", False)
    keyword_significance = text_result.get("keyword_significance", 0)
    suspicious_patterns = text_result.get("suspicious_patterns", False)

    # Weight factors differently for .onion sites vs clearnet sites
    if is_onion:
        # For .onion sites, maintain higher sensitivity
        overall_confidence = (
            0.5 * text_confidence +
            0.2 * suspicious_image_ratio +
            0.3 * (min(1.0, keyword_significance / 5))  # Scale keyword significance
        )
        
        # For .onion sites, be more sensitive but still require stronger signals
        is_narcotic = (
            text_is_narcotic or
            (suspicious_image_ratio > 0.4 and keyword_match) or  # Require both image and keyword signals
            (keyword_significance > 3 and suspicious_patterns)  # Require strong keyword presence and patterns
        )
    else:
        # For clearnet sites, be much more conservative to reduce false positives
        overall_confidence = (
            0.7 * text_confidence +
            0.15 * suspicious_image_ratio +
            0.15 * (min(1.0, keyword_significance / 5))
        )
        
        # For clearnet sites, require stronger combined evidence
        is_narcotic = (
            (text_is_narcotic and keyword_significance > 2) or  # Require model prediction AND significant keywords
            (suspicious_image_ratio > 0.5 and keyword_significance > 3) or  # Strong image and keyword signals
            (keyword_significance > 4 and suspicious_patterns)  # Very strong keyword presence AND suspicious patterns
        )

    logger.info(f"Analysis results for {url}: is_narcotic={is_narcotic}, confidence={overall_confidence:.4f}")
    
    return {
        "url": url,
        "is_narcotic": is_narcotic,
        "confidence": float(overall_confidence),
        "additional_signals": {
            "is_onion": str(is_onion),
            "keyword_match": str(keyword_match),
            "keyword_significance": int(keyword_significance),
            "suspicious_patterns": str(suspicious_patterns),
            "total_images": int(total_images),
            "suspicious_images": int(suspicious_images),
            "suspicious_image_ratio": float(suspicious_image_ratio),
            "text_confidence": float(text_confidence)
        },
        "image_analysis": image_results if total_images > 0 else []
    }

def keyword_count(text):
    """Count the number of unique narcotic-related keywords found in the text."""
    keywords = [
        'narcotic', 'drug', 'cocaine', 'heroin', 'mdma', 'lsd', 'marijuana', 'cannabis', 
        'buy drugs', 'pills', 'opioid', 'psychedelic', 'meth', 'methamphetamine', 'amphetamine', 
        'fentanyl', 'ketamine', 'pharmacy', 'prescription', 'xanax', 'valium', 'percocet', 
        'oxycodone', 'tramadol', 'codeine', 'steroids', 'buprenorphine', 'suboxone',
        'dispensary', 'weed', 'pot', 'grass', 'hash', 'hashish', 'dope', 'ecstasy', 'molly'
    ]
    
    text_lower = text.lower()
    matched_keywords = [keyword for keyword in keywords if keyword in text_lower]
    
    # Filter out cases where words are part of legitimate contexts
    # Filter out common false positives based on context
    content_snippets = text_lower.split()
    filtered_count = 0
    
    # Check for legitimate pharmaceutical contexts
    if any(term in text_lower for term in ['online pharmacy', 'legitimate', 'licensed pharmacy', 'medical', 'healthcare']):
        # Reduce count for likely legitimate pharmacy content
        filtered_count = max(0, len(matched_keywords) - 1)
    else:
        filtered_count = len(matched_keywords)
    
    logger.info(f"Narcotic keywords found: {matched_keywords}, filtered count: {filtered_count}")
    return filtered_count

def check_narcotic_keywords(text):
    """Check for narcotic-related keywords."""
    return keyword_count(text) > 0

def check_suspicious_patterns(content):
    """Check for suspicious patterns in content."""
    patterns = [
        r'bitcoin|btc|monero|xmr|cryptocurrency',
        r'escrow|vendor|marketplace',
        r'anonymous|encrypted|secure',
        r'shipping|delivery|tracking',
        r'hidden service|tor hidden|onion link',
        r'pgp|encryption key|verified vendor',
        r'darknet|darkweb|dark web|dark market',
        r'sample|tester|free sample'
    ]
    
    content_lower = content.lower()
    matched_patterns = [p for p in patterns if re.search(p, content_lower)]
    
    # For clearnet sites, require multiple pattern matches to reduce false positives
    # "encrypted", "secure", "shipping", etc. are common on legitimate sites
    is_suspicious = (
        len(matched_patterns) >= 3 or  # Multiple patterns required
        any(p in matched_patterns for p in [r'darknet|darkweb|dark web|dark market', r'hidden service|tor hidden|onion link'])  # Strong indicators
    )
    
    if matched_patterns:
        logger.info(f"Suspicious patterns found: {matched_patterns}, is suspicious: {is_suspicious}")
    
    return is_suspicious

@app.route("/")
def home():
    # Check Tor status for UI feedback
    tor_status = "Connected" if check_tor_connection() else "Not Connected"
    return render_template("indexx.html", tor_status=tor_status)

@app.route("/tor_status", methods=["GET"])
def tor_status():
    port = check_tor_connection()
    return jsonify({
        "tor_running": port is not None,
        "port": port
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400

    # Add http:// if no protocol specified
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    logger.info(f"Analyzing URL: {url}")
    result = classify_website(url)
    return jsonify(result)

if __name__ == "__main__":
    logger.info("Starting Dark Web Scanner application")
    app.run(debug=True)