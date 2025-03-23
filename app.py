import os
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from flask import Flask, request, render_template, jsonify
from PIL import Image, UnidentifiedImageError
import io
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Load the pre-trained classifier model
classifier = None
tokenizer = None

def load_model():
    """Load the pre-trained text classification model and tokenizer."""
    global classifier, tokenizer
    try:
        with open("narcotic_classifier.pkl", "rb") as f:
            models = pickle.load(f)
            classifier = models['text_model']
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the model when the app starts
load_model()

class ImageAnalyzer:
    """Class to handle image analysis for narcotic content detection."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)
        except Exception as e:
            print(f"Error loading image model: {e}")
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
            suspicious_class_indices = [67, 401, 463]  # Placeholder indices
            suspicious_probs = probabilities[0, suspicious_class_indices].sum().item()

            return {
                'suspicious': suspicious_probs > 0.3,
                'confidence': suspicious_probs
            }
        except UnidentifiedImageError:
            return {'suspicious': False, 'confidence': 0.0, 'error': 'Invalid image format'}
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {'suspicious': False, 'confidence': 0.0, 'error': str(e)}

image_analyzer = ImageAnalyzer()

def classify_website(url):
    """Classify a website as narcotic or not."""
    try:
        is_onion = '.onion' in url

        if is_onion:
            # Use Tor proxy for .onion URLs
            proxies = {
                'http': 'socks5h://127.0.0.1:9050',
                'https': 'socks5h://127.0.0.1:9050'
            }
            response = requests.get(url, proxies=proxies, timeout=10)
        else:
            response = requests.get(url, timeout=10)

        content = response.text
        images = extract_images(url, content)

        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text()

        text_result = analyze_text(text_content)
        image_results = analyze_images(images)
        combined_result = combine_analyses(text_result, image_results, url, is_onion)

        # Ensure all values in the result are JSON-serializable
        combined_result["is_narcotic"] = str(combined_result["is_narcotic"])  # Convert bool to str
        return combined_result
    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "is_narcotic": "False",  # Convert bool to str
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        return {
            "url": url,
            "is_narcotic": "False",  # Convert bool to str
            "error": str(e)
        }

def extract_images(base_url, html_content):
    """Extract image URLs from HTML content."""
    images = []
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')

    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            if not img_url.startswith(('http://', 'https://')):
                img_url = urljoin(base_url, img_url)
            images.append(img_url)

    return images

def analyze_text(text_content):
    """Analyze text content using the text model."""
    inputs = tokenizer(text_content, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        outputs = text_model(**inputs)

    features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().reshape(1, -1)
    prediction = classifier.predict(features)[0]
    probability = classifier.predict_proba(features)[0][1]

    keyword_match = check_narcotic_keywords(text_content)
    suspicious_patterns = check_suspicious_patterns(text_content)

    return {
        "is_narcotic": prediction,
        "confidence": probability,
        "keyword_match": keyword_match,
        "suspicious_patterns": suspicious_patterns
    }

def analyze_images(image_urls):
    """Analyze images from the website."""
    results = []

    for img_url in image_urls:
        try:
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                img_data = response.content
                analysis = image_analyzer.predict(img_data)
                results.append({
                    "url": img_url,
                    "suspicious": analysis.get("suspicious", False),
                    "confidence": analysis.get("confidence", 0.0)
                })
        except Exception as e:
            print(f"Error analyzing image {img_url}: {e}")

    return results

def combine_analyses(text_result, image_results, url, is_onion):
    """Combine text and image analyses for final decision."""
    total_images = len(image_results)
    suspicious_images = sum(1 for img in image_results if img.get("suspicious", False))
    suspicious_image_ratio = suspicious_images / max(1, total_images)

    text_is_narcotic = text_result.get("is_narcotic", False)
    text_confidence = text_result.get("confidence", 0.0)
    keyword_match = text_result.get("keyword_match", False)
    suspicious_patterns = text_result.get("suspicious_patterns", False)

    overall_confidence = (
        0.6 * text_confidence +
        0.3 * suspicious_image_ratio +
        0.1 * (1.0 if is_onion else 0.0)
    )

    is_narcotic = (
        text_is_narcotic or
        suspicious_image_ratio > 0.3 or
        (is_onion and (keyword_match or suspicious_patterns))
    )

    return {
        "url": url,
        "is_narcotic": str(is_narcotic),  # Convert bool to str
        "confidence": float(overall_confidence),  # Ensure float
        "additional_signals": {
            "is_onion": str(is_onion),  # Convert bool to str
            "keyword_match": str(keyword_match),  # Convert bool to str
            "suspicious_patterns": str(suspicious_patterns),  # Convert bool to str
            "total_images": int(total_images),  # Ensure int
            "suspicious_images": int(suspicious_images),  # Ensure int
            "suspicious_image_ratio": float(suspicious_image_ratio),  # Ensure float
            "text_confidence": float(text_confidence)  # Ensure float
        },
        "image_analysis": image_results if total_images > 0 else "No images found"
    }

def check_narcotic_keywords(text):
    """Check for narcotic-related keywords."""
    keywords = ['narcotic', 'drug', 'cocaine', 'heroin', 'mdma', 'lsd', 'marijuana', 'cannabis', 'buy drugs', 'pills', 'opioid']
    return any(keyword in text.lower() for keyword in keywords)

def check_suspicious_patterns(content):
    """Check for suspicious patterns in content."""
    patterns = [
        r'bitcoin|btc|monero|xmr|cryptocurrency',
        r'escrow|vendor|marketplace',
        r'anonymous|encrypted|secure',
        r'shipping|delivery|tracking',
        r'telegram|wickr|signal'
    ]
    return any(re.search(pattern, content.lower()) for pattern in patterns)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400

    result = classify_website(url)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)