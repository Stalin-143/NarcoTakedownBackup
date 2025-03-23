import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import json
import time
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.prompt import Prompt
from rich.text import Text
from io import BytesIO
from PIL import Image
import hashlib

console = Console()

# Data directory
DATA_DIR = "data/"

# TOR Proxy (for .onion sites)
TOR_PROXIES = {
    "http": "socks5h://127.0.0.1:9050",
    "https": "socks5h://127.0.0.1:9050"
}

# Known narcotic-related keywords
NARCOTIC_KEYWORDS = [
    "cocaine", "heroin", "marijuana", "cannabis", "weed", "mdma", "ecstasy", 
    "meth", "amphetamine", "lsd", "acid", "shrooms", "mushrooms", "ketamine",
    "opium", "fentanyl", "xanax", "valium", "adderall", "ritalin", "oxycodone", 
    "hydrocodone", "opioid", "benzo", "prescription", "pharmacy", "pills",
    "buy drugs", "buy pills", "bitcoin", "monero", "crypto", "escrow", 
    "darknet", "darkweb", "onion", "anonymous", "shipping", "stealth",
    "vendor", "marketplace"
]

def animated_message(message, delay=0.02):
    """Display an animated message in the console."""
    for char in message:
        console.print(char, end="", style="bold cyan", highlight=False)
        time.sleep(delay)
    print()

def has_tesseract():
    """Check if Tesseract OCR is available."""
    try:
        import cv2
        import pytesseract
        # Try to get tesseract version
        pytesseract.get_tesseract_version()
        return True
    except (ImportError, Exception):
        return False

def get_text_from_csv(file_path):
    """Extracts text content from a CSV file."""
    try:
        encodings = ["utf-8", "latin1", "ISO-8859-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                return " ".join(df.astype(str).values.flatten())
            except Exception:
                continue
        raise ValueError("Failed to read CSV with available encodings.")
    except Exception as e:
        console.print(f"⚠️ Error reading CSV {file_path}: {e}", style="bold red")
        return ""

def get_text_from_pdf(file_path):
    """Extracts text content from a PDF file."""
    try:
        import PyPDF2
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
            return text
    except ImportError:
        console.print("⚠️ PyPDF2 module not installed. Skipping PDF analysis.", style="bold yellow")
        return ""
    except Exception as e:
        console.print(f"⚠️ Error reading PDF {file_path}: {e}", style="bold red")
        return ""

def get_text_from_docx(file_path):
    """Extracts text content from a DOCX file."""
    try:
        import docx
        doc = docx.Document(file_path)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    except ImportError:
        console.print("⚠️ python-docx module not installed. Skipping DOCX analysis.", style="bold yellow")
        return ""
    except Exception as e:
        console.print(f"⚠️ Error reading DOCX {file_path}: {e}", style="bold red")
        return ""

def extract_text_from_image(file_path):
    """Extracts text from an image using OCR if available, otherwise collects image data."""
    if has_tesseract():
        try:
            import cv2
            import pytesseract
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Image could not be read")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text
        except Exception as e:
            console.print(f"⚠️ Error extracting text from image {file_path}: {e}", style="bold red")
            return ""
    else:
        # If OCR is not available, we'll use image hash and store the path
        try:
            # Calculate image hash
            img = Image.open(file_path)
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            
            # Return a special format that indicates this is an image hash, not text
            return f"[IMAGE_HASH:{img_hash}]"
        except Exception as e:
            console.print(f"⚠️ Error processing image {file_path}: {e}", style="bold red")
            return ""

def fetch_website_content(url):
    """Fetches text content and images from a website."""
    try:
        proxies = TOR_PROXIES if ".onion" in url else None
        with console.status("[bold yellow]Fetching website content...[/bold yellow]"):
            response = requests.get(url, proxies=proxies, timeout=50)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Extract image URLs
            image_urls = []
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if src and (src.startswith('http') or src.startswith('/')):
                    if src.startswith('/'):
                        base_url = '/'.join(url.split('/')[:3])
                        src = base_url + src
                    image_urls.append(src)
            
            # Extract all text from HTML, including alt attributes, title attributes, etc.
            all_text = []
            
            # Get text from title
            title = soup.find('title')
            if title and title.string:
                all_text.append(title.string)
                
            # Get text from meta descriptions
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                all_text.append(meta_desc.get('content'))
                
            # Get all visible text
            all_text.append(text_content)
            
            # Get alt text from images
            for img in soup.find_all('img'):
                if img.get('alt'):
                    all_text.append(img.get('alt'))
            
            # Get title attributes
            for elem in soup.find_all(attrs={'title': True}):
                all_text.append(elem['title'])
                
            combined_text = " ".join(all_text)
            
            return {
                "text": combined_text,
                "image_urls": image_urls[:5]  # Limit to first 5 images to avoid overloading
            }
    except requests.RequestException as e:
        console.print(f"❌ Error fetching {url}: {e}", style="bold red")
        return {"text": "", "image_urls": []}

def download_and_analyze_image(image_url, proxies=None):
    """Downloads an image from a URL and processes it."""
    try:
        response = requests.get(image_url, proxies=proxies, timeout=30)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        if has_tesseract():
            # If OCR is available, extract text
            img_np = np.array(img)
            import cv2
            import pytesseract
            
            # Convert to grayscale if the image is in color
            if len(img_np.shape) > 2 and img_np.shape[2] >= 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
                
            text = pytesseract.image_to_string(gray)
            return text
        else:
            # If OCR is not available, compute hash
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            return f"[IMAGE_HASH:{img_hash}]"
    except Exception as e:
        console.print(f"⚠️ Error analyzing image from {image_url}: {e}", style="bold red")
        return ""

def extract_all_data_files():
    """Extracts text from all files in the data directory."""
    all_data = []
    image_data = []
    
    if not os.path.exists(DATA_DIR):
        console.print(f"❌ Data directory '{DATA_DIR}' not found!", style="bold red")
        return all_data, image_data
    
    tesseract_available = has_tesseract()
    if not tesseract_available:
        console.print("[bold yellow]Tesseract OCR not detected. Image text extraction will be limited.[/bold yellow]")
        console.print("[bold yellow]Using image fingerprinting as an alternative method.[/bold yellow]")
    
    for root, _, files in os.walk(DATA_DIR):
        for file in track(files, description="[green]Loading reference data...[/green]"):
            file_path = os.path.join(root, file)
            file_text = ""
            
            if file.endswith(".csv"):
                file_text = get_text_from_csv(file_path)
                if file_text:
                    all_data.append({
                        "file": file_path,
                        "content": file_text
                    })
            elif file.endswith(".pdf"):
                file_text = get_text_from_pdf(file_path)
                if file_text:
                    all_data.append({
                        "file": file_path,
                        "content": file_text
                    })
            elif file.endswith(".docx"):
                file_text = get_text_from_docx(file_path)
                if file_text:
                    all_data.append({
                        "file": file_path,
                        "content": file_text
                    })
            elif file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
                file_text = extract_text_from_image(file_path)
                if file_text:
                    if file_text.startswith("[IMAGE_HASH:"):
                        # This is an image hash, store it separately
                        image_hash = file_text[12:-1]  # Extract hash from [IMAGE_HASH:abc123]
                        image_data.append({
                            "file": file_path,
                            "hash": image_hash
                        })
                    else:
                        # This is OCR text
                        all_data.append({
                            "file": file_path,
                            "content": file_text
                        })
    
    return all_data, image_data

def check_for_keywords(text):
    """Checks text for narcotic-related keywords."""
    matches = []
    text_lower = text.lower()
    
    for keyword in NARCOTIC_KEYWORDS:
        if keyword.lower() in text_lower:
            matches.append(keyword)
    
    return matches

def compare_image_hashes(hash1, hash2, threshold=5):
    """Compare two image hashes, return similarity (0-100)."""
    # This is a simple hash comparison - in a real system you'd use perceptual hashing
    # with hamming distance, but md5 hashes will be either identical or different
    return 100 if hash1 == hash2 else 0

def analyze_website(url):
    """Analyzes a website by comparing its content with reference data."""
    # Load all reference data first
    reference_data, reference_images = extract_all_data_files()
    if not reference_data and not reference_images:
        console.print("⚠️ No reference data found for comparison.", style="bold yellow")
        return
    
    # Fetch website content
    website_data = fetch_website_content(url)
    if not website_data["text"]:
        console.print("⚠️ Unable to fetch website content.", style="bold yellow")
        return
    
    animated_message("🔍 Analyzing website content against reference data...")
    
    # Check for narcotic keywords in website content
    keyword_matches = check_for_keywords(website_data["text"])
    
    # Check text content against reference data
    content_matches = []
    for ref in track(reference_data, description="[green]Comparing text content...[/green]"):
        similarity = SequenceMatcher(None, website_data["text"], ref["content"]).ratio() * 100
        if similarity > 30:  # Lower threshold to catch more potential matches
            content_matches.append({
                "file": os.path.basename(ref["file"]),
                "similarity": similarity
            })
    
    # Analyze images from the website
    image_matches = []
    if website_data["image_urls"] and reference_images:
        animated_message("🖼️ Analyzing images from the website...")
        proxies = TOR_PROXIES if ".onion" in url else None
        
        for img_url in track(website_data["image_urls"], description="[green]Processing images...[/green]"):
            img_result = download_and_analyze_image(img_url, proxies)
            
            if img_result.startswith("[IMAGE_HASH:"):
                # This is an image hash
                web_img_hash = img_result[12:-1]  # Extract hash
                
                # Compare with reference image hashes
                for ref_img in reference_images:
                    similarity = compare_image_hashes(web_img_hash, ref_img["hash"])
                    if similarity > 0:
                        image_matches.append({
                            "file": os.path.basename(ref_img["file"]),
                            "similarity": similarity,
                            "source": "image hash"
                        })
            elif img_result:
                # This is OCR text
                img_keyword_matches = check_for_keywords(img_result)
                if img_keyword_matches:
                    for keyword in img_keyword_matches:
                        image_matches.append({
                            "file": f"keyword:{keyword}",
                            "similarity": 90,  # High confidence for exact keyword match
                            "source": "image text"
                        })
                
                # Also compare with reference text data
                for ref in reference_data:
                    img_similarity = SequenceMatcher(None, img_result, ref["content"]).ratio() * 100
                    if img_similarity > 30:
                        image_matches.append({
                            "file": os.path.basename(ref["file"]),
                            "similarity": img_similarity,
                            "source": "image text"
                        })
    
    # Evaluate results
    all_matches = content_matches + image_matches
    
    console.print("\n[bold cyan]Analysis Results:[/bold cyan]")
    
    if keyword_matches:
        console.print(f"[bold yellow]Found {len(keyword_matches)} narcotic-related keywords:[/bold yellow]")
        console.print(", ".join(keyword_matches))
    
    if all_matches:
        console.print("\n[bold yellow]Potential content matches found:[/bold yellow]")
        for match in sorted(all_matches, key=lambda x: x["similarity"], reverse=True):
            source_type = match.get("source", "text content")
            console.print(f"⚠️ {match['similarity']:.2f}% match with {match['file']} ({source_type})")
        
        # Calculate overall threat score
        highest_match = max(match["similarity"] for match in all_matches) if all_matches else 0
        keyword_score = min(len(keyword_matches) * 15, 50)  # Cap at 50
        
        # Weighted scoring
        threat_score = (
            highest_match * 0.4 +  # 40% from highest content match
            keyword_score * 0.6    # 60% from keyword matches
        )
        
        console.print(f"\n[bold cyan]Overall threat score: {threat_score:.2f}/100[/bold cyan]")
        
        if threat_score > 60:
            console.print("🚨 [bold red]HIGH RISK: This website is highly likely to be a narcotics-related website![/bold red]")
        elif threat_score > 40:
            console.print("⚠️ [bold yellow]MEDIUM RISK: This website shows some characteristics of narcotics-related content.[/bold yellow]")
        else:
            console.print("ℹ️ [bold blue]LOW RISK: Some matches found, but unlikely to be a narcotics-related website.[/bold blue]")
    else:
        if keyword_matches:
            keyword_score = min(len(keyword_matches) * 15, 50)
            console.print(f"\n[bold cyan]Keyword-based threat score: {keyword_score:.2f}/100[/bold cyan]")
            
            if keyword_score > 30:
                console.print("⚠️ [bold yellow]MEDIUM RISK: This website contains multiple narcotic-related keywords.[/bold yellow]")
            else:
                console.print("ℹ️ [bold blue]LOW RISK: Few narcotic-related keywords found.[/bold blue]")
        else:
            console.print("✅ [bold green]No matches found. This website is likely NOT a narcotics-related website.[/bold green]")

def main():
    """Main function to handle user input and analyze data."""
    console.print("\n[bold magenta]===== Narcotic Website Detection Tool =====", style="bold magenta")
    console.print("[italic]This tool analyzes websites for narcotic-related content by comparing with reference data[/italic]\n")
    
    # First check if dependencies are available
    if not has_tesseract():
        console.print("[bold yellow]Notice: Tesseract OCR is not installed or not in PATH.[/bold yellow]")
        console.print("[bold yellow]Image text extraction will be limited to keyword matching and image fingerprinting.[/bold yellow]")
        console.print("[bold yellow]To install Tesseract OCR and enable full functionality:[/bold yellow]")
        console.print("  - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        console.print("  - macOS: brew install tesseract")
        console.print("  - Linux: sudo apt-get install tesseract-ocr")
    
    try:
        while True:
            console.print("\n[bold magenta]Choose an option:[/bold magenta]")
            console.print("[cyan]A) Check an OpenWeb link[/cyan]")
            console.print("[cyan]B) Check an Onion (Dark Web) link[/cyan]")
            console.print("[cyan]C) Scan local data directory[/cyan]")
            console.print("[cyan]D) Exit[/cyan]")
            
            choice = Prompt.ask("[bold yellow]Enter your choice (A/B/C/D)[/bold yellow]").strip().upper()
            
            if choice == "A" or choice == "B":
                url = Prompt.ask("[bold cyan]Enter the website URL[/bold cyan]")
                analyze_website(url)
            elif choice == "C":
                animated_message("🔍 Scanning local data directory...")
                data_files, image_files = extract_all_data_files()
                console.print(f"[bold green]Found {len(data_files)} text data files and {len(image_files)} image files for reference.[/bold green]")
                
                if data_files:
                    console.print("\n[bold cyan]Text data files:[/bold cyan]")
                    for idx, data in enumerate(data_files, 1):
                        console.print(f"{idx}. {os.path.basename(data['file'])}")
                
                if image_files:
                    console.print("\n[bold cyan]Image files:[/bold cyan]")
                    for idx, data in enumerate(image_files, 1):
                        console.print(f"{idx}. {os.path.basename(data['file'])}")
            elif choice == "D":
                animated_message("👋 Exiting program. Stay safe!")
                break
            else:
                console.print("[bold red]❌ Invalid choice. Please enter A, B, C, or D.[/bold red]")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Program interrupted by user. Exiting...[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")
        console.print("[bold yellow]Please check your configuration and try again.[/bold yellow]")

if __name__ == "__main__":
    main()