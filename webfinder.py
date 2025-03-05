import os
import pandas as pd
import cv2
import pytesseract
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import json
import PyPDF2
import docx
import time
from rich.console import Console
from rich.progress import track
from rich.prompt import Prompt
from rich.text import Text

console = Console()

# Data directory
DATA_DIR = "data/"

# TOR Proxy (for .onion sites)b

TOR_PROXIES = {
    "http": "socks5h://127.0.0.1:9050",
    "https": "socks5h://127.0.0.1:9050"
}

def animated_message(message, delay=0.02):
    for char in message:
        console.print(char, end="", style="bold cyan", highlight=False)
        time.sleep(delay)
    print()

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

def fetch_website_content(url):
    """Fetches text from a website."""
    try:
        proxies = TOR_PROXIES if ".onion" in url else None
        with console.status("[bold yellow]Fetching website content...[/bold yellow]"):
            response = requests.get(url, proxies=proxies, timeout=50)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        console.print(f"❌ Error fetching {url}: {e}", style="bold red")
        return ""

def analyze_website(url):
    """Analyzes a website by comparing its content with individual files."""
    website_content = fetch_website_content(url)
    if not website_content:
        console.print("⚠️ Unable to fetch website content.", style="bold yellow")
        return

    if not os.path.exists(DATA_DIR):
        console.print(f"❌ Data directory '{DATA_DIR}' not found!", style="bold red")
        return
    
    animated_message("🔍 Scanning files in the data directory...")
    narcotic_detected = False
    for root, _, files in os.walk(DATA_DIR):
        for file in track(files, description="[green]Analyzing files...[/green]"):
            file_path = os.path.join(root, file)
            if file.endswith(".csv"):
                file_text = get_text_from_csv(file_path)
            else:
                continue
            
            if file_text:
                similarity = SequenceMatcher(None, website_content, file_text).ratio() * 100
                if similarity > 50:
                    console.print(f"⚠️ [bold yellow]Warning:[/bold yellow] {similarity:.2f}% match found with {file}")
                    narcotic_detected = True
    
    if narcotic_detected:
        console.print("🚨 [bold red]This website is classified as a narcotics-related website![/bold red]")
    else:
        console.print("✅ [bold green]This website is NOT a narcotics-related website.[/bold green]")

def main():
    """Main function to handle user input and analyze data."""
    while True:
        console.print("\n[bold magenta]Choose an option:[/bold magenta]")
        console.print("[cyan]A) Check an OpenWeb link[/cyan]")
        console.print("[cyan]B) Check an Onion (Dark Web) link[/cyan]")
        console.print("[cyan]C) Exit[/cyan]")
        
        choice = Prompt.ask("[bold yellow]Enter your choice (A/B/C)[/bold yellow]").strip().upper()
        
        if choice in ("A", "B"):
            url = Prompt.ask("[bold cyan]Enter the website URL[/bold cyan]")
            analyze_website(url)
        elif choice == "C":
            animated_message("👋 Exiting program. Stay safe!")
            break
        else:
            console.print("[bold red]❌ Invalid choice. Please enter A, B, or C.[/bold red]")

if __name__ == "__main__":
    main()
