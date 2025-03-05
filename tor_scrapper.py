import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from tqdm import tqdm
from urllib.parse import urljoin, urlparse

# Tor proxy settings
proxies = {
    "http": "socks5h://127.0.0.1:9150",
    "https": "socks5h://127.0.0.1:9150",
}

# Create main data directory
base_dir = "data"
os.makedirs(base_dir, exist_ok=True)

# Ask user for multiple URLs
urls = input("Enter .onion URLs separated by commas: ").split(",")

# Keywords to detect payment information
payment_keywords = ["bitcoin", "btc", "monero", "xmr", "ethereum", "eth", "wallet", "crypto", "address", "paypal"]

def save_data(data, save_path, file_name):
    """Save data in multiple formats (CSV, JSON, TXT)"""
    if not data:
        return
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_path, f"{file_name}.csv"), index=False, encoding="utf-8")
    
    with open(os.path.join(save_path, f"{file_name}.json"), "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    
    with open(os.path.join(save_path, f"{file_name}.txt"), "w", encoding="utf-8") as txt_file:
        for item in data:
            txt_file.write(str(item) + "\n")

for url in urls:
    url = url.strip()
    if not url:
        continue

    print(f"\n🔍 Scraping: {url}")

    try:
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        domain = urlparse(url).netloc.replace(".", "_")
        save_path = os.path.join(base_dir, domain)
        os.makedirs(save_path, exist_ok=True)

        # Extract Text
        text_data = [{"Tag": tag.name, "Content": tag.text.strip()} for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])]
        save_data(text_data, save_path, "text")

        # Extract Links
        links_data = [{"Link Text": a.text.strip(), "URL": urljoin(url, a["href"])} for a in soup.find_all("a", href=True)]
        save_data(links_data, save_path, "links")

        # Extract and Download Images
        images_data = []
        images_path = os.path.join(save_path, "images")
        os.makedirs(images_path, exist_ok=True)

        for img in tqdm(soup.find_all("img", src=True), desc="Downloading Images"):
            img_url = urljoin(url, img["src"])
            img_name = os.path.join(images_path, os.path.basename(img_url))

            try:
                img_data = requests.get(img_url, proxies=proxies, timeout=10).content
                with open(img_name, "wb") as img_file:
                    img_file.write(img_data)
                images_data.append({"Image URL": img_url, "Alt Text": img.get("alt", "N/A"), "Saved Path": img_name})
            except:
                images_data.append({"Image URL": img_url, "Alt Text": img.get("alt", "N/A"), "Saved Path": "Download Failed"})
        save_data(images_data, save_path, "images")

        # Extract and Download Videos
        videos_data = []
        videos_path = os.path.join(save_path, "videos")
        os.makedirs(videos_path, exist_ok=True)

        for video in tqdm(soup.find_all("video", src=True), desc="Downloading Videos"):
            video_url = urljoin(url, video["src"])
            video_name = os.path.join(videos_path, os.path.basename(video_url))

            try:
                video_data = requests.get(video_url, proxies=proxies, timeout=20).content
                with open(video_name, "wb") as vid_file:
                    vid_file.write(video_data)
                videos_data.append({"Video URL": video_url, "Saved Path": video_name})
            except:
                videos_data.append({"Video URL": video_url, "Saved Path": "Download Failed"})
        save_data(videos_data, save_path, "videos")

        # Extract Tables
        tables_data = []
        for idx, table in enumerate(soup.find_all("table")):
            rows = []
            headers = [th.text.strip() for th in table.find_all("th")]

            for row in table.find_all("tr")[1:]:  # Skip header row
                cells = [td.text.strip() for td in row.find_all("td")]
                rows.append(cells)

            df_table = pd.DataFrame(rows, columns=headers) if headers else pd.DataFrame(rows)
            table_file = os.path.join(save_path, f"table_{idx+1}.csv")
            df_table.to_csv(table_file, index=False, encoding="utf-8")
            tables_data.append(table_file)
        save_data([{ "Table File": file} for file in tables_data], save_path, "tables")

        # Detect Payment Information
        payment_data = []
        for tag in soup.find_all(["p", "span", "div"]):
            content = tag.text.strip().lower()
            if any(keyword in content for keyword in payment_keywords):
                payment_data.append({"Content": tag.text.strip()})
        save_data(payment_data, save_path, "payments")

        print(f"✅ Scraping complete for {url}!")
        print(f"📂 Data saved in: {save_path}/")

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
