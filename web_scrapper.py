import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from urllib.parse import urljoin, urlparse

# Create "data" directory
DATA_DIR = os.path.abspath("data")
os.makedirs(DATA_DIR, exist_ok=True)

# Get user input
urls = input("Enter website URLs separated by commas: ").split(",")

# Payment-related keywords
payment_keywords = ["bitcoin", "btc", "monero", "xmr", "ethereum", "eth", "wallet", "crypto", "address", "paypal"]

# Set up requests session
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

for url in urls:
    url = url.strip()
    if not url:
        continue

    print(f"\n🔍 Scraping: {url}")

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        domain = urlparse(url).netloc.replace(".", "_")

        # Create site-specific directory
        site_dir = os.path.join(DATA_DIR, domain)
        os.makedirs(site_dir, exist_ok=True)

        def save_data(data, filename):
            """Saves data as CSV & JSON inside site_dir."""
            if not data:
                print(f"⚠ No data to save for {filename}. Skipping...")
                return

            csv_path = os.path.join(site_dir, f"{filename}.csv")
            json_path = os.path.join(site_dir, f"{filename}.json")

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False, encoding="utf-8")

            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)

            print(f"✅ Saved {filename} in {site_dir}")

        ### Extract Text ###
        text_data = [{"Tag": tag.name, "Content": tag.get_text(strip=True)} for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])]
        save_data(text_data, "text")

        ### Extract Links ###
        links_data = [{"Link Text": a.get_text(strip=True), "URL": urljoin(url, a["href"])} for a in soup.find_all("a", href=True)]
        save_data(links_data, "links")

        ### Extract & Download Images ###
        images_data = []
        img_folder = os.path.join(site_dir, "images")
        os.makedirs(img_folder, exist_ok=True)

        for img in tqdm(soup.find_all("img", src=True), desc="Downloading Images"):
            img_url = urljoin(url, img["src"])
            img_name = os.path.join(img_folder, os.path.basename(urlparse(img_url).path) or "image.jpg")

            try:
                img_data = session.get(img_url, timeout=10).content
                with open(img_name, "wb") as img_file:
                    img_file.write(img_data)
                images_data.append({"Image URL": img_url, "Alt Text": img.get("alt", "N/A"), "Saved Path": img_name})
            except Exception:
                images_data.append({"Image URL": img_url, "Alt Text": img.get("alt", "N/A"), "Saved Path": "Download Failed"})

        save_data(images_data, "images")

        ### Extract & Download Videos ###
        videos_data = []
        vid_folder = os.path.join(site_dir, "videos")
        os.makedirs(vid_folder, exist_ok=True)

        for video in tqdm(soup.find_all("video", src=True), desc="Downloading Videos"):
            video_url = urljoin(url, video["src"])
            video_name = os.path.join(vid_folder, os.path.basename(urlparse(video_url).path) or "video.mp4")

            try:
                video_data = session.get(video_url, timeout=20).content
                with open(video_name, "wb") as vid_file:
                    vid_file.write(video_data)
                videos_data.append({"Video URL": video_url, "Saved Path": video_name})
            except Exception:
                videos_data.append({"Video URL": video_url, "Saved Path": "Download Failed"})

        save_data(videos_data, "videos")

        ### Extract Tables ###
        for idx, table in enumerate(soup.find_all("table")):
            rows = []
            headers = [th.get_text(strip=True) for th in table.find_all("th")]

            for row in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if cells:
                    rows.append(cells)

            if rows:
                df_table = pd.DataFrame(rows, columns=headers if headers else None)
                table_file = os.path.join(site_dir, f"table_{idx+1}.csv")
                df_table.to_csv(table_file, index=False, encoding="utf-8")
                print(f"✅ Saved table {idx+1} in {site_dir}")

        ### Detect Payment Information ###
        payment_data = []
        for tag in soup.find_all(["p", "span", "div"]):
            content = tag.get_text(strip=True).lower()
            if any(keyword in content for keyword in payment_keywords):
                payment_data.append({"Content": tag.get_text(strip=True)})

        save_data(payment_data, "payments")

        print(f"✅ Scraping complete for {url}!")
        print(f"📂 Data saved in: {site_dir}/")

    except requests.RequestException as e:
        print(f"❌ Error scraping {url}: {e}")
