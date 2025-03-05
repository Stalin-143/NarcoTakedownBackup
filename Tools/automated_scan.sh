#!/bin/bash

# Ask user for the website URL
read -p "Enter the target website URL (e.g., http://example.com): " url

# Extract domain from URL
domain=$(echo "$url" | awk -F[/:] '{print $4}')

# Output directory
OUTPUT_DIR="scan_results"
mkdir -p "$OUTPUT_DIR"

# Timestamped scan file
SCAN_FILE="$OUTPUT_DIR/${domain}_$(date +%Y%m%d_%H%M%S).txt"

echo "Scanning: $domain"
echo "Results will be saved to $SCAN_FILE"
echo "--------------------------------------------------" | tee -a "$SCAN_FILE"

# Define URLs
http_url="http://$domain"
https_url="https://$domain"

# Function to scan a given URL
scan_website() {
    local url=$1

    echo "Scanning $url..." | tee -a "$SCAN_FILE"

    # 1️⃣ Subdomain Enumeration
    echo "Running Subdomain Enumeration..." | tee -a "$SCAN_FILE"
    subfinder -d "$domain" | tee -a "$SCAN_FILE"
    assetfinder --subs-only "$domain" | tee -a "$SCAN_FILE"

    # 2️⃣ Full Port Scan
    echo "Running Full Port Scan (Nmap + Masscan)..." | tee -a "$SCAN_FILE"
    masscan -p1-65535 --rate=1000 "$domain" | tee -a "$SCAN_FILE"
    nmap -p- -A "$domain" | tee -a "$SCAN_FILE"

    # 3️⃣ Directory & File Enumeration
    echo "Running Directory Enumeration (Gobuster + Dirsearch)..." | tee -a "$SCAN_FILE"
    gobuster dir -u "$url" -w /usr/share/wordlists/dirb/common.txt -o "$SCAN_FILE"
    dirsearch -u "$url" -e php,html,js,txt -t 50 | tee -a "$SCAN_FILE"

    # 4️⃣ Technology & Service Detection
    echo "Detecting Technologies (WhatWeb + Wappalyzer)..." | tee -a "$SCAN_FILE"
    whatweb "$url" | tee -a "$SCAN_FILE"
    wappalyzer "$url" | tee -a "$SCAN_FILE"

    # 5️⃣ Vulnerability Scanning (Nikto, SQLMap, XSS, RCE)
    echo "Running Web Vulnerability Scans..." | tee -a "$SCAN_FILE"
    nikto -h "$url" | tee -a "$SCAN_FILE"
    sqlmap --url "$url" --batch --dbs | tee -a "$SCAN_FILE"
    xsser --url "$url" | tee -a "$SCAN_FILE"
    
    # 6️⃣ WordPress Security Scan
    echo "Checking if $domain is a WordPress site..." | tee -a "$SCAN_FILE"
    if whatweb "$url" | grep -q "WordPress"; then
        echo "Running WPScan..." | tee -a "$SCAN_FILE"
        wpscan --url "$url" --enumerate vp,ap,u | tee -a "$SCAN_FILE"
    else
        echo "Not a WordPress site. Skipping WPScan." | tee -a "$SCAN_FILE"
    fi

    echo "Scan complete for $url. Results saved in $SCAN_FILE"
    echo "--------------------------------------------------"
}

# Run scans for both HTTP and HTTPS versions
scan_website "$http_url"
scan_website "$https_url"
