#!/bin/bash

# Ensure required tools are installed (Arch Linux specific)
TOOLS_GROUPS=(
    "subfinder assetfinder amass"         # Subdomain Enumeration
    "masscan naabu nmap"                  # Port Scanning
    "gobuster dirsearch ffuf feroxbuster" # Directory Enumeration
    "whatweb wappalyzer"                   # Technology Detection
    "nikto sqlmap xsser nuclei"            # Web Vulnerability Scanning
    "wpscan"                               # WordPress Scanning
)

# Function to find the first available tool from a group
find_available_tool() {
    for tool in $1; do
        if command -v "$tool" &>/dev/null; then
            echo "$tool"
            return
        fi
    done
    echo "none"
}

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
echo "--------------------------------------------------" | tee "$SCAN_FILE"

# Subdomain Enumeration
subdomain_tool=$(find_available_tool "${TOOLS_GROUPS[0]}")
if [ "$subdomain_tool" != "none" ]; then
    echo "[+] Running Subdomain Enumeration with $subdomain_tool..." | tee -a "$SCAN_FILE"
    $subdomain_tool -d "$domain" | tee -a "$SCAN_FILE"
else
    echo "[!] No subdomain enumeration tools found. Skipping." | tee -a "$SCAN_FILE"
fi

# Port Scanning
port_scan_tool=$(find_available_tool "${TOOLS_GROUPS[1]}")
if [ "$port_scan_tool" != "none" ]; then
    echo "[+] Running Port Scan with $port_scan_tool..." | tee -a "$SCAN_FILE"
    if [ "$port_scan_tool" == "masscan" ]; then
        masscan -p1-65535 --rate=1000 "$domain" | tee -a "$SCAN_FILE"
    elif [ "$port_scan_tool" == "naabu" ]; then
        naabu -host "$domain" | tee -a "$SCAN_FILE"
    else
        nmap -p- -A "$domain" | tee -a "$SCAN_FILE"
    fi
else
    echo "[!] No port scanning tools found. Skipping." | tee -a "$SCAN_FILE"
fi

# Directory Enumeration
dir_enum_tool=$(find_available_tool "${TOOLS_GROUPS[2]}")
if [ "$dir_enum_tool" != "none" ]; then
    echo "[+] Running Directory Enumeration with $dir_enum_tool..." | tee -a "$SCAN_FILE"
    $dir_enum_tool -u "$url" -w /usr/share/wordlists/dirb/common.txt | tee -a "$SCAN_FILE"
else
    echo "[!] No directory enumeration tools found. Skipping." | tee -a "$SCAN_FILE"
fi

# Technology Detection
tech_tool=$(find_available_tool "${TOOLS_GROUPS[3]}")
if [ "$tech_tool" != "none" ]; then
    echo "[+] Detecting Technologies with $tech_tool..." | tee -a "$SCAN_FILE"
    $tech_tool "$url" | tee -a "$SCAN_FILE"
else
    echo "[!] No technology detection tools found. Skipping." | tee -a "$SCAN_FILE"
fi

# Web Vulnerability Scanning
vuln_scan_tool=$(find_available_tool "${TOOLS_GROUPS[4]}")
if [ "$vuln_scan_tool" != "none" ]; then
    echo "[+] Running Web Vulnerability Scan with $vuln_scan_tool..." | tee -a "$SCAN_FILE"
    $vuln_scan_tool -h "$url" | tee -a "$SCAN_FILE"
else
    echo "[!] No vulnerability scanning tools found. Skipping." | tee -a "$SCAN_FILE"
fi

# WordPress Security Scan
wp_tool=$(find_available_tool "${TOOLS_GROUPS[5]}")
if [ "$wp_tool" != "none" ] && whatweb "$url" | grep -q "WordPress"; then
    echo "[+] WordPress detected! Running WPScan..." | tee -a "$SCAN_FILE"
    $wp_tool --url "$url" --enumerate vp,ap,u | tee -a "$SCAN_FILE"
else
    echo "[!] WordPress scan skipped (not detected or missing WPScan)." | tee -a "$SCAN_FILE"
fi

echo "[✔] Scan complete for $domain. Results saved in $SCAN_FILE"
