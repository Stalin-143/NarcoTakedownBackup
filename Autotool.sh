#!/bin/bash

# Function to check if a package is installed
is_installed() {
    pacman -Qi "$1" &> /dev/null
    return $?
}

# Function to install packages with proper error handling
install_packages() {
    echo "Installing required packages for Arch Linux..."
    
    # Core packages from official repositories
    CORE_PACKAGES="figlet lolcat tor proxychains-ng nmap dirb whatweb gobuster nikto sqlmap wfuzz python-pip"
    
    # Check and install core packages
    for package in $CORE_PACKAGES; do
        if ! is_installed "$package"; then
            echo "Installing $package..."
            sudo pacman -S --noconfirm "$package" || echo "Warning: Failed to install $package"
        fi
    done
    
    # Some packages might be in AUR, try installing with yay if available
    if command -v yay &> /dev/null; then
        AUR_PACKAGES="zaproxy metasploit"
        for package in $AUR_PACKAGES; do
            if ! is_installed "$package"; then
                echo "Installing $package from AUR..."
                yay -S --noconfirm "$package" || echo "Warning: Failed to install $package"
            fi
        done
    else
        echo "Warning: yay is not installed. Some packages from AUR might not be installed."
    fi
    
    # Install Python packages
    echo "Installing required Python packages..."
    pip install --user requests beautifulsoup4 stem python-nmap torrequest mechanize || echo "Warning: Failed to install some Python packages"
}

# Function to configure Tor service
configure_tor() {
    echo "Configuring Tor service..."
    
    # Check if Tor is installed
    if ! command -v tor &> /dev/null; then
        echo "Error: Tor is not installed. Trying to install it."
        sudo pacman -S --noconfirm tor || { echo "Failed to install Tor. Exiting."; exit 1; }
    fi
    
    # Enable and start Tor service
    if ! systemctl is-active --quiet tor; then
        sudo systemctl start tor || { echo "Failed to start Tor service. Exiting."; exit 1; }
    fi
    
    # Create default proxychains config if it doesn't exist
    if [ ! -f /etc/proxychains.conf ]; then
        echo "Creating default proxychains configuration..."
        sudo bash -c 'cat > /etc/proxychains.conf << EOF
# proxychains.conf
strict_chain
proxy_dns
remote_dns_subnet 224
tcp_read_time_out 15000
tcp_connect_time_out 8000
[ProxyList]
socks5 127.0.0.1 9050
EOF'
    fi
}

# Function to check Tor connection
check_tor() {
    echo "Checking Tor connection..."
    curl --socks5 127.0.0.1:9050 --socks5-hostname 127.0.0.1:9050 -s https://check.torproject.org | grep -q "Congratulations"
    if [ $? -eq 0 ]; then
        echo "Tor connection successful!"
        return 0
    else
        echo "Tor connection failed! Please check your Tor configuration."
        return 1
    fi
}

# Function to run the All-in-One scan
run_all_in_one() {
    echo "==== ALL-IN-ONE SCANNER ===="
    read -p "Enter target URL (clearnet or .onion): " target_url
    read -p "Is this a .onion address? (y/n): " is_onion
    
    # Create output directory
    timestamp=$(date +%F_%H-%M-%S)
    output_dir="all_in_one_scan_${timestamp}"
    mkdir -p "$output_dir"
    echo "All results will be saved to: $output_dir/"
    
    # Determine if we should use Tor for this scan
    use_tor=false
    if [[ "$is_onion" =~ ^[Yy] ]]; then
        use_tor=true
        if ! check_tor; then
            echo "Tor connection required for .onion scanning but failed. Exiting."
            exit 1
        fi
    fi
    
    # Find appropriate wordlists
    common_wordlist="/usr/share/wordlists/dirb/common.txt"
    small_wordlist="/usr/share/wordlists/dirb/small.txt"
    
    # Fallback for Arch Linux if standard paths don't exist
    if [ ! -f "$common_wordlist" ]; then
        common_wordlist="/usr/share/dirb/wordlists/common.txt"
    fi
    if [ ! -f "$small_wordlist" ]; then
        small_wordlist="/usr/share/dirb/wordlists/small.txt"
    fi
    
    echo "==== STEP 1: INITIAL RECON ===="
    if $use_tor; then
        echo "[+] Checking site availability..."
        proxychains curl --max-time 30 -o "$output_dir/site_content.html" "$target_url" 2>/dev/null
        
        echo "[+] Running whatweb analysis..."
        proxychains whatweb -v "$target_url" --log-verbose="$output_dir/whatweb.txt" 2>/dev/null
    else
        echo "[+] Checking site availability..."
        curl --max-time 30 -o "$output_dir/site_content.html" "$target_url" 2>/dev/null
        
        echo "[+] Running whatweb analysis..."
        whatweb -v "$target_url" --log-verbose="$output_dir/whatweb.txt"
    fi
    
    echo "==== STEP 2: PORT SCANNING ===="
    if $use_tor; then
        echo "[+] Running stealth port scan via Tor..."
        proxychains nmap -sT -P0 -p 80,443,8080,8443 -T2 "$target_url" -oN "$output_dir/ports.txt" 2>/dev/null
    else
        echo "[+] Running comprehensive port scan..."
        sudo nmap -sS -sV -p- -T4 --top-ports 1000 "$target_url" -oN "$output_dir/ports.txt"
    fi
    
    echo "==== STEP 3: CONTENT DISCOVERY ===="
    if $use_tor; then
        echo "[+] Running directory bruteforce via Tor..."
        proxychains wfuzz -c -w "$small_wordlist" --hc 404 "$target_url/FUZZ" -o "$output_dir/directories.txt" 2>/dev/null
    else
        echo "[+] Running gobuster directory scan..."
        gobuster dir -u "$target_url" -w "$common_wordlist" -o "$output_dir/directories.txt"
        
        echo "[+] Running dirb scan..."
        dirb "$target_url" "$common_wordlist" -o "$output_dir/dirb.txt"
    fi
    
    echo "==== STEP 4: VULNERABILITY SCANNING ===="
    if $use_tor; then
        echo "[+] Running vulnerability scan via Tor..."
        proxychains nikto -h "$target_url" -o "$output_dir/nikto.txt" 2>/dev/null
    else
        echo "[+] Running nikto vulnerability scan..."
        nikto -h "$target_url" -o "$output_dir/nikto.txt"
        
        # Only run on clearnet (sqlmap via Tor is very slow)
        echo "[+] Running SQL injection scan..."
        sqlmap -u "$target_url" --forms --batch --crawl=2 -o -v 2 --output-dir="$output_dir/sqlmap"
    fi
    
    echo "==== STEP 5: DATA COLLECTION ===="
    if $use_tor; then
        echo "[+] Running site scraper via Tor..."
        echo "#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import os
import sys

# Set up for Tor proxy
session = requests.session()
session.proxies = {'http': 'socks5h://127.0.0.1:9050',
                   'https': 'socks5h://127.0.0.1:9050'}

try:
    response = session.get('$target_url', timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all links
    links = [a.get('href') for a in soup.find_all('a', href=True)]
    
    # Extract all text
    text = soup.get_text()
    
    # Save results
    with open('$output_dir/extracted_links.txt', 'w') as f:
        for link in links:
            f.write(link + '\\n')
    
    with open('$output_dir/extracted_text.txt', 'w') as f:
        f.write(text)
    
    print('Scraping complete. Found', len(links), 'links')
except Exception as e:
    print('Error during scraping:', str(e))
" > "$output_dir/tor_scraper.py"
        python "$output_dir/tor_scraper.py" || python3 "$output_dir/tor_scraper.py"
    else
        echo "[+] Running site scraper..."
        echo "#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import os
import sys

try:
    response = requests.get('$target_url', timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all links
    links = [a.get('href') for a in soup.find_all('a', href=True)]
    
    # Extract all text
    text = soup.get_text()
    
    # Save results
    with open('$output_dir/extracted_links.txt', 'w') as f:
        for link in links:
            f.write(link + '\\n')
    
    with open('$output_dir/extracted_text.txt', 'w') as f:
        f.write(text)
    
    print('Scraping complete. Found', len(links), 'links')
except Exception as e:
    print('Error during scraping:', str(e))
" > "$output_dir/scraper.py"
        python "$output_dir/scraper.py" || python3 "$output_dir/scraper.py"
    fi
    
    echo "==== SCAN SUMMARY ===="
    echo "All-in-One scan completed for: $target_url"
    echo "Results stored in: $output_dir/"
    echo "Scans performed:"
    echo "- Site recon and technology detection"
    echo "- Port scanning"
    echo "- Directory/content discovery"
    echo "- Vulnerability assessment"
    echo "- Data extraction and link discovery"
    
    # Create summary file
    cat > "$output_dir/SUMMARY.txt" << EOF
ALL-IN-ONE SCAN SUMMARY
=======================
Target: $target_url
Scan Date: $(date)
Tor Mode: $use_tor

FILES:
- site_content.html: Raw HTML content of the main page
- whatweb.txt: Web technology detection results
- ports.txt: Open ports and services
- directories.txt: Directory discovery results
- nikto.txt: Vulnerability scan results
- extracted_links.txt: All links found on the site
- extracted_text.txt: All text content extracted from the site

NEXT STEPS:
1. Review the vulnerability scan in nikto.txt
2. Check directory discovery results for sensitive content
3. Analyze extracted links for additional targets
4. Examine port scan to identify service vulnerabilities
EOF
    
    echo "Summary file created: $output_dir/SUMMARY.txt"
}

# Install required packages
install_packages

# Configure Tor
configure_tor

# Main display
clear
figlet -f slant "Narco Eraser" -w 200 | lolcat
echo "Advanced Web Scanner & Analysis Tool for Arch Linux" | lolcat

# Main menu
echo -e "\nPlease select an option:"
echo "1. Website Scraper"
echo "2. Narcotics Website Finder"
echo "3. Vulnerability Scanner"
echo "4. Network Enumeration"
echo "5. Content Discovery"
echo "6. SQL Injection Scanner"
echo "7. Comprehensive Site Analysis"
echo "8. Anonymous Port Scanner"
echo "9. Tor Hidden Service Scanner"
echo "10. All-in-One Scan (Run multiple tools at once)"

read -p "Enter your choice (1-10): " choice

# All-in-One Scan option
if [ "$choice" -eq 10 ]; then
    figlet -f slant "All-in-One" -w 200 | lolcat
    run_all_in_one

# Website Scraper
elif [ "$choice" -eq 1 ]; then
    echo "Select the scraping option:"
    echo "1. Clear Net Scraper"
    echo "2. Dark Web Scraper"
    
    read -p "Enter your choice (1/2): " scraper_choice
    
    if [ "$scraper_choice" -eq 1 ]; then
        figlet -f slant "Web Scraper" -w 200 | lolcat
        echo "Running Clear Net Scraper..."
        python web_scrapper.py || python3 web_scrapper.py
    elif [ "$scraper_choice" -eq 2 ]; then
        figlet -f slant "Tor Scraper" -w 200 | lolcat
        echo "Running Dark Web Scraper..."
        if check_tor; then
            python tor_scrapper.py || python3 tor_scrapper.py
        fi
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi

# Narcotics Website Finder
elif [ "$choice" -eq 2 ]; then
    figlet -f slant "Narcotics Website Finder" -w 200 | lolcat
    echo "Running Narcotics Website Finder..."
    if check_tor; then
        python webfinder.py || python3 webfinder.py
    fi

# Vulnerability Scanner
elif [ "$choice" -eq 3 ]; then
    figlet -f slant "Vuln Scanner" -w 200 | lolcat
    echo "Select scan type:"
    echo "1. Clear Net Vulnerability Scan"
    echo "2. Dark Web Vulnerability Scan"
    
    read -p "Enter your choice (1/2): " vuln_choice
    read -p "Enter target URL: " target_url
    
    if [ "$vuln_choice" -eq 1 ]; then
        echo "Running vulnerability scan on $target_url..."
        nikto -h "$target_url" -o "vuln_scan_$(date +%F).txt"
        whatweb -v "$target_url" --log-verbose="whatweb_$(date +%F).txt"
    elif [ "$vuln_choice" -eq 2 ]; then
        echo "Running vulnerability scan on Tor hidden service..."
        if check_tor; then
            proxychains nikto -h "$target_url" -o "tor_vuln_scan_$(date +%F).txt"
            proxychains whatweb -v "$target_url" --log-verbose="tor_whatweb_$(date +%F).txt"
        fi
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi

# Network Enumeration
elif [ "$choice" -eq 4 ]; then
    figlet -f slant "Network Enum" -w 200 | lolcat
    echo "Select scan type:"
    echo "1. Clear Net Network Scan"
    echo "2. Anonymous Network Scan (via Tor)"
    
    read -p "Enter your choice (1/2): " network_choice
    read -p "Enter target (IP/domain): " target
    
    if [ "$network_choice" -eq 1 ]; then
        echo "Running network enumeration on $target..."
        sudo nmap -sS -sV -p- -T4 -A -v "$target" -oN "nmap_scan_$(date +%F).txt"
    elif [ "$network_choice" -eq 2 ]; then
        echo "Running anonymous network enumeration on $target..."
        if check_tor; then
            proxychains nmap -sT -P0 -sV -p 1-1000 -T2 "$target" -oN "tor_nmap_scan_$(date +%F).txt"
        fi
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi

# Content Discovery
elif [ "$choice" -eq 5 ]; then
    figlet -f slant "Content Discovery" -w 200 | lolcat
    echo "Select scan type:"
    echo "1. Clear Web Content Discovery"
    echo "2. Dark Web Content Discovery"
    
    read -p "Enter your choice (1/2): " content_choice
    read -p "Enter target URL: " target_url
    
    wordlist="/usr/share/wordlists/dirb/common.txt"
    # Fallback for Arch Linux if standard path doesn't exist
    if [ ! -f "$wordlist" ]; then
        wordlist="/usr/share/dirb/wordlists/common.txt"
    fi
    
    if [ "$content_choice" -eq 1 ]; then
        echo "Running content discovery on $target_url..."
        gobuster dir -u "$target_url" -w "$wordlist" -o "gobuster_$(date +%F).txt"
        dirb "$target_url" "$wordlist" -o "dirb_$(date +%F).txt"
    elif [ "$content_choice" -eq 2 ]; then
        echo "Running content discovery on Tor hidden service..."
        if check_tor; then
            proxychains gobuster dir -u "$target_url" -w "$wordlist" -o "tor_gobuster_$(date +%F).txt"
            proxychains wfuzz -c -w "$wordlist" --hc 404 "$target_url/FUZZ" -o "tor_wfuzz_$(date +%F).txt"
        fi
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi

# SQL Injection Scanner
elif [ "$choice" -eq 6 ]; then
    figlet -f slant "SQLi Scanner" -w 200 | lolcat
    echo "Select scan type:"
    echo "1. Clear Web SQL Injection Scan"
    echo "2. Dark Web SQL Injection Scan"
    
    read -p "Enter your choice (1/2): " sqli_choice
    read -p "Enter target URL: " target_url
    
    if [ "$sqli_choice" -eq 1 ]; then
        echo "Running SQL injection scan on $target_url..."
        sqlmap -u "$target_url" --forms --batch --crawl=5 -o -v 2 --output-dir="sqlmap_$(date +%F)"
    elif [ "$sqli_choice" -eq 2 ]; then
        echo "Running SQL injection scan on Tor hidden service..."
        if check_tor; then
            proxychains sqlmap -u "$target_url" --forms --batch --crawl=3 -o -v 2 --tor --output-dir="tor_sqlmap_$(date +%F)"
        fi
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi

# Comprehensive Site Analysis
elif [ "$choice" -eq 7 ]; then
    figlet -f slant "Full Analysis" -w 200 | lolcat
    echo "Select scan type:"
    echo "1. Clear Web Full Analysis"
    echo "2. Dark Web Full Analysis"
    
    read -p "Enter your choice (1/2): " full_choice
    read -p "Enter target URL: " target_url
    
    mkdir -p "comprehensive_scan_$(date +%F)"
    output_dir="comprehensive_scan_$(date +%F)"
    
    wordlist="/usr/share/wordlists/dirb/common.txt"
    # Fallback for Arch Linux if standard path doesn't exist
    if [ ! -f "$wordlist" ]; then
        wordlist="/usr/share/dirb/wordlists/common.txt"
    fi
    
    if [ "$full_choice" -eq 1 ]; then
        echo "Running comprehensive analysis on $target_url..."
        
        echo "Starting network scan..."
        sudo nmap -sS -sV -p- -T4 -A "$target_url" -oN "$output_dir/nmap.txt"
        
        echo "Starting vulnerability scan..."
        nikto -h "$target_url" -o "$output_dir/nikto.txt"
        
        echo "Starting content discovery..."
        gobuster dir -u "$target_url" -w "$wordlist" -o "$output_dir/gobuster.txt"
        
        echo "Starting web technology analysis..."
        whatweb -v "$target_url" --log-verbose="$output_dir/whatweb.txt"
        
        echo "Starting SQL injection scan..."
        sqlmap -u "$target_url" --forms --batch --crawl=3 -o -v 2 --output-dir="$output_dir/sqlmap"
        
        # Check if ZAP is available
        if command -v zap-cli &> /dev/null; then
            echo "Starting ZAP analysis..."
            zap-cli quick-scan -s all -r "$target_url" -o "$output_dir/zap_report.html"
        else
            echo "ZAP not found, skipping ZAP analysis."
        fi
        
    elif [ "$full_choice" -eq 2 ]; then
        echo "Running comprehensive analysis on Tor hidden service..."
        if check_tor; then
            echo "Starting anonymous network scan..."
            proxychains nmap -sT -P0 -sV -p 1-1000 -T2 "$target_url" -oN "$output_dir/tor_nmap.txt"
            
            echo "Starting vulnerability scan..."
            proxychains nikto -h "$target_url" -o "$output_dir/tor_nikto.txt"
            
            echo "Starting content discovery..."
            proxychains wfuzz -c -w "$wordlist" --hc 404 "$target_url/FUZZ" -o "$output_dir/tor_wfuzz.txt"
            
            echo "Starting web technology analysis..."
            proxychains whatweb -v "$target_url" --log-verbose="$output_dir/tor_whatweb.txt"
            
            echo "Starting SQL injection scan..."
            proxychains sqlmap -u "$target_url" --forms --batch --crawl=2 -o -v 2 --tor --output-dir="$output_dir/tor_sqlmap"
        fi
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi
    
    echo "Comprehensive scan complete. Results saved to $output_dir/"

# Anonymous Port Scanner
elif [ "$choice" -eq 8 ]; then
    figlet -f slant "Anon Port Scan" -w 200 | lolcat
    read -p "Enter target (IP/domain): " target
    
    if check_tor; then
        echo "Running anonymous port scan on $target..."
        proxychains nmap -sT -P0 -p1-65535 -T2 "$target" -oN "anon_portscan_$(date +%F).txt"
        echo "Scan complete. Results saved to anon_portscan_$(date +%F).txt"
    fi

# Tor Hidden Service Scanner
elif [ "$choice" -eq 9 ]; then
    figlet -f slant "Onion Scanner" -w 200 | lolcat
    read -p "Enter .onion URL: " onion_url
    
    if check_tor; then
        echo "Running Tor hidden service scan on $onion_url..."
        
        mkdir -p "onion_scan_$(date +%F)"
        output_dir="onion_scan_$(date +%F)"
        
        echo "Checking site availability..."
        proxychains curl --max-time 30 -o "$output_dir/site_content.html" "$onion_url"
        
        echo "Running stealth port scan..."
        proxychains nmap -sT -P0 -p 80,443 -T2 "$onion_url" -oN "$output_dir/ports.txt"
        
        echo "Running web technology analysis..."
        proxychains whatweb -v "$onion_url" --log-verbose="$output_dir/technologies.txt"
        
        wordlist="/usr/share/wordlists/dirb/small.txt"
        # Fallback for Arch Linux if standard path doesn't exist
        if [ ! -f "$wordlist" ]; then
            wordlist="/usr/share/dirb/wordlists/small.txt"
        fi
        
        echo "Running basic content discovery..."
        proxychains wfuzz -c -w "$wordlist" --hc 404 "$onion_url/FUZZ" -o "$output_dir/directories.txt"
        
        echo "Scan complete. Results saved to $output_dir/"
    fi

else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "Operation completed."
echo "Thanks for using Narco Eraser!" | lolcat