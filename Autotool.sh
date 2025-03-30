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
    CORE_PACKAGES="figlet lolcat tor torsocks proxychains-ng nmap dirb whatweb gobuster nikto sqlmap wfuzz python-pip wordlists"
    
    # Additional BlackArch tools
    BLACKARCH_TOOLS="blackarch-recon blackarch-webapp blackarch-scanner blackarch-fuzzer blackarch-exploitation"
    
    # Check and install core packages
    for package in $CORE_PACKAGES; do
        if ! is_installed "$package"; then
            echo "Installing $package..."
            sudo pacman -S --noconfirm "$package" || echo "Warning: Failed to install $package"
        fi
    done
    
    # Install BlackArch tools if available
    if pacman -Sg blackarch &>/dev/null; then
        for tool in $BLACKARCH_TOOLS; do
            if ! is_installed "$tool"; then
                echo "Installing BlackArch tool group: $tool..."
                sudo pacman -S --noconfirm "$tool" || echo "Warning: Failed to install $tool"
            fi
        done
    else
        echo "Warning: BlackArch repository not enabled. Some advanced tools might not be available."
    fi
    
    # Some packages might be in AUR, try installing with yay if available
    if command -v yay &> /dev/null; then
        AUR_PACKAGES="zaproxy metasploit wpscan"
        for package in $AUR_PACKAGES; do
            if ! is_installed "$package"; then
                echo "Installing $package from AUR..."
                yay -S --noconfirm "$package" || echo "Warning: Failed to install $package"
            fi
        done
    else
        echo "Warning: yay is not installed. Some packages from AUR might not be installed."
    fi
    
    # Install wordlists if not present
    if [ ! -d "/usr/share/wordlists" ]; then
        echo "Installing additional wordlists..."
        sudo pacman -S --noconfirm wordlists seclists || echo "Warning: Failed to install wordlists"
    fi
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
    torsocks curl -s https://check.torproject.org | grep -q "Congratulations"
    if [ $? -eq 0 ]; then
        echo "Tor connection successful!"
        return 0
    else
        echo "Tor connection failed! Please check your Tor configuration."
        return 1
    fi
}

# Function to find wordlists
find_wordlist() {
    # Common paths for wordlists
    local paths=(
        "/usr/share/wordlists"
        "/usr/share/dirb/wordlists"
        "/usr/share/seclists"
        "/usr/share/wordlists/dirb"
        "/usr/share/wordlists/seclists"
    )
    
    local wordlist_name="$1"
    
    for path in "${paths[@]}"; do
        if [ -f "$path/$wordlist_name" ]; then
            echo "$path/$wordlist_name"
            return 0
        fi
        
        # Try to find the wordlist recursively
        found=$(find "$path" -name "$wordlist_name" -print -quit 2>/dev/null)
        if [ -n "$found" ]; then
            echo "$found"
            return 0
        fi
    done
    
    # If not found, return default
    case "$wordlist_name" in
        "common.txt") echo "/usr/share/wordlists/dirb/common.txt" ;;
        "small.txt") echo "/usr/share/wordlists/dirb/small.txt" ;;
        "big.txt") echo "/usr/share/wordlists/dirb/big.txt" ;;
        "directory-list-2.3-medium.txt") echo "/usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt" ;;
        *) echo "/usr/share/wordlists/dirb/common.txt" ;;
    esac
}

# Function to run Tor scan
run_tor_scan() {
    local command="$1"
    local output="$2"
    local args="$3"
    
    echo "[+] Running $command via Tor..."
    torsocks $command $args > "$output" 2>&1
    echo "    Results saved to $output"
}

# Function to run WordPress scan
run_wpscan() {
    echo "==== WORDPRESS SCAN ===="
    read -p "Enter target URL (WordPress site): " target_url
    read -p "Is this a .onion address? (y/n): " is_onion
    
    # Ask for output directory
    read -p "Enter output directory name [default: wpscan_$(date +%F)]: " output_dir
    output_dir=${output_dir:-wpscan_$(date +%F)}
    mkdir -p "$output_dir"
    
    if [[ "$is_onion" =~ ^[Yy] ]]; then
        if ! check_tor; then
            echo "Tor connection required for .onion scanning but failed. Exiting."
            exit 1
        fi
        echo "Running WordPress scan via Tor..."
        torsocks wpscan --url "$target_url" --output "$output_dir/wpscan_results.txt" --format cli-no-color --random-user-agent --throttle 100
    else
        echo "Running WordPress scan..."
        wpscan --url "$target_url" --output "$output_dir/wpscan_results.txt" --format cli-no-color
    fi
    
    echo "WordPress scan completed. Results saved to $output_dir/wpscan_results.txt"
}

# Function to run the All-in-One scan
run_all_in_one() {
    echo "==== ALL-IN-ONE SCANNER ===="
    read -p "Enter target URL (clearnet or .onion): " target_url
    read -p "Is this a .onion address? (y/n): " is_onion
    
    # Ask for output directory
    read -p "Enter output directory name [default: scan_$(date +%F)]: " output_dir
    output_dir=${output_dir:-scan_$(date +%F)}
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
    common_wordlist=$(find_wordlist "common.txt")
    small_wordlist=$(find_wordlist "small.txt")
    medium_wordlist=$(find_wordlist "directory-list-2.3-medium.txt")
    big_wordlist=$(find_wordlist "big.txt")
    
    echo "==== STEP 1: INITIAL RECON ===="
    if $use_tor; then
        run_tor_scan "curl --max-time 30" "$output_dir/site_content.html" "$target_url"
        run_tor_scan "whatweb -v" "$output_dir/whatweb.txt" "$target_url"
    else
        echo "[+] Checking site availability..."
        curl --max-time 30 -o "$output_dir/site_content.html" "$target_url" 2>/dev/null
        
        echo "[+] Running whatweb analysis..."
        whatweb -v "$target_url" --log-verbose="$output_dir/whatweb.txt"
    fi
    
    echo "==== STEP 2: PORT SCANNING ===="
    if $use_tor; then
        echo "[+] Running stealth port scan via Tor..."
        torsocks nmap -sT -Pn -n -p 80,443,8080,8443 -T2 --open "$target_url" -oN "$output_dir/ports.txt"
    else
        echo "[+] Running comprehensive port scan..."
        sudo nmap -sS -sV -p- -T4 --top-ports 1000 "$target_url" -oN "$output_dir/ports.txt"
    fi
    
    echo "==== STEP 3: CONTENT DISCOVERY ===="
    if $use_tor; then
        echo "[+] Running directory bruteforce via Tor (small wordlist)..."
        torsocks gobuster dir -u "$target_url" -w "$small_wordlist" -o "$output_dir/directories_small.txt" -t 10
        
        echo "[+] Running directory bruteforce via Tor (medium wordlist)..."
        torsocks gobuster dir -u "$target_url" -w "$medium_wordlist" -o "$output_dir/directories_medium.txt" -t 5
    else
        echo "[+] Running gobuster directory scan (small wordlist)..."
        gobuster dir -u "$target_url" -w "$small_wordlist" -o "$output_dir/directories_small.txt" -t 20
        
        echo "[+] Running gobuster directory scan (medium wordlist)..."
        gobuster dir -u "$target_url" -w "$medium_wordlist" -o "$output_dir/directories_medium.txt" -t 10
    fi
    
    echo "==== STEP 4: VULNERABILITY SCANNING ===="
    if $use_tor; then
        echo "[+] Running vulnerability scan via Tor..."
        torsocks nikto -h "$target_url" -o "$output_dir/nikto.txt"
        
        # Check if it's a WordPress site and run wpscan if available
        if grep -qi "wordpress" "$output_dir/whatweb.txt" && command -v wpscan &> /dev/null; then
            echo "[+] WordPress detected, running wpscan..."
            torsocks wpscan --url "$target_url" --output "$output_dir/wpscan_results.txt" --format cli-no-color --random-user-agent --throttle 100
        fi
    else
        echo "[+] Running nikto vulnerability scan..."
        nikto -h "$target_url" -o "$output_dir/nikto.txt"
        
        # Only run on clearnet (sqlmap via Tor is very slow)
        echo "[+] Running SQL injection scan..."
        sqlmap -u "$target_url" --forms --batch --crawl=2 -o -v 2 --output-dir="$output_dir/sqlmap"
        
        # Check if it's a WordPress site and run wpscan if available
        if grep -qi "wordpress" "$output_dir/whatweb.txt" && command -v wpscan &> /dev/null; then
            echo "[+] WordPress detected, running wpscan..."
            wpscan --url "$target_url" --output "$output_dir/wpscan_results.txt" --format cli-no-color
        fi
    fi
    
    echo "==== STEP 5: DARKWEB-SPECIFIC CHECKS ===="
    if $use_tor; then
        echo "[+] Checking for hidden service directories..."
        torsocks curl --max-time 30 "$target_url/robots.txt" -o "$output_dir/robots.txt" 2>/dev/null
        torsocks curl --max-time 30 "$target_url/onion-version.txt" -o "$output_dir/onion-version.txt" 2>/dev/null
        
        echo "[+] Checking for common hidden service admin panels..."
        torsocks gobuster dir -u "$target_url" -w "$small_wordlist" -o "$output_dir/admin_panels.txt" -x php,html,txt -t 5
    fi
    
    echo "==== SCAN SUMMARY ===="
    echo "All-in-One scan completed for: $target_url"
    echo "Results stored in: $output_dir/"
    echo "Scans performed:"
    echo "- Site recon and technology detection"
    echo "- Port scanning"
    echo "- Directory/content discovery"
    echo "- Vulnerability assessment"
    if $use_tor; then
        echo "- Darkweb-specific checks"
    fi
    
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
- directories_*.txt: Directory discovery results
- nikto.txt: Vulnerability scan results
EOF

    if $use_tor; then
        cat >> "$output_dir/SUMMARY.txt" << EOF
- robots.txt: Robots.txt file if found
- onion-version.txt: Onion service version if found
- admin_panels.txt: Common admin panel locations
EOF
    fi

    cat >> "$output_dir/SUMMARY.txt" << EOF

NEXT STEPS:
1. Review the vulnerability scan in nikto.txt
2. Check directory discovery results for sensitive content
3. Analyze port scan to identify service vulnerabilities
EOF
    
    echo "Summary file created: $output_dir/SUMMARY.txt"
}

# Function to run darkweb-specific scan
run_darkweb_scan() {
    echo "==== DARKWEB SCANNER ===="
    read -p "Enter .onion URL: " target_url
    
    if ! check_tor; then
        echo "Tor connection required for darkweb scanning but failed. Exiting."
        exit 1
    fi
    
    # Ask for output directory
    read -p "Enter output directory name [default: darkweb_scan_$(date +%F)]: " output_dir
    output_dir=${output_dir:-darkweb_scan_$(date +%F)}
    mkdir -p "$output_dir"
    
    # Find appropriate wordlists
    small_wordlist=$(find_wordlist "small.txt")
    medium_wordlist=$(find_wordlist "directory-list-2.3-medium.txt")
    darkweb_wordlist=$(find_wordlist "darkweb-top-10000.txt")
    
    echo "==== STEP 1: BASIC RECON ===="
    run_tor_scan "curl -v --max-time 30" "$output_dir/curl_verbose.txt" "$target_url"
    run_tor_scan "whatweb -v" "$output_dir/whatweb.txt" "$target_url"
    
    echo "==== STEP 2: PORT SCANNING ===="
    echo "[+] Running stealth port scan via Tor..."
    torsocks nmap -sT -Pn -n -p- --open --max-retries 1 --min-rate 50 --host-timeout 60m -T2 -v -oN "$output_dir/all_ports.txt" "$target_url"
    
    echo "==== STEP 3: CONTENT DISCOVERY ===="
    echo "[+] Running directory bruteforce via Tor (darkweb-specific wordlist)..."
    if [ -f "$darkweb_wordlist" ]; then
        torsocks gobuster dir -u "$target_url" -w "$darkweb_wordlist" -o "$output_dir/directories_darkweb.txt" -t 5
    else
        torsocks gobuster dir -u "$target_url" -w "$small_wordlist" -o "$output_dir/directories_small.txt" -t 5
    fi
    
    echo "[+] Checking for common hidden service files..."
    torsocks gobuster dir -u "$target_url" -w "$small_wordlist" -o "$output_dir/common_files.txt" -x php,txt,html -t 5
    
    echo "==== STEP 4: VULNERABILITY SCANNING ===="
    echo "[+] Running darkweb-optimized vulnerability scan..."
    torsocks nikto -h "$target_url" -o "$output_dir/nikto.txt" -Tuning x567
    
    echo "==== STEP 5: SPECIALIZED DARKWEB CHECKS ===="
    echo "[+] Checking for common darkweb admin panels..."
    torsocks gobuster dir -u "$target_url" -w "$small_wordlist" -o "$output_dir/admin_panels.txt" -x php -t 5
    
    echo "[+] Checking for PGP keys and contact info..."
    torsocks curl --max-time 30 "$target_url/contact.txt" -o "$output_dir/contact.txt" 2>/dev/null
    torsocks curl --max-time 30 "$target_url/pgp.txt" -o "$output_dir/pgp.txt" 2>/dev/null
    
    echo "==== SCAN SUMMARY ===="
    echo "Darkweb scan completed for: $target_url"
    echo "Results stored in: $output_dir/"
    
    # Create summary file
    cat > "$output_dir/SUMMARY.txt" << EOF
DARKWEB SCAN SUMMARY
====================
Target: $target_url
Scan Date: $(date)

FILES:
- curl_verbose.txt: Verbose curl output
- whatweb.txt: Web technology detection
- ports.txt: Open ports and services
- directories_*.txt: Directory discovery results
- nikto.txt: Vulnerability scan results
- admin_panels.txt: Common admin panel locations
- contact.txt: Contact information if found
- pgp.txt: PGP key if found

NEXT STEPS:
1. Review the vulnerability scan in nikto.txt
2. Check directory discovery results for hidden content
3. Analyze contact information for potential leads
EOF
    
    echo "Summary file created: $output_dir/SUMMARY.txt"
}

# Install required packages
install_packages

# Configure Tor
configure_tor

# Main display
clear
figlet -f slant "DarkNarco Eraser" -w 200 | lolcat
echo "Advanced Darkweb Scanner & Analysis Tool" | lolcat

# Main menu
echo -e "\nPlease select an option:"
echo "1. Standard Website Scanner"
echo "2. Vulnerability Scanner"
echo "3. Network Enumeration"
echo "4. Content Discovery"
echo "5. SQL Injection Scanner"
echo "6. Comprehensive Site Analysis"
echo "7. Anonymous Port Scanner"
echo "8. Darkweb Hidden Service Scanner"
echo "9. WordPress Vulnerability Scan"
echo "10. All-in-One Scan (Run multiple tools at once)"
echo "11. Darkweb-Specific Deep Scan"

read -p "Enter your choice (1-11): " choice

# WordPress Scan option
if [ "$choice" -eq 9 ]; then
    figlet -f slant "WP Scan" -w 200 | lolcat
    run_wpscan

# All-in-One Scan option
elif [ "$choice" -eq 10 ]; then
    figlet -f slant "All-in-One" -w 200 | lolcat
    run_all_in_one

# Darkweb-Specific Deep Scan
elif [ "$choice" -eq 11 ]; then
    figlet -f slant "Darkweb Scan" -w 200 | lolcat
    run_darkweb_scan

# Darkweb Hidden Service Scanner
elif [ "$choice" -eq 8 ]; then
    figlet -f slant "Onion Scanner" -w 200 | lolcat
    run_darkweb_scan

# Other options (similar structure but with Tor support)
else
    echo "Selected option: $choice"
    read -p "Enter target URL or IP: " target
    read -p "Is this a .onion address? (y/n): " is_onion
    read -p "Enter output directory name [default: scan_$(date +%F)]: " output_dir
    output_dir=${output_dir:-scan_$(date +%F)}
    mkdir -p "$output_dir"
    
    use_tor=false
    if [[ "$is_onion" =~ ^[Yy] ]]; then
        use_tor=true
        if ! check_tor; then
            echo "Tor connection required for .onion scanning but failed. Exiting."
            exit 1
        fi
    fi
    
    case "$choice" in
        1)  # Standard Website Scanner
            if $use_tor; then
                run_tor_scan "whatweb -v" "$output_dir/whatweb.txt" "$target"
            else
                whatweb -v "$target" --log-verbose="$output_dir/whatweb.txt"
            fi
            ;;
        2)  # Vulnerability Scanner
            if $use_tor; then
                run_tor_scan "nikto -h" "$output_dir/nikto.txt" "$target"
            else
                nikto -h "$target" -o "$output_dir/nikto.txt"
            fi
            ;;
        3)  # Network Enumeration
            if $use_tor; then
                torsocks nmap -sT -Pn -n -p 80,443,8080,8443 -T2 --open "$target" -oN "$output_dir/ports.txt"
            else
                sudo nmap -sS -sV -p- -T4 --top-ports 1000 "$target" -oN "$output_dir/ports.txt"
            fi
            ;;
        4)  # Content Discovery
            wordlist=$(find_wordlist "directory-list-2.3-medium.txt")
            if $use_tor; then
                torsocks gobuster dir -u "$target" -w "$wordlist" -o "$output_dir/directories.txt" -t 5
            else
                gobuster dir -u "$target" -w "$wordlist" -o "$output_dir/directories.txt" -t 20
            fi
            ;;
        5)  # SQL Injection Scanner
            if $use_tor; then
                torsocks sqlmap -u "$target" --forms --batch --crawl=2 -o -v 2 --tor --output-dir="$output_dir/sqlmap"
            else
                sqlmap -u "$target" --forms --batch --crawl=3 -o -v 2 --output-dir="$output_dir/sqlmap"
            fi
            ;;
        6)  # Comprehensive Site Analysis
            run_all_in_one
            ;;
        7)  # Anonymous Port Scanner
            if check_tor; then
                torsocks nmap -sT -Pn -n -p- -T2 --open "$target" -oN "$output_dir/ports.txt"
            fi
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
    
    echo "Scan completed. Results saved to $output_dir/"
fi

echo "Operation completed."
echo "Thanks for using DarkNarco Eraser!" | lolcat