#!/bin/bash
sudo apt install -y figlet >/dev/null 2>&1  # Hide installation output
sudo apt install -y lolcat >/dev/null 2>&1  # Hide installation output
figlet -f slant "Narco Eraser" -w 200 | lolcat
echo "Please select an option:"
echo "1. website scraper"
echo "2. Narcotics Website Finder"

read -p "Enter your choice (1/2): " choice

if [ "$choice" -eq 1 ]; then

    echo "Select the scraping option:"

    echo "1. Clear Net Scraper"
    echo "2. Dark Web Scraper"

    read -p "Enter your choice (1/2): " scraper_choice

    if [ "$scraper_choice" -eq 1 ]; then
        figlet -f slant "web scraper" -w 200 | lolcat
        echo "Running Clear Net Scraper..."
        python3 web_scrapper.py
    elif [ "$scraper_choice" -eq 2 ]; then
        figlet -f slant "Tor scraper" -w 200 | lolcat
        echo "Running Dark Web Scraper..."
        python3 tor_scrapper.py
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi

elif [ "$choice" -eq 2 ]; then
    figlet -f slant "Narcotics Website Finder" -w 200 | lolcat
    echo "Running Narcotics Website Finder..."
    python3 webfinder.py

else
    echo "Invalid choice. Exiting."
    exit 1
fi
