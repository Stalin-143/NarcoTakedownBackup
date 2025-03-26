import os

# Install required packages
os.system('sudo pacman -Sy --noconfirm jdk17-openjdk nodejs-lts qb64')

# Install required Node.js packages globally
os.system('npm install -g url https net crypto axios request header-generator child_process')

# Run MegaMedusa.js
os.system('node MegaMedusa.js')

# Success message
print("\x1b[31m[\x1b[33mMedusa\x1b[31m]\x1b[0m \x1b[32m> \x1b[33mInstallation Successful")
