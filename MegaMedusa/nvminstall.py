import os
import sys

# Install required packages
os.system('sudo pacman -Sy --noconfirm npm curl')

# Install NVM
os.system('curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash')

# Load NVM and install the latest LTS version of Node.js
os.system('export NVM_DIR="$HOME/.nvm" && source "$NVM_DIR/nvm.sh" && nvm install --lts')

# Success message
print("\x1b[31m[\x1b[33mMedusa\x1b[31m]\x1b[0m \x1b[32m> \x1b[33mInstallation Successful")
