#!/bin/bash

# Store the current dir variable
CUR_DIR=$(pwd)

echo -e "\n\033[1;32mPulling in latest changes for all repositories...\033[0m"

# Find all git repositories and update it
for gitfolder in $(find . -name ".git" | cut -c 3-); do

    # Go to the parent directory of .git
    cd "$gitfolder/..";

    echo
    echo -e "\033[1;33mUpdating $(basename $(pwd))\033[0m"

    # Do git pull
    git pull;

    # Get back to the CUR_DIR
    cd $CUR_DIR
done

echo -e "\n\033[32mComplete!\033[0m\n"
