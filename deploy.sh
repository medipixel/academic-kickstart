#!/bin/bash

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# Build the project.
hugo # if using a theme, replace with `hugo -t <YOURTHEME>`

# Go To Public folder
cd public
# Add changes to git.
git add .

# Commit changes.
msg="rebuilding site `date`"
if [ $# -eq 1 ]
then 
    echo "Input commit message!"
    msg="$1"
    git commit -m "$msg"
else
    echo "Don't input commit message!"
    git commit
fi
   
# Push source and build repos.
git push origin master
 
# Come Back up to the Project Root
cd ..

# Update submodule
git submodule update
git submodule update --remote

