@echo off
echo 🚀 Heart Disease Analysis - GitHub Upload Script
echo ================================================

echo.
echo ⚠️  IMPORTANT: Before running this script, you need to:
echo 1. Install Git from https://git-scm.com/download/win
echo 2. Create a GitHub account at https://github.com
echo 3. Create a repository named 'heart-disease-analysis'
echo 4. Replace 'YOURUSERNAME' below with your actual GitHub username
echo.

set /p GITHUB_USERNAME="Enter your GitHub username: "

echo.
echo Step 1: Initializing Git repository...
git init

echo.
echo Step 2: Adding all files to staging...
git add .

echo.
echo Step 3: Creating initial commit...
git commit -m "Initial commit: Heart Disease Analysis ML Pipeline"

echo.
echo Step 4: Adding remote repository...
git remote add origin https://github.com/%GITHUB_USERNAME%/heart-disease-analysis.git

echo.
echo Step 5: Pushing to GitHub...
git push -u origin main

echo.
echo ✅ Upload completed!
echo.
echo Your repository is now available at:
echo https://github.com/%GITHUB_USERNAME%/heart-disease-analysis
echo.
echo Next steps:
echo 1. Go to your GitHub repository
echo 2. Update README.md with your actual details
echo 3. Add project topics/tags
echo 4. Share your repository!
echo.
pause
