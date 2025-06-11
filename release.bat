@echo off
setlocal enabledelayedexpansion

:: Release script for netcdfkit (Windows version)
:: This script automates the release process by:
:: 1. Checking working directory is clean
:: 2. Bumping version using bumpversion
:: 3. Building the package
:: 4. Uploading to PyPI

echo [INFO] Starting release process...

:: Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not in a git repository!
    exit /b 1
)

:: Check if working directory is clean
for /f %%i in ('git status --porcelain') do (
    echo [ERROR] Working directory is not clean. Please commit or stash your changes.
    git status --short
    exit /b 1
)

:: Get current branch
for /f "tokens=*" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i
echo [INFO] Current branch: !CURRENT_BRANCH!

:: Check if we're on main/master branch (optional warning)
if not "!CURRENT_BRANCH!"=="main" if not "!CURRENT_BRANCH!"=="master" (
    echo [WARNING] You're not on main/master branch. Current branch: !CURRENT_BRANCH!
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" (
        echo [INFO] Release cancelled.
        exit /b 0
    )
)

:: Get version bump type
set BUMP_TYPE=%1
if "!BUMP_TYPE!"=="" set BUMP_TYPE=patch
if not "!BUMP_TYPE!"=="major" if not "!BUMP_TYPE!"=="minor" if not "!BUMP_TYPE!"=="patch" (
    echo [ERROR] Invalid bump type: !BUMP_TYPE!
    echo Usage: %0 [major^|minor^|patch]
    echo   major: 1.0.0 -^> 2.0.0
    echo   minor: 1.0.0 -^> 1.1.0
    echo   patch: 1.0.0 -^> 1.0.1 ^(default^)
    exit /b 1
)

:: Get current version
for /f "tokens=2 delims==" %%i in ('findstr "version = " pyproject.toml') do (
    set CURRENT_VERSION=%%i
    set CURRENT_VERSION=!CURRENT_VERSION: =!
    set CURRENT_VERSION=!CURRENT_VERSION:"=!
)
echo [INFO] Current version: !CURRENT_VERSION!

:: Confirm release
echo.
echo [INFO] About to perform a !BUMP_TYPE! release from version !CURRENT_VERSION!
set /p CONTINUE="Continue? (y/N): "
if /i not "!CONTINUE!"=="y" (
    echo [INFO] Release cancelled.
    exit /b 0
)

:: Bump version using bumpversion
echo [INFO] Bumping version (!BUMP_TYPE!)...
uv run bumpver update --!BUMP_TYPE!
if errorlevel 1 (
    echo [ERROR] Failed to bump version
    exit /b 1
)

:: Get new version
for /f "tokens=2 delims==" %%i in ('findstr "version = " pyproject.toml') do (
    set NEW_VERSION=%%i
    set NEW_VERSION=!NEW_VERSION: =!
    set NEW_VERSION=!NEW_VERSION:"=!
)
echo [INFO] New version: !NEW_VERSION!

:: Build the package
echo [INFO] Building package...
uv build
if errorlevel 1 (
    echo [ERROR] Failed to build package
    exit /b 1
)

:: Check if built files exist
if not exist "dist\netcdfkit-!NEW_VERSION!.tar.gz" (
    echo [ERROR] Built tar.gz file not found in dist/
    exit /b 1
)
if not exist "dist\netcdfkit-!NEW_VERSION!-py3-none-any.whl" (
    echo [ERROR] Built wheel file not found in dist/
    exit /b 1
)

:: Push changes to remote
echo [INFO] Pushing changes to remote...
git push origin !CURRENT_BRANCH!
if errorlevel 1 (
    echo [ERROR] Failed to push changes
    exit /b 1
)

:: Push tags
echo [INFO] Pushing tags...
git push origin --tags
if errorlevel 1 (
    echo [ERROR] Failed to push tags
    exit /b 1
)

:: Upload to PyPI
echo [INFO] Uploading to PyPI...
echo About to upload to PyPI. Make sure you have your PyPI credentials configured.
set /p UPLOAD="Continue with PyPI upload? (y/N): "
if /i "!UPLOAD!"=="y" (
    uv publish
    if errorlevel 1 (
        echo [ERROR] Failed to upload to PyPI
        echo [WARNING] Version has been tagged and pushed to git.
        echo [WARNING] You can manually upload later with: uv publish
        exit /b 1
    )
    echo [INFO] Successfully uploaded to PyPI!
) else (
    echo [WARNING] Skipped PyPI upload.
    echo [INFO] To upload later, run: uv publish
)

:: Summary
echo.
echo [INFO] Release !NEW_VERSION! completed successfully!
echo [INFO] Changes have been pushed to git with tag v!NEW_VERSION!
if /i "!UPLOAD!"=="y" (
    echo [INFO] Package has been uploaded to PyPI
)

:: Show what was created
echo.
echo [INFO] Created files:
dir dist\netcdfkit-!NEW_VERSION!*

echo.
echo [INFO] Git log (last 3 commits):
git log --oneline -3

pause
