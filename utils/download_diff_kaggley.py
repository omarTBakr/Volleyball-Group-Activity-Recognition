import subprocess
import os

NOTEBOOK_ID = "omaraymanbakr/volleyball-project-baseline-1"
LOCAL_OUTPUT = "./kaggle_outputs"  # your local outputs folder

os.makedirs(LOCAL_OUTPUT, exist_ok=True)

# 1. Pull all output files from Kaggle into outputs/
print("⬇️  Pulling outputs from Kaggle...")
subprocess.run(
    ["kaggle", "kernels", "output", NOTEBOOK_ID, "-p", LOCAL_OUTPUT], check=True
)

# 2. Stage new/changed files
print("📦 Staging changes...")
subprocess.run(["git", "add", LOCAL_OUTPUT], check=True)

# 3. Commit
print("💾 Committing...")
subprocess.run(
    ["git", "commit", "-m", "sync: pull latest outputs from Kaggle"], check=True
)

# 4. Push to GitHub (LFS handles large files automatically)
print("🚀 Pushing to GitHub...")
subprocess.run(["git", "push"], check=True)

print("✅ Done! Outputs are now local and on GitHub.")
