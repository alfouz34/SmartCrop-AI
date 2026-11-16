import subprocess
import sys

print("Choose what to run:")
print("1. Train and evaluate models")
print("2. Generate EDA figures")
print("3. Start FastAPI backend")
print("4. Start Streamlit frontend")

choice = input("Enter 1, 2, 3 or 4: ").strip()
commands = {
    "1": [sys.executable, "-m", "src.train"],
    "2": [sys.executable, "-m", "src.eda"],
    "3": [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload"],
    "4": [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
}
if choice not in commands:
    print("Invalid choice.")
else:
    subprocess.run(commands[choice], check=False)
