import subprocess
import sys
import time

print("=" * 60)
print("🌱 SmartCrop AI — College Project Launcher")
print("=" * 60)
print("Choose what to run:")
print("1. Train and evaluate models (src.train)")
print("2. Generate EDA figures (src.eda)")
print("3. Start FastAPI backend (uvicorn backend.main:app)")
print("4. Start Streamlit frontend (streamlit run frontend/app.py)")
print("5. Run automated test suite (pytest)")
print("6. Launch Full Stack (FastAPI + Streamlit concurrently)")
print("=" * 60)

choice = input("Enter 1, 2, 3, 4, 5 or 6: ").strip()

commands = {
    "1": [sys.executable, "-m", "src.train"],
    "2": [sys.executable, "-m", "src.eda"],
    "3": [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--port", "8000"],
    "4": [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
    "5": [sys.executable, "-m", "pytest", "tests/test_prediction.py", "tests/test_preprocessing.py", "tests/test_api.py", "-v"],
}

if choice in commands:
    subprocess.run(commands[choice], check=False)
elif choice == "6":
    print("\nStarting FastAPI backend on port 8000...")
    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000"])
    time.sleep(2)
    print("Starting Streamlit dashboard on port 8501...")
    frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])
    print("\nBoth services running! Press Ctrl+C to terminate both.\n")
    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend.terminate()
        frontend.terminate()
else:
    print("Invalid choice.")
