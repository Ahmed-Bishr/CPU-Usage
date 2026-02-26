import subprocess
import sys
import webbrowser
import threading


HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"


def open_browser():
    """Open the app in the default browser after a short delay."""
    import time

    time.sleep(2)  # wait for the server to start
    webbrowser.open(BASE_URL)


def main():
    print("Starting CPU_Usage Backend...")
    print()
    print("  Quick Links:")
    print(f"    UI:          {BASE_URL}")
    print(f"    Health:      {BASE_URL}/health")
    print(f"    Predict:     {BASE_URL}/predict  (POST)")
    print(f"    Model Info:  {BASE_URL}/model-info")
    print(f"    API Docs:    {BASE_URL}/docs")
    print()

    # Open browser automatically
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch the FastAPI server via uvicorn
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.backend:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--reload",
        ]
    )


if __name__ == "__main__":
    main()
