import numpy as np
import subprocess
import sys


def main():
    print("Starting CPU_Usage Backend...")
    # Launch the FastAPI server via uvicorn
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.backend:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--reload",
        ]
    )


if __name__ == "__main__":
    main()
