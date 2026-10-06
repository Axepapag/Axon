"""Run the local combined frontend/API without changing the shared tunnel."""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Axon Lab local service")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run("lab.backend.app:create_app", factory=True, host="127.0.0.1",
                port=args.port, workers=1)


if __name__ == "__main__":
    main()
