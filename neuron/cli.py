import os
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="NEURON CLI - Cognitive Memory for AI Agents")
    subparsers = parser.add_subparsers(dest="command")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new NEURON project")
    
    args = parser.parse_args()

    if args.command == "init":
        print("Initializing NEURON project...")
        # Create .env template
        with open(".env.example", "w") as f:
            f.write("GROQ_API_KEY=your_key\nLLM_PROVIDER=groq\nDATABASE_URL=postgresql://postgres:postgres@localhost:5432/neuron\n")
        
        print("Done! Created .env.example.")
        print("Next steps:")
        print("1. Configure your .env file")
        print("2. Run 'docker-compose up -d' to start the memory infrastructure.")

if __name__ == "__main__":
    main()
