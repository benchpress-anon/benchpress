"""
Entry point for running the pipeline as a module.

Usage:
    python -m automation --video input.mp4 --prompt "dog" --edit remove --out runs/run1
"""

from pipeline import main

if __name__ == "__main__":
    main()
