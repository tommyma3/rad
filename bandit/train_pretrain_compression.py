if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit.training import main

if __name__ == "__main__":
    main(default_model="rad", pretrain=True)
