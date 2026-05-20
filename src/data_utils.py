import pandas as pd
from pathlib import Path

def load_exact_process_rows():
    return pd.read_csv(Path("data/exact_literature_process_rows.csv"))

def load_paper_facts():
    return pd.read_csv(Path("data/paper_level_numeric_facts.csv"))

def load_demo_training():
    return pd.read_csv(Path("data/demo_training_seed.csv"))

def combined_training(include_demo=False):
    exact = load_exact_process_rows()
    # Keep rows with actual target columns only; exact process rows include many blanks.
    frames = [exact]
    if include_demo:
        frames.append(load_demo_training())
    return pd.concat(frames, ignore_index=True, sort=False)
