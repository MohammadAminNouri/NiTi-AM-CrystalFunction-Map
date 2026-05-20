from pathlib import Path

import pandas as pd


def _safe_read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def load_exact_process_rows() -> pd.DataFrame:
    return _safe_read_csv("data/exact_literature_process_rows.csv")


def load_paper_facts() -> pd.DataFrame:
    return _safe_read_csv("data/paper_level_numeric_facts.csv")


def load_demo_training() -> pd.DataFrame:
    return _safe_read_csv("data/demo_training_seed.csv")


def load_scenario_rules() -> pd.DataFrame:
    return _safe_read_csv("data/scenario_rules.csv")


def combined_training(include_demo: bool = False) -> pd.DataFrame:
    exact = load_exact_process_rows()

    if not include_demo:
        return exact

    demo = load_demo_training()

    return pd.concat([exact, demo], ignore_index=True, sort=False)
