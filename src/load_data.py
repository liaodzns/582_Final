import os

import pandas as pd
from scipy import sparse


# find the data folder relative to this file, so it works from notebooks/ too
SRC_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")


def load_split(split):
    """Load one split ("train", "test1", or "test2").

    Returns cells, rna, protein, labels.
    labels is None for the test split.
    """
    split_folder = os.path.join(DATA_FOLDER, "splits", split)

    # load the cell metadata
    cells = pd.read_csv(os.path.join(split_folder, "cells.csv"))

    # load the sparse RNA matrix (rows follow cells.csv)
    rna = sparse.load_npz(os.path.join(split_folder, "rna_log_normalized_hvg.npz"))

    # load the protein matrix (row index is cell_id)
    protein = pd.read_csv(os.path.join(split_folder, "protein_clr_normalized.csv"), index_col=0)

    # load the labels, only the train split has them
    labels_path = os.path.join(split_folder, "labels.csv")
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path)
    else:
        labels = None

    # check that every file has the same cells in the same order
    check_alignment(cells, rna, protein, labels)

    return cells, rna, protein, labels


def check_alignment(cells, rna, protein, labels):
    """Raise an error if RNA, protein, or labels are not in cells.csv order."""
    cell_ids = cells["cell_id"].values

    # the RNA matrix has no ids, so we can only check the row count
    if rna.shape[0] != len(cell_ids):
        raise ValueError("RNA row count does not match cells.csv")

    # the protein rows should be in the same cell_id order
    if not (protein.index.values == cell_ids).all():
        raise ValueError("protein rows are not in cells.csv order")

    # the labels should be in the same cell_id order
    if labels is not None:
        if not (labels["cell_id"].values == cell_ids).all():
            raise ValueError("labels are not in cells.csv order")


def load_sample_submission():
    """Load the example Kaggle submission file."""
    return pd.read_csv(os.path.join(DATA_FOLDER, "sample_submission.csv"))
