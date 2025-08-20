import os
import pandas as pd


def load_dataframe(dataframe_path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(dataframe_path)
    match ext:
        case ".csv":
            return pd.read_csv(dataframe_path, index_col=0)
        case ".h5" | ".hdf5":
            return pd.read_hdf(dataframe_path)
        case ".feather":
            return pd.read_feather(dataframe_path)
