import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.const import (
    BASE_CITE_CSV,
    BASE_TEST_CSV,
    BASE_TRAIN_CSV,
    CITE_IMAGES_PATH,
    QUERY_IMAGES_PATH,
    SAMPLE_SUBMMISION_CSV,
)


def add_fold_idx(train_df, n_fold, seed, target_columns):
    fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    train_df["fold"] = -1
    for i_fold, (train_index, val_index) in enumerate(
        fold.split(train_df, train_df[target_columns])
    ):
        train_df.iloc[val_index, train_df.columns.get_loc("fold")] = int(i_fold)
    train_df["fold"] = train_df["fold"].astype(int)
    return train_df


def load_trainable_df(
    train_csv_path=BASE_TRAIN_CSV,
    test_csv_path=BASE_TEST_CSV,
    cite_csv_path=BASE_CITE_CSV,
    sample_submission_csv_path=SAMPLE_SUBMMISION_CSV,
    seed=0,
    debug=False,
):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    cite_df = pd.read_csv(cite_csv_path).rename(
        columns={"cite_gid": "gid", "cite_filename": "filename"}
    )
    sample_submission_df = pd.read_csv(sample_submission_csv_path)
    if debug:
        train_df = train_df.sample(1000, random_state=seed).reset_index(drop=True)
        cite_df = cite_df.sample(100, random_state=seed).reset_index(drop=True)
    train_df = make_train_label(train_df)
    train_df = make_filepath(train_df)
    test_df = make_filepath(test_df)
    cite_df = make_filepath(cite_df)
    return (train_df, test_df, cite_df, sample_submission_df)


def make_train_label(train_df) -> pd.DataFrame:
    le = LabelEncoder()
    train_df["brand_id_label"] = le.fit_transform(train_df["brand_id"])
    train_df["meigara_label"] = le.fit_transform(train_df["meigara"])

    return train_df


def make_filepath(
    input_df, query_images_path=QUERY_IMAGES_PATH, cite_images_path=CITE_IMAGES_PATH
):
    def join_path(dirpath, filename):
        return (dirpath / filename).as_posix()

    output_df = input_df.assign(
        filepath=input_df["filename"].apply(
            lambda x: join_path(query_images_path, x)
            if str(x)[0] == "2"
            else join_path(cite_images_path, x)
        )
    )
    return output_df


def make_label_dict(train_df):
    mlabel2meigara = (
        train_df[["meigara", "meigara_label"]]
        .set_index("meigara_label")
        .to_dict()["meigara"]
    )
    blabel2bland = (
        train_df[["brand_id", "brand_id_label"]]
        .set_index("brand_id_label")
        .to_dict()["brand_id"]
    )
    return mlabel2meigara, blabel2bland
