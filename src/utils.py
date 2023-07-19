import datetime
import logging
import os
import random

import numpy as np
import torch
from tqdm import tqdm


def seed_torch(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mrr_at_k(y_true, y_pred, k: int = 20):
    """
    MRR（Mean Reciprocal Rank）@k を計算する関数。
    Args:
        y_true : 各クエリに対する正解ラベルのリスト（正解アイテムのインデックス）。
        y_pred : モデルの予測スコア。各クエリに対する全アイテムの予測スコアのリスト。
    Returns:
        MRR
    """
    mrr = 0.0
    for i in tqdm(range(len(y_true)), desc="[mrr]"):
        # Sort predictions and get the top k
        top_k_indices = np.argsort(y_pred[i])[::-1][:k]
        for rank, index in enumerate(top_k_indices):
            if index == y_true[i]:
                mrr += 1.0 / (rank + 1)
                break
    mrr /= len(y_true)
    return mrr


def mrr_at_k_by_searched_index(queries, cites, searched_index):
    """
    MRR（Mean Reciprocal Rank）@k を計算する関数(各queryに対する類似citeインデックスを使用する場合)
    Args:
        queries: クエリラベル (クエリ画像数, )
        cites: 参照ラベル (参照画像数, )
        searched_index: 各クエリ画像と類似した参照画像のトップk個を並べたインデックス
    Notes:
        クエリ画像: validationに使用する画像
        参照画像: trainに使用する画像
    """
    mrr = 0.0
    for i in tqdm(range(len(queries)), desc="[mrr]"):
        selected_cites = cites[searched_index[i]]
        for rank, index in enumerate(selected_cites):
            if index == queries[i]:
                mrr += 1.0 / (rank + 1)
                break
    mrr /= len(queries)
    return mrr


class Logger:
    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, "Experiment.log"))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)
