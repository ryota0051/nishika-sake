from pathlib import Path

DATA_ROOT = Path("/work/data")

BASE_TRAIN_CSV = DATA_ROOT / "train.csv"
BASE_TEST_CSV = DATA_ROOT / "test.csv"
BASE_CITE_CSV = DATA_ROOT / "cite.csv"
SAMPLE_SUBMMISION_CSV = DATA_ROOT / "sample_submission.csv"

TRAIN_CSV_WITH_IMG_PATH = DATA_ROOT / "tain_with_img_path.csv"
CITE_CSV_WITH_IMG_PATH = DATA_ROOT / "cite_with_img_path.csv"
TEST_CSV_WITH_IMG_PATH = DATA_ROOT / "test_with_img_path.csv"

QUERY_IMAGES_PATH = DATA_ROOT / "query_images"
CITE_IMAGES_PATH = DATA_ROOT / "cite_images"

OUTPUT_ROOT = Path("/work/output")
