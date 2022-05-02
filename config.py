from typing import Optional

GEN_DATA_SET_SIZE: int = 1500
GEN_DATA_SET_TEST_SIZE: int = 300

LR: float = 1e-5
DEVICE: str = "cuda"
LOAD_CHECKPOINT: bool = False
CHECKPOINT_PATH: Optional[str] = "./checkpoints/checkpoints_0"
SRC_DATA: str = "dataset/A"
DST_DATA: str = "dataset/B"
SRC_TEST: str = "dataset_test/A"
DST_TEST: str = "dataset_test/B"
BATCH_SIZE: int = 1
EPOCH: int = 100
LAMBDA_CYCLE: float = 10
LAMBDA_IDENTITY: float = 1
LAMBDA_TRUE: float = 10

RANDOM_FLIP: bool = True
TO_GRAY: bool = True

