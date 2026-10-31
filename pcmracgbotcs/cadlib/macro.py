import numpy as np


ALL_COMMANDS = [
    "UNUSED0",
    "ARC",
    "CIRCLE",
    "EDGE",
    "WIRE",
    "FACE",
    "PRISM",
    "REVOL",
    "THRUSECTIONS",
    "CUT",
    "FUSE",
    "COMMON",
    "GROUP",
    "END_CHILD",
    "END_ROOT",
]

ARC_IDX = 1
CIRCLE_IDX = 2
EDGE_IDX = 3
WIRE_IDX = 4
FACE_IDX = 5
PRISM_IDX = 6
REVOL_IDX = 7
THRUSECTIONS_IDX = 8
CUT_IDX = 9
FUSE_IDX = 10
COMMON_IDX = 11
GROUP_IDX = 12
END_CHILD_IDX = 13
EOS_IDX = 14

PAD_VAL = -1
N_ARGS = 17
ARGS_DIM = 256
MAX_TOTAL_LEN = 100

# END_ROOT serves as both the sequence terminator and the trailing padding token.
EOS_VEC = np.asarray([EOS_IDX] + [PAD_VAL] * N_ARGS, dtype=np.int64)

# CMD_ARGS_MASK enables only the parameter slots used by each command.
CMD_ARGS_MASK = np.zeros((len(ALL_COMMANDS), N_ARGS), dtype=np.int64)


def use_args(command, count):
    CMD_ARGS_MASK[command, :count] = 1


use_args(ARC_IDX, 9)
use_args(CIRCLE_IDX, 10)
use_args(EDGE_IDX, 6)
use_args(FACE_IDX, 9)
use_args(PRISM_IDX, 7)
use_args(REVOL_IDX, 6)
