"""Dataset-level label mappings for the volleyball group-activity dataset."""

# 8 group-activity (scene-level) classes
GROUP_ACTIVITY_TO_IDX: dict[str, int] = {
    "l-pass": 0,
    "r-pass": 1,
    "l-spike": 2,
    "r_spike": 3,
    "l_set": 4,
    "r_set": 5,
    "l_winpoint": 6,
    "r_winpoint": 7,
}

IDX_TO_GROUP_ACTIVITY: dict[int, str] = {v: k for k, v in GROUP_ACTIVITY_TO_IDX.items()}

# 9 person-action (player-level) classes
PERSON_ACTION_TO_IDX: dict[str, int] = {
    "blocking": 0,
    "digging": 1,
    "falling": 2,
    "jumping": 3,
    "moving": 4,
    "setting": 5,
    "spiking": 6,
    "standing": 7,
    "waiting": 8,
}

IDX_TO_PERSON_ACTION: dict[int, str] = {v: k for k, v in PERSON_ACTION_TO_IDX.items()}

NUM_GROUP_ACTIVITIES = len(GROUP_ACTIVITY_TO_IDX)
NUM_PERSON_ACTIONS = len(PERSON_ACTION_TO_IDX)
