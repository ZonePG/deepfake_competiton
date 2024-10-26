import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load the data
data = pd.read_csv("train_label.csv")


def get_frame_count(video_id):
    path = f"/data/zonepg/datasets/dataset/frames_1s/train/{video_id}"
    return len(os.listdir(path))


data["id"] = data["id"].apply(lambda x: x.replace(".mp4", ""))
data["frame_count"] = data["id"].apply(get_frame_count)

# Set the number of splits
n_splits = 5

# Prepare Stratified K-Fold for 5 random splits
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform the splits
for split_num, (train_idx, val_idx) in enumerate(
    skf.split(data, data["label"]), start=1
):
    # Split the dataset
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    # Combine and format output as required
    # output_train = train_set[["id", "label"]].sort_values(by="id")
    # output_val = val_set[["id", "label"]].sort_values(by="id")

    # Save train and validation sets to files
    train_data.to_csv(f"set_{split_num}_train.txt", sep=" ", index=False, header=False)
    val_data.to_csv(f"set_{split_num}_val.txt", sep=" ", index=False, header=False)

    print(f"Split {split_num} completed.")
