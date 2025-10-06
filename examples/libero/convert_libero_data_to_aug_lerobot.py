"""
Convert paraphrased Libero dataset to LeRobot format.

This version correctly tiles the paraphrased instructions across 10-step chunks
and truncates to match the number of steps in each episode.

Usage:
uv run convert_paraphrased_libero_to_lerobot.py --data_dir /path/to/libero/rlds --json_dir /path/to/paraphrased/json
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

import tensorflow_datasets as tfds
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import imageio

REPO_NAME = "aug/libero"
#RAW_DATASET_NAMES = [
#    "libero_10_no_noops",
#    "libero_goal_no_noops",
#    "libero_object_no_noops",
#    "libero_spatial_no_noops",
#]
RAW_DATASET_NAMES = [
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

CHUNK_SIZE = 10  # Tile size for paraphrased instructions

def visualize(steps, aug_labels, task_instruction=None, stride=5, filename=None):
    """
    Visualize an episode:
    - Left: End effector trajectory with current step highlighted (fixed limits, fixed view)
    - Right: Image frame
    - Task shown as multi-line centered title (top)
    - Motion shown as multi-line centered caption (bottom)
    - Optionally save to GIF if filename is provided
    """
    # extract delta actions and grippers
    delta_actions = np.array([s["action"][:3] for s in steps])
    grippers = np.array([s["action"][-1] for s in steps])

    # integrate deltas -> trajectory
    positions = np.cumsum(delta_actions, axis=0)
    positions = np.vstack([[0,0,0], positions])  # start at origin

    # global bounds
    x_min, x_max = positions[:,0].min(), positions[:,0].max()
    y_min, y_max = positions[:,1].min(), positions[:,1].max()
    z_min, z_max = positions[:,2].min(), positions[:,2].max()

    # setup fig
    fig = plt.figure(figsize=(10,6))
    ax_traj = fig.add_subplot(1,2,1, projection="3d")
    ax_img = fig.add_subplot(1,2,2)

    # format task instruction for multi-line (split roughly in half)
    if task_instruction:
        words = task_instruction.split()
        mid = len(words) // 2
        task_instruction = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        fig.suptitle(task_instruction, fontsize=16, y=0.97, ha="center")

    # placeholder motion caption at bottom
    motion_text_obj = fig.text(0.5, 0.02, "", ha="center", va="bottom",
                               fontsize=16)

    # collect frames if saving
    frames = []

    for t in range(0, len(steps), stride):
        ax_traj.cla()
        ax_img.cla()

        # plot trajectory up to t
        ax_traj.plot(positions[:t+1,0], positions[:t+1,1], positions[:t+1,2], "r-")

        # current gripper state (black = closed, white = open)
        color = "black" if grippers[t] > 0 else "white"
        ax_traj.scatter(
            positions[t,0], positions[t,1], positions[t,2],
            c=color, s=80, edgecolor="k"
        )

        # fixed limits and view
        ax_traj.set_xlim(x_max, x_min)
        ax_traj.set_ylim(y_min, y_max)
        ax_traj.set_zlim(z_min, z_max)
        ax_traj.view_init(elev=20, azim=-180)

        # show image
        img = steps[t]["observation"]["image"]
        ax_img.imshow(img)
        ax_img.axis("off")

        # update motion caption (split if long)
        motion_label = aug_labels[t]
        words = motion_label.split()
        if len(words) > 4:
            mid = len(words)//2
            motion_label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        motion_text_obj.set_text(motion_label)

        # capture frame if saving
        if filename:
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 4)   # ARGB
            
            # reorder ARGB -> RGBA
            buf = buf[:, :, [1, 2, 3, 0]]
            
            # drop alpha if you just want RGB
            frame = buf[:, :, :3]
            frames.append(frame)
        else:
            plt.pause(0.1)

    # save gif if requested
    if filename:
        imageio.mimsave(filename, frames, fps=20)
        print(f"Saved visualization to {filename}")
    else:
        plt.show()



def load_paraphrased_json(json_dir: str) -> Dict[str, dict]:
    """
    Load all paraphrased JSON files from the directory.
    Returns:
        dict mapping dataset_name -> JSON content
    """
    json_dir = Path(json_dir)
    paraphrased_data = {}
    for json_file in json_dir.glob("paraphrased_instructions_*.json"):
        dataset_name = json_file.stem.replace("paraphrased_instructions_", "")
        with open(json_file, "r") as f:
            paraphrased_data[dataset_name] = json.load(f)
    return paraphrased_data


def chunk_and_tile_labels(n_steps: int, ep_json: dict, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Tile paraphrased instructions across chunks of size chunk_size,
    then truncate to exactly n_steps.
    """
    motion_labels = []

    # Number of chunks in this episode
    num_chunks = (n_steps + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        # JSON key for chunk
        chunk_key = f"step_{chunk_idx}"  # Each chunk in JSON is "step_0", "step_1", etc.
        chunk_json = ep_json.get(chunk_key, {})
        label = chunk_json.get("paraphrased", "")

        # Tile label across chunk_size steps
        motion_labels.extend([label] * chunk_size)

    # Truncate to match exactly n_steps
    motion_labels = motion_labels[:n_steps]
    return motion_labels


def main(data_dir: str, json_dir: str, *, push_to_hub: bool = False):
    data_dir = Path(data_dir)
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    paraphrased_data = load_paraphrased_json(json_dir)

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Process each raw dataset
    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"Processing dataset: {raw_dataset_name}")
        raw_dataset = tfds.load(raw_dataset_name, data_dir=str(data_dir), split="train")

        dataset_json = paraphrased_data.get(raw_dataset_name, {})
        if not dataset_json:
            print(f"Warning: No paraphrased JSON found for {raw_dataset_name}, skipping.")
            continue

        for ep_idx, episode in enumerate(raw_dataset):
            steps = list(episode["steps"].as_numpy_iterator())
            n_steps = len(steps)

            episode_key = f"episode_{ep_idx}"
            ep_json = dataset_json[raw_dataset_name][episode_key]

            # Get motion labels tiled and truncated per 10-step chunk
            augmented_instructions = chunk_and_tile_labels(n_steps, ep_json, chunk_size=CHUNK_SIZE)

            task_instruction = steps[0]["language_instruction"].decode()
            visualize(steps, augmented_instructions, task_instruction=task_instruction, stride=1)
            for t, step in enumerate(steps):
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": augmented_instructions[t],
                    }
                )

            dataset.save_episode()

        print(f"Saved paraphrased dataset for {raw_dataset_name} to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "paraphrased"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
