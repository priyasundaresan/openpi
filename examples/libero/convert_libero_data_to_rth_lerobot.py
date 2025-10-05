"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import numpy as np
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import imageio

REPO_NAME = "rth/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def visualize(steps, motion_labels, task_instruction=None, stride=5, filename=None):
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
        motion_label = motion_labels[t]
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

def chunk_motion_labels(delta_actions, grippers, chunk_size=50, move_thresh=1e-3, frac_thresh=0.25, tile=True):
    """
    Coarse motion labels with per-axis dominant direction and gripper *change detection*.
    If no gripper change in the window, no gripper label is added.
    """
    n = len(delta_actions)
    chunk_labels = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        window_actions = delta_actions[start:end]
        window_grippers = grippers[start:end]

        # --- Movement axes ---
        x_moves, y_moves, z_moves = [], [], []
        for d in window_actions:
            if abs(d[0]) > move_thresh:
                x_moves.append("forward" if d[0] > 0 else "backward")
            if abs(d[1]) > move_thresh:
                y_moves.append("right" if d[1] > 0 else "left")
            if abs(d[2]) > move_thresh:
                z_moves.append("up" if d[2] > 0 else "down")

        def dominant(moves, neutral="stay"):
            if not moves:
                return neutral
            counts = Counter(moves)
            top_label, count = counts.most_common(1)[0]
            if count / len(moves) >= frac_thresh:
                return top_label
            return neutral

        x_label = dominant(x_moves)
        y_label = dominant(y_moves)
        z_label = dominant(z_moves)

        # --- Gripper changes ---
        g_signs = np.sign(window_grippers)
        
        # prepend the last gripper state from previous chunk for continuity
        if start > 0:
            prev_state = np.sign(grippers[start-1])
            g_signs = np.insert(g_signs, 0, prev_state)
        
        g_changes = np.where(np.diff(g_signs) != 0)[0]
        
        g_label = None
        if len(g_changes) > 0:
            # take the last change in this chunk (ignoring the synthetic prepend index)
            last_idx = g_changes[-1]
            new_state = g_signs[last_idx + 1]  # state after change
            if new_state > 0:
                g_label = "close gripper"
            elif new_state < 0:
                g_label = "open gripper"


        # --- Combine ---
        parts = [p for p in [x_label, y_label, z_label, g_label] if p not in ("stay", None)]
        label = "move " + " ".join(parts) if parts else "stay"

        if tile:
            chunk_labels.extend([label] * (end - start))
        else:
            chunk_labels.append(label)

    return chunk_labels

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    print(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")

        for i, episode in enumerate(raw_dataset):

            steps = list(episode["steps"].as_numpy_iterator())
            delta_actions = [step["action"][:3] for step in steps]
            grippers = [step["action"][-1] for step in steps]

            motion_labels = chunk_motion_labels(
                np.array(delta_actions),
                np.array(grippers),
                chunk_size=10, 
                tile=True
            )

            task_instruction = steps[0]["language_instruction"].decode()
            #visualize(steps, motion_labels, task_instruction=task_instruction, stride=1, filename='%s.gif'%(str(i)))
            #visualize(steps, motion_labels, task_instruction=task_instruction, stride=1)
            for t, step in enumerate(steps):
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": motion_labels[t],
                    }
                )

            dataset.save_episode()

        print(f"Saved motion-centric dataset to {output_path}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)

