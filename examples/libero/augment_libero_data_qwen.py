import argparse
import json
import random
import signal
import sys
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline
import tensorflow_datasets as tfds
import imageio
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "modified_libero_rlds"
DEFAULT_OUTPUT_DIR = Path("output_qwen_checkpoints")
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CHUNK_SIZE = 10
BATCH_SIZE = 64
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

PARAPHRASED_JSONS = {
    "libero_10_no_noops": "paraphrased_instructions_libero_10_no_noops.json",
    "libero_goal_no_noops": "paraphrased_instructions_libero_goal_no_noops.json",
    "libero_object_no_noops": "paraphrased_instructions_libero_object_no_noops.json",
    "libero_spatial_no_noops": "paraphrased_instructions_libero_spatial_no_noops.json",
}

# ----------------------------
# Global state for signal handling
# ----------------------------
terminate_requested = False
last_ckpt_path = None
last_ckpt_data = None

def handle_termination(signum, frame):
    """Handle SIGTERM and SIGINT by saving current checkpoint."""
    global terminate_requested, last_ckpt_data, last_ckpt_path
    print(f"\nReceived signal {signum} — saving checkpoint before exit...")
    terminate_requested = True
    if last_ckpt_data is not None and last_ckpt_path is not None:
        try:
            save_checkpoint(last_ckpt_data, last_ckpt_path)
            print(f"Checkpoint saved at {last_ckpt_path}")
        except Exception as e:
            print(f"Failed to save checkpoint on signal: {e}")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_termination)
signal.signal(signal.SIGINT, handle_termination)

# ----------------------------
# Motion chunking
# ----------------------------
def chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE, move_thresh=1e-3, frac_thresh=0.25):
    n = len(delta_actions)
    chunk_labels = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        window_actions = delta_actions[start:end]
        window_grippers = grippers[start:end]

        x_moves, y_moves, z_moves = [], [], []
        for d in window_actions:
            if abs(d[0]) > move_thresh:
                x_moves.append("forward" if d[0] > 0 else "backward")
            if abs(d[1]) > move_thresh:
                y_moves.append("right" if d[1] > 0 else "left")
            if abs(d[2]) > move_thresh:
                z_moves.append("up" if d[2] > 0 else "down")

        def dominant(moves, neutral="stay"):
            if not moves: return neutral
            counts = Counter(moves)
            top_label, count = counts.most_common(1)[0]
            return top_label if count / len(moves) >= frac_thresh else neutral

        x_label = dominant(x_moves)
        y_label = dominant(y_moves)
        z_label = dominant(z_moves)

        g_signs = list(map(lambda x: 1 if x > 0 else -1 if x < 0 else 0, window_grippers))
        if start > 0:
            g_signs.insert(0, 1 if grippers[start - 1] > 0 else -1)
        g_changes = [i for i in range(len(g_signs)-1) if g_signs[i+1] != g_signs[i]]
        g_label = None
        if g_changes:
            last_idx = g_changes[-1]
            new_state = g_signs[last_idx + 1]
            g_label = "close gripper" if new_state > 0 else "open gripper"

        parts = [p for p in [x_label, y_label, z_label, g_label] if p not in ("stay", None)]
        label = "move " + " ".join(parts) if parts else "stay"
        chunk_labels.append(label)
    return chunk_labels

def make_prompt(instruction: str) -> str:
    style = random.choice([
        "as a casual person speaking naturally to a home robot",
        "as someone giving quick, efficient task commands",
        "as a curious user testing the robot's understanding",
        "as if giving polite, conversational instructions",
        "say it like you would to a helpful robot at home",
        "keep it simple and casual",
        "give quick, direct instructions",
        "describe it like you’re telling someone in the room",
        "use everyday words you’d naturally say",
        "say it as clearly as possible",
        "make it short and to the point",
        "use plain, friendly language"
        "as someone a bit skeptical if the robot will understand",
        "as someone talking to a clumsy robot that often messes up",
        "as someone teaching a robot patiently for the first time",
        "as someone in a hurry giving quick orders",
        "as someone slightly frustrated or impatient",
        "as someone trying to sound encouraging to the robot",
        "as someone speaking like they’re giving directions to a friend",
        "as someone treating the robot like a pet they’re fond of",
        "as a person who’s multitasking and distracted",
        "as a person testing if the robot can handle vague phrasing",
        "as a person who over-explains things to make sure it listens",
        "as someone who gives clipped, military-style commands",
        "as someone pretending the robot is a coworker",
        "as someone playfully joking with the robot",
        "as someone who doubts the robot’s abilities and over-clarifies",
        "as someone using slang or informal shorthand",
        "as someone who’s being polite but firm",
        "as someone who’s cautious and hesitant about breaking something",
        "as someone impressed by the robot and talking enthusiastically",
        "as someone slightly sarcastic about the robot following orders",
        "as someone used to giving short, efficient directions",
        "as someone issuing calm, steady instructions",
        "as someone giving clear steps during a task",
        "as someone who trusts the robot and speaks casually",
        "as someone who gives short, helpful reminders",
        "as someone trying to sound natural but firm",
        "as someone summarizing what to do quickly",
        "as someone who’s focused and wants minimal words",
        "as someone who speaks efficiently but not abruptly",
        "as someone directing a coworker briefly and clearly",
        "as someone rephrasing the instruction with natural synonyms for the action verbs",
        "as someone saying the same thing but swapping verbs like pick up → grab or put → place",
        "as someone using everyday verbs that mean the same thing for handling or moving objects",
        "as someone restating it with slightly different action words but the same intent",
        "as someone giving identical instructions using familiar hand-action verbs",
        "as someone who changes the verbs to more natural alternatives",
        "as someone who swaps out stiff action words for casual ones",
        "as someone who tweaks the verbs to sound like everyday speech",
        "as someone who replaces robotic commands with friendly verbs",
        "as someone who rewords the actions using playful, human verbs",
        "as someone who modifies the verbs to make instructions more lively",
        "as someone who alternates the verbs while keeping the same meaning",
        "as someone who adjusts the action words to sound like a person talking",
        "as someone who substitutes dry verbs with more expressive ones",
        "as someone who remixes the verbs to give a human touch to the instructions"
        "as someone replacing technical verbs with natural ones people actually say",
        "as someone keeping the meaning identical but using a more natural verb choice",
        "as someone using simple synonyms for manipulation actions like lift, set, or hold",
        "as someone who changes only the verbs to sound more human while keeping the task the same",
        "as someone paraphrasing it using alternative but equivalent verbs for the same motion",
        "as someone who calls objects different things if objects are mentioned",
        "as someone who sometimes refers to objects with slightly different names if objects are mentioned",
        "as someone who uses alternative common terms for objects if objects are mentioned",
        "as someone who swaps object names with natural synonyms if objects are mentioned",
        "as someone rewording object references in a casual, human way if objects are mentioned",
        "as someone who varies how objects are named if objects are mentioned",
        "as someone giving instructions and occasionally changing object wording naturally if objects are mentioned",
        "as someone who describes objects differently if objects are mentioned"
    ])
    return f"""
You are rewriting action commands for a Panda parallel-jaw gripper robot into natural human language.
Your goal is to show how people with different communication styles might speak to a robot.
Rephrase each instruction to sound realistic and human — not robotic or repetitive.
Vary your verbs, phrasing, sentence structure, and tone.
Output only valid JSON in this format: {{"rephrased": "<your version>"}}.

Examples:
Instruction: move left down close gripper
Rephrased: {{"rephrased": "go a bit left and lower down, then close the gripper"}}

Instruction: put the white and yellow mug to the right of the plate
Rephrased: {{"rephrased": "grab the white and yellow mug and place it by the plate"}}

Instruction: stay
Rephrased: {{"rephrased": "don't move"}}

---
Now, write a **concise** rephrased instruction {style}.
Instruction: {instruction}
Rephrased:
"""

def parse_first_json(text: str) -> str:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        data = json.loads(text)
        return data.get("rephrased", "").strip().lower().replace('.', '').replace(',', '')
    except Exception:
        return text.strip().lower().replace('.', '').replace(',', '')

# ----------------------------
# GIF / Trajectory visualization
# ----------------------------
def visualize(steps, labels, task_instruction, filename):
    delta_actions = np.array([s["action"][:3] for s in steps])
    grippers = np.array([s["action"][-1] for s in steps])
    positions = np.cumsum(delta_actions, axis=0)
    positions = np.vstack([[0,0,0], positions])
    x_min, x_max = positions[:,0].min(), positions[:,0].max()
    y_min, y_max = positions[:,1].min(), positions[:,1].max()
    z_min, z_max = positions[:,2].min(), positions[:,2].max()
    fig = plt.figure(figsize=(10,6))
    ax_traj = fig.add_subplot(1,2,1, projection="3d")
    ax_img = fig.add_subplot(1,2,2)
    fig.suptitle(task_instruction, fontsize=16, y=0.97, ha="center")
    motion_text_obj = fig.text(0.5,0.02,"",ha="center",va="bottom",fontsize=16)
    frames_img = []
    for t in range(len(steps)):
        ax_traj.cla()
        ax_img.cla()
        ax_traj.plot(positions[:t+1,0],positions[:t+1,1],positions[:t+1,2],"r-")
        color="black" if grippers[t]>0 else "white"
        ax_traj.scatter(positions[t,0],positions[t,1],positions[t,2],c=color,s=80,edgecolor="k")
        ax_traj.set_xlim(x_min,x_max)
        ax_traj.set_ylim(y_min,y_max)
        ax_traj.set_zlim(z_min,z_max)
        ax_traj.view_init(elev=20,azim=-180)
        img = Image.fromarray(steps[t]["observation"]["image"])
        ax_img.imshow(img)
        ax_img.axis("off")
        label = labels[t]
        words = label.split()
        if len(words)>4:
            mid = len(words)//2
            label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        motion_text_obj.set_text(label)
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(),dtype=np.uint8).reshape(h,w,4)
        buf = buf[:,:, [1,2,3,0]]
        frames_img.append(buf[:,:,:3])
    imageio.mimsave(filename, frames_img, fps=10, loop=0)
    print(f"Saved GIF: {filename}")

def swap_directions(instruction):
    instruction = instruction.replace('left', '__TEMP_LEFT__').replace('right', 'left').replace('__TEMP_LEFT__', 'right')
    instruction = instruction.replace('forward', '__TEMP_FORWARD__').replace('backward', 'forward').replace('__TEMP_FORWARD__', 'backward')
    return instruction

# ----------------------------
# Checkpoint helpers
# ----------------------------
def load_checkpoint(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        print(f"Warning: checkpoint at {path} is corrupt, restarting from scratch.")
        return {}

def save_checkpoint(data: dict, path: Path):
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    tmp_path.replace(path)

# ----------------------------
# MAIN
# ----------------------------
def main():
    global last_ckpt_data, last_ckpt_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    dataset_name = args.dataset
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = PARAPHRASED_JSONS.get(dataset_name)
    if json_file is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    in_path = Path("output") / json_file
    print(f"Loading paraphrased JSON: {in_path}")
    data = json.load(open(in_path))

    print(f"Loading dataset: {dataset_name}")
    raw_dataset = tfds.load(dataset_name, data_dir=DATA_DIR, split="train")

    ckpt_path = output_dir / f"paraphrased_instructions_{dataset_name}.json"
    last_ckpt_path = ckpt_path
    augmented_data = load_checkpoint(ckpt_path)
    last_ckpt_data = augmented_data

    print(f"Initializing Qwen model on device {device} ...")
    pipe = pipeline(
        task="text2text-generation",
        model=MODEL_ID,
        device=device,
        torch_dtype=torch.bfloat16
    )

    for ep_idx, episode in enumerate(tqdm(raw_dataset, desc=f"Dataset {dataset_name}")):
        if terminate_requested:
            print("Termination requested, stopping cleanly...")
            break

        ep_key = f"episode_{ep_idx}"
        if ep_key in augmented_data:
            continue  # already processed

        steps = list(episode["steps"].as_numpy_iterator())
        delta_actions = [s["action"][:3] for s in steps]
        grippers = [s["action"][-1] for s in steps]
        motion_labels = chunk_motion_labels(delta_actions, grippers)
        task_instruction = steps[0]["language_instruction"].decode()

        task_instruction = swap_directions(task_instruction)
        motion_labels = [swap_directions(l) for l in motion_labels]

        step_labels = []
        sources = ["original", "motion", "gemini"]
        weights = [0.2, 0.4, 0.4]
        for i in range(len(motion_labels)):
            source = random.choices(sources, weights=weights, k=1)[0]
            if source == "original":
                label = task_instruction
            elif source == "motion":
                label = motion_labels[i]
            else:
                label = data[f"episode_{ep_idx}"][f"step_{i}"]["paraphrased"]
            step_labels.append(label)

        paraphrased_labels = []
        for i in range(0, len(step_labels), BATCH_SIZE):
            if terminate_requested:
                break
            batch = step_labels[i:i+BATCH_SIZE]
            prompts = [make_prompt(lbl) for lbl in batch]
            results = pipe(prompts, max_new_tokens=64, temperature=1.2, top_p=0.9)
            paraphrased_labels.extend([parse_first_json(r["generated_text"]) for r in results])

        augmented_data[ep_key] = {
            f"step_{i}": {"original": step_labels[i], "paraphrased": paraphrased_labels[i]}
            for i in range(len(step_labels))
        }

        last_ckpt_data = augmented_data
        save_checkpoint(augmented_data, ckpt_path)
        print(f"Saved checkpoint after episode {ep_idx}")

    print(f"Finished or stopped for {dataset_name}")
    print(f"Final output: {ckpt_path}")

if __name__ == "__main__":
    main()

