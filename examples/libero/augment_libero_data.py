import json
import random
from collections import Counter
from pathlib import Path
import argparse

import torch
from tqdm import tqdm
from transformers import pipeline
import tensorflow_datasets as tfds

# ----------------------------
# Command-line args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="GPU device index")
parser.add_argument("--dataset", type=str, default=None, help="Optional: process only this dataset")
args = parser.parse_args()

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "modified_libero_rlds"  # <-- Set your dataset path here
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = args.device
CHUNK_SIZE = 10
BATCH_SIZE = 128

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# If user specified a dataset, override the list
if args.dataset:
    RAW_DATASET_NAMES = [args.dataset]

# ----------------------------
# Load model as a pipeline
# ----------------------------
pipe = pipeline(
    task="text2text-generation",
    model=MODEL_ID,
    device=DEVICE,
    dtype=torch.bfloat16
)

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
        g_signs = list(map(lambda x: 1 if x > 0 else -1 if x < 0 else 0, window_grippers))
        if start > 0:
            g_signs.insert(0, 1 if grippers[start - 1] > 0 else -1)
        g_changes = [i for i in range(len(g_signs)-1) if g_signs[i+1] != g_signs[i]]
        g_label = None
        if g_changes:
            last_idx = g_changes[-1]
            new_state = g_signs[last_idx + 1]
            g_label = "close gripper" if new_state > 0 else "open gripper"

        # --- Combine ---
        parts = [p for p in [x_label, y_label, z_label, g_label] if p not in ("stay", None)]
        label = "move " + " ".join(parts) if parts else "stay"
        chunk_labels.append(label)

    return chunk_labels

# ----------------------------
# Prompt for human-like paraphrasing
# ----------------------------
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
Now, write a rephrased instruction {style}.
Instruction: {instruction}
Rephrased:
"""

# ----------------------------
# JSON parsing
# ----------------------------
def parse_first_json(text: str) -> str:
    text = text.strip()
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        data = json.loads(text)
        return data.get("rephrased", "").strip().lower().replace('.', '').replace(',', '')
    except Exception:
        return text.strip().lower().replace('.', '',).replace(',', '')

# ----------------------------
# Main loop
# ----------------------------
def main():
    data_dir = Path(DATA_DIR)
    augmented_data = {}

    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=str(data_dir), split="train")
        print(f"Processing dataset {raw_dataset_name} with {len(raw_dataset)} episodes")

        augmented_data[raw_dataset_name] = {}

        for ep_idx, episode in enumerate(tqdm(raw_dataset, desc=f"Dataset {raw_dataset_name}")):
            steps = list(episode["steps"].as_numpy_iterator())
            delta_actions = [step["action"][:3] for step in steps]
            grippers = [step["action"][-1] for step in steps]

            task_instruction = steps[0]["language_instruction"].decode()
            motion_labels = chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE)

            # Optional: first label is always task instruction
            if motion_labels:
                motion_labels[0] = task_instruction

            paraphrased_labels = []

            # ----------------------------
            # Batch inference
            # ----------------------------
            for i in range(0, len(motion_labels), BATCH_SIZE):
                batch = motion_labels[i:i+BATCH_SIZE]
                prompts = [make_prompt(label) for label in batch]
                results = pipe(prompts, max_new_tokens=64, temperature=1.2, top_p=0.9)
                paraphrased_labels.extend([parse_first_json(r["generated_text"]) for r in results])

            # ----------------------------
            # Save per-step JSON
            # ----------------------------
            episode_dict = {}
            for step_idx, (orig, para) in enumerate(zip(motion_labels, paraphrased_labels)):
                episode_dict[f"step_{step_idx}"] = {"original": orig, "paraphrased": para}

            augmented_data[raw_dataset_name][f"episode_{ep_idx}"] = episode_dict

    # Save final JSON
    output_json = f"paraphrased_instructions_flat_gpu{DEVICE}.json"
    with open(output_json, "w") as f:
        json.dump(augmented_data, f, indent=2)
    print(f"Augmented dataset saved to {output_json}")

if __name__ == "__main__":
    main()

