import json
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
import tensorflow_datasets as tfds

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "modified_libero_rlds"  # <-- Set your dataset path here
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda"
CHUNK_SIZE = 10       # steps per motion chunk
BATCH_SIZE = 16       # process 16 chunks at a time
OUTPUT_JSON = "paraphrased_instructions.json"

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# ----------------------------
# Load model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()

# ----------------------------
# Motion chunking
# ----------------------------
def chunk_motion_labels(delta_actions, grippers, chunk_size=10, move_thresh=1e-3, frac_thresh=0.25):
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
    return f"""
You are a human instructing a Panda robot with a parallel-jaw gripper.
Your goal is to rephrase the following given instruction differently.
Be casual and creative, consider phrasings like "slide", "move your arm", "bring your gripper", etc.
Output strictly as JSON.

Example Instruction: move right down close gripper
Example Rephrased 1: {{"rephrased": "go a little to the right then down and close your gripper"}}
Example Rephrased 2: {{"rephrased": "slide to the right down and close the gripper"}}
Example Rephrased 3: {{"rephrased": "go to the right and down and close"}}
---

Instruction: {instruction}
Rephrased: 
"""

# ----------------------------
# JSON parsing
# ----------------------------
def parse_first_json(text: str) -> str:
    """
    Extract 'command' from model output with ```json fences.
    Returns lowercase string. Falls back to stripped text if JSON parsing fails or is empty.
    """
    text = text.strip()
    text = text.split('```json')[1].split('```')[0]
    text = text.strip()
    data = json.loads(text)
    return data.get("rephrased", "").strip().lower().replace('.', '').replace(',', '')

# ----------------------------
# Main loop
# ----------------------------
def main():
    data_dir = Path(DATA_DIR)
    augmented_data = {}

    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=str(data_dir), split="train")
        print(f"Processing dataset {raw_dataset_name} with {len(raw_dataset)} episodes")

        for ep_idx, episode in enumerate(tqdm(raw_dataset, desc=f"Dataset {raw_dataset_name}")):
            steps = list(episode["steps"].as_numpy_iterator())
            delta_actions = [step["action"][:3] for step in steps]
            grippers = [step["action"][-1] for step in steps]

            motion_labels = chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE)

            paraphrased_labels = []

            # ----------------------------
            # Batch inference
            # ----------------------------
            paraphrased_labels = []
            
            for i in range(0, len(motion_labels), BATCH_SIZE):
                batch = motion_labels[i:i+BATCH_SIZE]
                prompts = [make_prompt(label) for label in batch]
            
                # Tokenize batch like in your test script
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
                # Generate paraphrases
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=0.7)
            
                # Decode outputs exactly like test script
                paraphrased_labels.extend([
                    parse_first_json(tokenizer.decode(o, skip_special_tokens=True))
                    for o in outputs
                ])


            augmented_data[f"{raw_dataset_name}_episode_{ep_idx}"] = {
                "original": motion_labels,
                "paraphrased": paraphrased_labels
            }

            # Print a few spaced-out examples
            for j in range(0, len(motion_labels)):
                print(f"Original: {motion_labels[j]} -> Paraphrased: {paraphrased_labels[j]}")
            print("-" * 50)

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(augmented_data, f, indent=2)

    print(f"Augmented dataset saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

