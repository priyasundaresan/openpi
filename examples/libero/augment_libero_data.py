import json
import random
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline
import tensorflow_datasets as tfds

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "modified_libero_rlds"  # <-- Set your dataset path here
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = 0  # GPU device index
CHUNK_SIZE = 10      
BATCH_SIZE = 128       
OUTPUT_JSON = "paraphrased_instructions.json"

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

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
    # Encourage different communication styles people might use with robots
    style = random.choice([
        "as a casual person speaking naturally to a home robot",
        "as a friendly collaborator in a shared workspace",
        "as someone giving quick, efficient task commands",
        "as a curious user testing the robot's understanding",
        "as if giving polite, conversational instructions",
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

Instruction: move forward up open gripper
Rephrased: {{"rephrased": "move forward and lift up a little, then open your gripper"}}

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
    """
    Extract 'rephrased' from model output JSON.
    Returns lowercase string. Falls back to stripped text if JSON parsing fails or is empty.
    """
    text = text.strip()
    try:
        # Remove code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        data = json.loads(text)
        return data.get("rephrased", "").strip().lower().replace('.', '').replace(',', '')
    except Exception:
        # fallback: return the raw text
        return text.strip().lower().replace('.', '').replace(',', '')

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
            for i in range(0, len(motion_labels), BATCH_SIZE):
                batch = motion_labels[i:i+BATCH_SIZE]
                prompts = [make_prompt(label) for label in batch]

                # Run pipeline
                results = pipe(prompts, max_new_tokens=64, temperature=1.2, top_p=0.9)

                # Parse JSON output
                paraphrased_labels.extend([parse_first_json(r["generated_text"]) for r in results])

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

