import os
import json
import random
from collections import Counter
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont

import torch
from tqdm import tqdm
import tensorflow_datasets as tfds
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

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
DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 10

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

if args.dataset:
    RAW_DATASET_NAMES = [args.dataset]

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load Qwen2.5-VL
# ----------------------------
#model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen3-VL-30B-A3B-Instruct",
#    dtype=torch.bfloat16,
#    attn_implementation="flash_attention_2",
#    device_map="auto",
#)

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

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

        g_signs = [1 if g > 0 else -1 if g < 0 else 0 for g in window_grippers]
        if start > 0:
            g_signs.insert(0, 1 if grippers[start - 1] > 0 else -1)
        g_changes = [i for i in range(len(g_signs) - 1) if g_signs[i + 1] != g_signs[i]]
        g_label = None
        if g_changes:
            last_idx = g_changes[-1]
            new_state = g_signs[last_idx + 1]
            g_label = "close gripper" if new_state > 0 else "open gripper"

        parts = [p for p in [x_label, y_label, z_label, g_label] if p not in ("stay", None)]
        label = "move " + " ".join(parts) if parts else "stay"
        chunk_labels.append(label)

    return chunk_labels

# ----------------------------
# Multimodal instruction generation
# ----------------------------
def swap_directions(instruction):
    # Create temporary placeholders to avoid conflict while replacing
    instruction = instruction.replace('left', '__TEMP_LEFT__')
    instruction = instruction.replace('right', 'left')
    instruction = instruction.replace('__TEMP_LEFT__', 'right')

    instruction = instruction.replace('forward', '__TEMP_FORWARD__')
    instruction = instruction.replace('backward', 'forward')
    instruction = instruction.replace('__TEMP_FORWARD__', 'backward')

    return instruction


def generate_intermediate_instruction(start_img, end_img, task_instruction, motion_label):
    task_instruction = swap_directions(task_instruction)
    motion_label = swap_directions(motion_label)
    prompt = f"""
    Goal of the Robot: {task_instruction}.
    Current motion of the robot: {motion_label}.
    Your Task: Describe what subtask the robot is doing right now given its overall goal and the current motion.
    Input: The first image is the previous state of the robot performing the task, and the second image is the current state.
    Output format:
    ```json
    {{
        "all stages": # given the goal of the robot and first image, describe all subtasks that the robot needs to do to complete its goal,
        "reasoning": # explain how you determined which stage the robot is in and what it is doing,
        "subtask": # a short sentence describing the robot’s current subtask
    }}
    ```
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "image": start_img,
                },
                {
                    "type": "image",
                    "image": end_img,
                },
            ],
        }
    ]


    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids.to('cuda'), generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

# ----------------------------
# Visualization
# ----------------------------
def save_visualization(start_img, end_img, task_instruction, motion_label, generated_label, dataset_name, ep_idx, chunk_idx):
    start_img_rgb = start_img.convert("RGB")
    end_img_rgb = end_img.convert("RGB")
    width, height = start_img_rgb.width, start_img_rgb.height
    canvas = Image.new("RGB", (width * 2, height + 150), color=(255, 255, 255))
    canvas.paste(start_img_rgb, (0, 0))
    canvas.paste(end_img_rgb, (width, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_lines = [
        f"Task: {task_instruction}",
        f"Low-level motions: {motion_label}",
        f"Generated: {generated_label}"
    ]
    for i, line in enumerate(text_lines):
        draw.text((10, height + i * 20), line, fill="black", font=font)

    vis_path = OUTPUT_DIR / f"{dataset_name}_episode{ep_idx}_{str(chunk_idx).zfill(3)}.png"
    canvas.save(vis_path)

# ----------------------------
# Main augmentation loop
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

            paraphrased_labels = []

            # Generate multimodal instructions per chunk
            for i in range(len(motion_labels)):
                start_idx = i * CHUNK_SIZE
                end_idx = min((i + 2) * CHUNK_SIZE - 1, len(steps) - 1)

                start_img_array = steps[start_idx]["observation"]["image"]
                end_img_array = steps[end_idx]["observation"]["image"]
                start_img = Image.fromarray(start_img_array)
                end_img = Image.fromarray(end_img_array)

                paraphrased_label = generate_intermediate_instruction(
                    start_img, end_img, task_instruction, motion_labels[i],
                )
                paraphrased_labels.append(paraphrased_label)

                # Save visualization
                save_visualization(
                    start_img, end_img,
                    task_instruction, motion_labels[i],
                    paraphrased_label,
                    raw_dataset_name, ep_idx, i
                )

            # Save per-step JSON
            episode_dict = {}
            for step_idx, (orig, para) in enumerate(zip(motion_labels, paraphrased_labels)):
                episode_dict[f"step_{step_idx}"] = {"original": orig, "paraphrased": para}

            augmented_data[raw_dataset_name][f"episode_{ep_idx}"] = episode_dict

            # ----------------------------
            # Print a few spaced-out examples
            # ----------------------------
            for j in range(0, len(motion_labels)):
                print(f"Original: {motion_labels[j]}, {task_instruction}\n-> Paraphrased: {paraphrased_labels[j]}")
                print("-" * 50)

        # Save dataset JSON
        output_json = f"paraphrased_instructions_{raw_dataset_name}.json"
        with open(output_json, "w") as f:
            json.dump(augmented_data, f, indent=2)
        print(f"Augmented dataset saved to {output_json}")

if __name__ == "__main__":
    main()

