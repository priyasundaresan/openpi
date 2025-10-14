import os
import numpy as np
import imageio
import random
import matplotlib.pyplot as plt
import io
import json
import asyncio
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import tensorflow_datasets as tfds
from google.genai import Client, types

DATA_DIR = "modified_libero_rlds"
CHUNK_SIZE = 10
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
#RPM = 10000
RPM = 1000
RPS = RPM / 60
SEMAPHORE_LIMIT = 200
RAW_DATASET_NAME = "libero_10_no_noops"

def chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE, move_thresh=1e-3, frac_thresh=0.25):
    n = len(delta_actions)
    chunk_labels = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        window_actions = delta_actions[start:end]
        window_grippers = grippers[start:end]
        x_moves, y_moves, z_moves = [], [], []
        for d in window_actions:
            if abs(d[0]) > move_thresh: x_moves.append("forward" if d[0] > 0 else "backward")
            if abs(d[1]) > move_thresh: y_moves.append("right" if d[1] > 0 else "left")
            if abs(d[2]) > move_thresh: z_moves.append("up" if d[2] > 0 else "down")
        def dominant(moves, neutral="stay"):
            if not moves: return neutral
            counts = Counter(moves)
            top_label, count = counts.most_common(1)[0]
            return top_label if count / len(moves) >= frac_thresh else neutral
        x_label = dominant(x_moves)
        y_label = dominant(y_moves)
        z_label = dominant(z_moves)
        g_signs = [1 if g > 0 else -1 if g < 0 else 0 for g in window_grippers]
        if start > 0: g_signs.insert(0, 1 if grippers[start - 1] > 0 else -1)
        g_changes = [i for i in range(len(g_signs)-1) if g_signs[i+1] != g_signs[i]]
        g_label = None
        if g_changes:
            last_idx = g_changes[-1]
            g_label = "close gripper" if g_signs[last_idx+1] > 0 else "open gripper"
        parts = [p for p in [x_label, y_label, z_label, g_label] if p not in ("stay", None)]
        label = "move " + " ".join(parts) if parts else "stay"
        chunk_labels.append(label)
    return chunk_labels

def swap_directions(instruction):
    instruction = instruction.replace('left', '__TEMP_LEFT__').replace('right', 'left').replace('__TEMP_LEFT__', 'right')
    instruction = instruction.replace('forward', '__TEMP_FORWARD__').replace('backward', 'forward').replace('__TEMP_FORWARD__', 'backward')
    return instruction

def array_to_jpeg_bytes(img_array) -> bytes:
    buf = io.BytesIO()
    img_array.save(buf, format="JPEG")
    return buf.getvalue()

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

def save_visualization(start_img, end_img, task_instruction, motion_label, generated_label, ep_idx, chunk_idx):
    start_img_rgb = start_img.convert("RGB")
    end_img_rgb = end_img.convert("RGB")
    width, height = start_img_rgb.width, start_img_rgb.height
    canvas = Image.new("RGB", (width*2, height + 150), color=(255,255,255))
    canvas.paste(start_img_rgb, (0,0))
    canvas.paste(end_img_rgb, (width,0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_lines = [
        f"Task: {task_instruction}",
        f"Low-level motions: {motion_label}",
        f"Generated: {generated_label}"
    ]
    for i, line in enumerate(text_lines):
        draw.text((10, height+i*20), line, fill="black", font=font)
    vis_path = OUTPUT_DIR / f"episode{ep_idx}_{str(chunk_idx).zfill(3)}.png"
    canvas.save(vis_path)

async def generate_intermediate_instruction(semaphore, client, start_img, end_img, task_instruction, motion_label, i):
    styles = [
        # reasoning / anticipation
        "as someone predicting the next helpful move based on what’s happening",
        "as someone planning one or two steps ahead to reach the goal smoothly",
        "as someone explaining why the next adjustment makes sense",
        "as someone guiding the robot through intermediate setup steps before the main action",
        "as someone thinking strategically about approach, angle, or positioning",
    
        # monitoring / correction
        "as someone noticing a small mistake and steering the robot back on course",
        "as someone observing progress and suggesting fine-tuned corrections",
        "as someone course-correcting mid-way rather than restarting the whole task",
        "as someone refining what’s already happening instead of starting a new command",
        "as someone adjusting the robot’s behavior after seeing how it moved",
    
        # contextual / situational language
        "as someone giving directions relative to objects already in view",
        "as someone referencing the scene’s layout to describe the next step",
        "as someone talking about relative position rather than explicit coordinates",
        "as someone using visual cues ('toward the handle', 'near the edge') to guide motion",
        "as someone focusing on alignment, distance, or orientation adjustments",
    
        # intent-oriented phrasing
        "as someone emphasizing purpose ('get it lined up', 'prepare to place it')",
        "as someone telling the robot to ready or position itself before an action",
        "as someone expressing the intent behind the move, not just the motion",
        "as someone describing a transitional state rather than a completed action",
        "as someone prompting setup behaviors like orienting or steadying",
    
        # interpretive / reflective tone
        "as someone reasoning aloud about what the robot should do next",
        "as someone narrating the robot’s logic while giving the instruction",
        "as someone verbalizing an adjustment after interpreting the robot’s posture",
        "as someone explaining their reasoning to help the robot understand intent",
        "as someone reflecting on progress and updating the plan conversationally",
    
        # conversational naturalness
        "as someone giving mid-task coaching in natural, flowing speech",
        "as someone mixing small confirmations with guidance ('okay, now ease it forward')",
        "as someone talking to the robot like a collaborator mid-process",
        "as someone offering suggestions instead of direct orders",
        "as someone giving adaptive feedback while watching it act",
    ]
    style = random.choice(styles)


    prompt = f"""
    Task: {task_instruction}
    Inputs: Previous and current view of the robot
    Current Motion: {motion_label}
    
    Notes:
    - left/right = in-image left/right
    - forward = into screen; backward = out toward viewer
    
    Your task: Describe what the robot is doing now based on the images, motion, and task.
    Constraint: Only mention objects mentioned in the task.
    
    Output (JSON):
    ```json
    {{
      "all stages": # List all subtasks needed to finish the task,
      "subtask": # Briefly describe the robot’s current subtask,
      "reasoning": # Briefly explain how you determined which stage the robot is in and what it is doing,
      "command": # Rephrase 'subtask' as a natural, user-style instruction to the robot, {style},
    }}
    ```
    """

    start_img_bytes = array_to_jpeg_bytes(start_img)
    end_img_bytes = array_to_jpeg_bytes(end_img)
    async with semaphore:
        response = await client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=start_img_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=end_img_bytes, mime_type="image/jpeg"),
                prompt
            ]
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        try:
            data = json.loads(text)
            command = data.get("command", task_instruction).strip().lower().replace('.', '').replace(',', '')
        except json.JSONDecodeError:
            command = task_instruction
    return command

async def process_dataset(api_key):
    client = Client(api_key=api_key).aio
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    dataset = tfds.load(RAW_DATASET_NAME, data_dir=DATA_DIR, split="train")
    augmented_data = {}
    async with client as aclient:
        for ep_idx, episode in enumerate(tqdm(dataset, desc=f"Dataset {RAW_DATASET_NAME}")):
            steps = list(episode["steps"].as_numpy_iterator())
            delta_actions = [step["action"][:3] for step in steps]
            grippers = [step["action"][-1] for step in steps]
            task_instruction = swap_directions(steps[0]["language_instruction"].decode())
            motion_labels = chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE)
            tasks = []

            for i, motion_label in enumerate(motion_labels):
                start_idx = i*CHUNK_SIZE
                end_idx = min((i+4)*CHUNK_SIZE-1, len(steps)-1)
                start_img = Image.fromarray(steps[start_idx]["observation"]["image"])
                end_img = Image.fromarray(steps[end_idx]["observation"]["image"])
                motion_label_swapped = swap_directions(motion_label)
                tasks.append(generate_intermediate_instruction(
                    semaphore, aclient, start_img, end_img, task_instruction, motion_label_swapped, i
                ))
            paraphrased_labels = await asyncio.gather(*tasks)
            episode_dict = {}
            for i, (orig, para) in enumerate(zip(motion_labels, paraphrased_labels)):
                start_idx = i*CHUNK_SIZE
                end_idx = min((i+4)*CHUNK_SIZE-1, len(steps)-1)
                start_img = Image.fromarray(steps[start_idx]["observation"]["image"])
                end_img = Image.fromarray(steps[end_idx]["observation"]["image"])
                #save_visualization(start_img, end_img, task_instruction, motion_labels[i], para, ep_idx, i)
                episode_dict[f"step_{i}"] = {"original": orig, "paraphrased": para}
            augmented_data[f"episode_{ep_idx}"] = episode_dict

            
            augmented_instructions = []
            for p in paraphrased_labels:
                for _ in range(CHUNK_SIZE):
                    augmented_instructions.append(p)

            visualize(steps, augmented_instructions, task_instruction=task_instruction, stride=1, filename='%s.gif'%(str(ep_idx)))

            for j in range(len(motion_labels)):
                print(f"Original: {motion_labels[j]}, {task_instruction}\n-> Paraphrased: {paraphrased_labels[j]}")
                print("-" * 50)
            print("Done with 1 episode")
            break
    with open(f"paraphrased_instructions_{RAW_DATASET_NAME}.json", "w") as f:
        json.dump(augmented_data, f, indent=2)
    print(f"Augmented dataset saved to paraphrased_instructions_{RAW_DATASET_NAME}.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Google API key for Gemini")
    args = parser.parse_args()
    asyncio.run(process_dataset(args.api_key))

