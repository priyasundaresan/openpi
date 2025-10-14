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
from tqdm.asyncio import tqdm as async_tqdm
import tensorflow_datasets as tfds
from google.genai import Client, types

DATA_DIR = "modified_libero_rlds"
CHUNK_SIZE = 10
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
RPM = 4000  # Requests per minute
SEMAPHORE_LIMIT = 32  # concurrency per inference
EPISODE_CONCURRENCY = 4  # number of episodes processed concurrently
RAW_DATASET_NAME = "libero_10_no_noops"

# -------------------- Rate Limiter --------------------
class RateLimiter:
    def __init__(self, rpm):
        self.delay = 60 / rpm
        self._lock = asyncio.Lock()
        self._last_time = 0.0

    async def wait(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait_time = self.delay - (now - self._last_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_time = asyncio.get_event_loop().time()

# -------------------- Utilities --------------------
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

# -------------------- Visualization --------------------
def visualize(steps, aug_labels, task_instruction=None, stride=5, filename=None):
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

    if task_instruction:
        words = task_instruction.split()
        mid = len(words) // 2
        task_instruction = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        fig.suptitle(task_instruction, fontsize=16, y=0.97, ha="center")

    motion_text_obj = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=16)
    frames = []

    for t in range(0, len(steps), stride):
        ax_traj.cla()
        ax_img.cla()
        ax_traj.plot(positions[:t+1,0], positions[:t+1,1], positions[:t+1,2], "r-")
        color = "black" if grippers[t] > 0 else "white"
        ax_traj.scatter(positions[t,0], positions[t,1], positions[t,2], c=color, s=80, edgecolor="k")
        ax_traj.set_xlim(x_max, x_min)
        ax_traj.set_ylim(y_min, y_max)
        ax_traj.set_zlim(z_min, z_max)
        ax_traj.view_init(elev=20, azim=-180)
        img = steps[t]["observation"]["image"]
        ax_img.imshow(img)
        ax_img.axis("off")
        motion_label = aug_labels[t]
        words = motion_label.split()
        if len(words) > 4:
            mid = len(words)//2
            motion_label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        motion_text_obj.set_text(motion_label)

        if filename:
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 4)
            buf = buf[:, :, [1,2,3,0]]
            frame = buf[:, :, :3]
            frames.append(frame)
        else:
            plt.pause(0.1)

    if filename:
        imageio.mimsave(filename, frames, fps=20)
        print(f"Saved visualization to {filename}")
    else:
        plt.show()

# -------------------- Generate Intermediate Instruction --------------------
async def generate_intermediate_instruction(semaphore, rate_limiter, client, start_img, end_img, task_instruction, motion_label, i):
    styles = [
        "describe the next motion as targeting or aligning with a specific object",
        "give feedback that corrects the motion incrementally using small adjustments",
        "rephrase the step as part of a sequence of preparation, alignment, and placement",
        "describe motion relative to another object (e.g., behind, beside, toward)",
        "explain the step in terms of anticipating the next required movement",
        "mention avoiding collisions or staying within spatial constraints",
        "focus on how to use an object’s affordances (e.g., handle, surface, opening)",
    ]

    style = random.choice(styles)

    if 'gripper' not in motion_label:
        motion_label += ', and keeping the gripper unchanged'
    motion_label = motion_label.replace('move', 'Moving').replace('open', 'and opening the').replace('close', 'and closing the')
    motion_label += '.'
    task_instruction = task_instruction[0].upper() + task_instruction[1:]

    prompt = f"""
    Task: {task_instruction}.
    Current motion of the robot: {motion_label} Note that left/right refers to left/right in the image, forward is into the screen, and backward is out toward you as the viewer.
    Inputs: An image of the robot at the *start* of the task, and an image of the robot *currently*.

    Your task: Describe what subtask the robot is doing right now given its task and the current motion/image.
    Constraint: Only mention objects mentioned in the task.

    Output (JSON):
    ```json
    {{
      "all stages": # Describe all subtasks that the robot needs to do in order to complete the task,
      "subtask": # Briefly describe the robot’s current subtask,
      "reasoning": # Briefly explain how you determined which stage the robot is in and what it is doing,
      "command": # Rephrase 'subtask' as a natural, user-style instruction to the robot, as someone who would {style}.
    }}
    ```
    """

    start_img_bytes = array_to_jpeg_bytes(start_img)
    end_img_bytes = array_to_jpeg_bytes(end_img)
    await rate_limiter.wait()
    async with semaphore:
        start_img_bytes = array_to_jpeg_bytes(start_img)
        end_img_bytes = array_to_jpeg_bytes(end_img)
        response = await client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=start_img_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=end_img_bytes, mime_type="image/jpeg"),
                prompt
            ]
        )
        text = response.text.strip()
        print(text)
        if text.startswith("```json"): text = text[len("```json"):].strip()
        if text.endswith("```"): text = text[:-3].strip()
        try:
            data = json.loads(text)
            command = data.get("command", task_instruction).strip().lower().replace('.', '').replace(',', '')
        except json.JSONDecodeError:
            command = task_instruction
    return command

# -------------------- Process Episode --------------------
async def process_episode(episode, ep_idx, semaphore, rate_limiter, client):
    steps = list(episode["steps"].as_numpy_iterator())
    delta_actions = [step["action"][:3] for step in steps]
    grippers = [step["action"][-1] for step in steps]
    task_instruction = swap_directions(steps[0]["language_instruction"].decode())
    motion_labels = chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE)

    tasks = []
    for i, motion_label in async_tqdm(enumerate(motion_labels), total=len(motion_labels),
                                      desc=f"Episode {ep_idx} steps", leave=False):
        #start_idx = i*CHUNK_SIZE
        start_idx = 0
        end_idx = min((i+4)*CHUNK_SIZE-1, len(steps)-1)
        start_img = Image.fromarray(steps[start_idx]["observation"]["image"])
        end_img = Image.fromarray(steps[end_idx]["observation"]["image"])
        motion_label_swapped = swap_directions(motion_label)
        tasks.append(generate_intermediate_instruction(
            semaphore, rate_limiter, client, start_img, end_img, task_instruction, motion_label_swapped, i
        ))

    paraphrased_labels = await asyncio.gather(*tasks)

    augmented_instructions = []
    for p in paraphrased_labels:
        for _ in range(CHUNK_SIZE):
            augmented_instructions.append(p)

    visualize(steps, augmented_instructions, task_instruction=task_instruction, stride=1,
              filename=OUTPUT_DIR/f"{ep_idx}.gif")

    episode_dict = {}
    for i, (orig, para) in enumerate(zip(motion_labels, paraphrased_labels)):
        episode_dict[f"step_{i}"] = {"original": orig, "paraphrased": para}

    return f"episode_{ep_idx}", episode_dict

# -------------------- Process Dataset --------------------
async def process_dataset(api_key):
    client = Client(api_key=api_key).aio
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    rate_limiter = RateLimiter(RPM)

    dataset = tfds.load(RAW_DATASET_NAME, data_dir=DATA_DIR, split="train")
    augmented_data = {}

    async with client as aclient:
        ep_queue = []
        for ep_idx, episode in enumerate(tqdm(dataset, desc="Queueing episodes")):
            ep_queue.append(process_episode(episode, ep_idx, semaphore, rate_limiter, aclient))
            if len(ep_queue) >= EPISODE_CONCURRENCY:
                results = await asyncio.gather(*ep_queue)
                for ep_key, ep_data in results:
                    augmented_data[ep_key] = ep_data
                ep_queue = []

        # process any remaining episodes
        if ep_queue:
            results = await asyncio.gather(*ep_queue)
            for ep_key, ep_data in results:
                augmented_data[ep_key] = ep_data

    out_file = OUTPUT_DIR/f"paraphrased_instructions_{RAW_DATASET_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(augmented_data, f, indent=2)
    print(f"Augmented dataset saved to {out_file}")

# -------------------- Main --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Google API key for Gemini")
    args = parser.parse_args()
    asyncio.run(process_dataset(args.api_key))
