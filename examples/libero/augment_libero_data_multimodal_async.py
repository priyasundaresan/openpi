import os
import random
import io
import json
import asyncio
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import numpy as np
import tensorflow_datasets as tfds
from google.genai import Client, types

# Parameters
DATA_DIR = "modified_libero_rlds"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 10
BATCH_SIZE = 100  # max concurrent episodes
RPM = 10000
RPS = RPM / 60

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

def tile_frames(frames, thumbnail_size=(128,128), grid_cols=6):
    # Load a larger TrueType font
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)  # 16 pt font
    #font = ImageFont.load_default()  # fallback if arial not found

    n_frames = len(frames)
    grid_rows = (n_frames + grid_cols - 1) // grid_cols
    tile_width, tile_height = thumbnail_size
    canvas_width = grid_cols * tile_width
    canvas_height = grid_rows * tile_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255,255,255))

    for idx, frame in enumerate(frames):
        if idx >= grid_rows * grid_cols:
            break
        frame_thumb = frame.resize(thumbnail_size)
        row = idx // grid_cols
        col = idx % grid_cols
        x, y = col * tile_width, row * tile_height
        canvas.paste(frame_thumb, (x, y))

        draw = ImageDraw.Draw(canvas)
        draw.text((x+2, y+2), str(idx), fill="white", font=font)  # white text

    return canvas

def visualize(steps, aug_labels, task_instruction=None, stride=1, filename=None):
    delta_actions = np.array([s["action"][:3] for s in steps])
    grippers = np.array([s["action"][-1] for s in steps])
    positions = np.cumsum(delta_actions, axis=0)
    positions = np.vstack([[0,0,0], positions])
    x_min, x_max = positions[:,0].min(), positions[:,0].max()
    y_min, y_max = positions[:,1].min(), positions[:,1].max()
    z_min, z_max = positions[:,2].min(), positions[:,2].max()
    import matplotlib.pyplot as plt
    import imageio
    fig = plt.figure(figsize=(10,6))
    ax_traj = fig.add_subplot(1,2,1, projection="3d")
    ax_img = fig.add_subplot(1,2,2)
    if task_instruction:
        words = task_instruction.split()
        mid = len(words)//2
        task_instruction = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        fig.suptitle(task_instruction, fontsize=16, y=0.97, ha="center")
    motion_text_obj = fig.text(0.5,0.02,"",ha="center",va="bottom",fontsize=16)
    frames_img = []
    for t in range(0,len(steps),stride):
        ax_traj.cla()
        ax_img.cla()
        ax_traj.plot(positions[:t+1,0],positions[:t+1,1],positions[:t+1,2],"r-")
        color="black" if grippers[t]>0 else "white"
        ax_traj.scatter(positions[t,0],positions[t,1],positions[t,2],c=color,s=80,edgecolor="k")
        ax_traj.set_xlim(x_max,x_min)
        ax_traj.set_ylim(y_min,y_max)
        ax_traj.set_zlim(z_min,z_max)
        ax_traj.view_init(elev=20,azim=-180)
        img = steps[t]["observation"]["image"]
        ax_img.imshow(img)
        ax_img.axis("off")
        motion_label = aug_labels[t]
        words = motion_label.split()
        if len(words)>4:
            mid = len(words)//2
            motion_label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        motion_text_obj.set_text(motion_label)
        if filename:
            fig.canvas.draw()
            w,h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(),dtype=np.uint8)
            buf = buf.reshape(h,w,4)
            buf = buf[:,:, [1,2,3,0]]
            frames_img.append(buf[:,:,:3])
        else:
            plt.pause(0.1)
    if filename:
        imageio.mimsave(filename, frames_img, fps=20, loop=0)
        print(f"Saved visualization to {filename}")
    else:
        plt.show()

async def generate_episode_instruction(semaphore, ep_idx, frames, task_instruction, motion_labels, client):
    async with semaphore:
        # tile frames
        tiled_image = await asyncio.to_thread(lambda: tile_frames(frames[::CHUNK_SIZE]))
        tiled_bytes = await asyncio.to_thread(lambda: array_to_jpeg_bytes(tiled_image))

        task_instruction = swap_directions(task_instruction)

        task_instruction = task_instruction[0].upper() + task_instruction[1:]
        motion_labels_text = []
        for idx, ml in enumerate(motion_labels):
            ml = swap_directions(ml)
            ml_text = ml.replace('move','Moving').replace('open','and opening the').replace('close','and closing the')
            if 'gripper' not in ml_text:
                ml_text += ', and keeping the gripper unchanged'
            ml_text += '.'
            motion_labels_text.append(f"step_{idx}: {ml_text}")
        motion_labels_text_str = "\n".join(motion_labels_text)
        prompt = f"""
Overall Task: {task_instruction}.
Motions per step:
{motion_labels_text_str}
Note: left/right refers to left/right in the image, forward is into the screen, backward is toward the viewer.

Inputs: A tiled image of the robot performing the task, numbered by step.

Your task: Describe what subtask the robot is doing for each frame given its overall task and current motion.
Constraint: Only mention objects mentioned in the task, match the number of frames above.

Output format (JSON):
```json
{{
  "step_0": {{
      "subtask": # brief description of the current subtask,
      "reasoning": # explain the subtask and which strategy you picked for the command,
      "command": # rephrase 'subtask' as a natural, user-style instruction to the robot
  }},
  "step_1": {{ ... }},
  ...
}}
```
"""
        response = await client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=tiled_bytes, mime_type="image/jpeg"),
                prompt
            ]
        )
        try:
            text = response.text.strip()
            if text.startswith("```json"): text=text[len("```json"):].strip()
            if text.endswith("```"): text=text[:-3].strip()
            data = json.loads(text)
        except:
            print(response)
            data = {}
            print("Failed to parse JSON. Returning empty dictionary.")

        episode_dict = {}
        for step_idx, orig in enumerate(motion_labels):
            step_key = f"step_{step_idx}"
            # Use 'subtask' from Gemini if available; fallback to original motion label
            para = data.get(step_key, {}).get("command", orig).strip().lower().replace(',', '').replace('.', '')
            episode_dict[step_key] = {"original": orig, "paraphrased": para}

        return f"episode_{ep_idx}", episode_dict


async def process_dataset(api_key, num_episodes=None):
    client = Client(api_key=api_key).aio
    all_augmented_data = {}

    for RAW_DATASET_NAME in RAW_DATASET_NAMES:
        dataset = tfds.load(RAW_DATASET_NAME, data_dir=DATA_DIR, split="train")
        dataset = list(dataset)  # materialize so we can slice into batches
        total_episodes = len(dataset)
        print(f"Processing dataset {RAW_DATASET_NAME} with {total_episodes} episodes")

        if num_episodes:
            dataset = dataset[:num_episodes]
            total_episodes = len(dataset)

        all_augmented_data[RAW_DATASET_NAME] = {}

        async with client as aclient:
            # process in batches of BATCH_SIZE
            for start in range(0, total_episodes, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total_episodes)
                batch = dataset[start:end]
                print(f"  → Processing episodes {start}–{end-1}")

                semaphore = asyncio.Semaphore(BATCH_SIZE)
                tasks = []

                for ep_idx, episode in enumerate(batch, start=start):
                    steps = list(episode["steps"].as_numpy_iterator())
                    frames = [Image.fromarray(s["observation"]["image"]) for s in steps]
                    delta_actions = [s["action"][:3] for s in steps]
                    grippers = [s["action"][-1] for s in steps]
                    motion_labels = chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE)
                    task_instruction = steps[0]["language_instruction"].decode()

                    tasks.append(
                        generate_episode_instruction(semaphore, ep_idx, frames, task_instruction, motion_labels, aclient)
                    )

                # Run this batch concurrently
                results = await asyncio.gather(*tasks)

                # Store results
                for ep_key, ep_data in results:
                    all_augmented_data[RAW_DATASET_NAME][ep_key] = ep_data

                # optional: save intermediate JSON after each batch
                out_file = OUTPUT_DIR / f"paraphrased_instructions_{RAW_DATASET_NAME}.json"
                with open(out_file, "w") as f:
                    json.dump(all_augmented_data[RAW_DATASET_NAME], f, indent=2)
                print(f"  ✓ Saved progress through episode {end-1}")

        print(f"Finished {RAW_DATASET_NAME}")

## ---- Main dataset processing ----
#async def process_dataset(api_key, num_episodes=None):
#    client = Client(api_key=api_key).aio
#    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
#    all_augmented_data = {}
#
#    for RAW_DATASET_NAME in tqdm(RAW_DATASET_NAMES, desc="Datasets"):
#        dataset = tfds.load(RAW_DATASET_NAME, data_dir=DATA_DIR, split="train")
#        print(f"Processing dataset {RAW_DATASET_NAME} with {len(dataset)} episodes")
#
#        all_augmented_data[RAW_DATASET_NAME] = {}
#
#        async with client as aclient:
#            tasks = []
#            for ep_idx, episode in tqdm(enumerate(dataset), desc=f"Loading {RAW_DATASET_NAME} episodes"):
#                if num_episodes and ep_idx >= num_episodes:
#                    break
#        
#                steps = list(episode["steps"].as_numpy_iterator())
#                frames = [Image.fromarray(s["observation"]["image"]) for s in steps]
#                delta_actions = [s["action"][:3] for s in steps]
#                grippers = [s["action"][-1] for s in steps]
#                motion_labels = chunk_motion_labels(delta_actions, grippers, chunk_size=CHUNK_SIZE)
#                task_instruction = steps[0]["language_instruction"].decode()
#        
#                tasks.append(
#                    generate_episode_instruction(semaphore, ep_idx, frames, task_instruction, motion_labels, aclient)
#                )
#        
#            # Wrap asyncio.gather with tqdm for progress
#            results = []
#            for f in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {RAW_DATASET_NAME}"):
#                res = await f
#                results.append(res)
#
#            # Save per-dataset JSON
#            for ep_key, ep_data in results:
#                all_augmented_data[RAW_DATASET_NAME][ep_key] = ep_data
#
#        # Save JSON file
#        out_file = OUTPUT_DIR / f"paraphrased_instructions_{RAW_DATASET_NAME}.json"
#        with open(out_file, "w") as f:
#            json.dump(all_augmented_data[RAW_DATASET_NAME], f, indent=2)
#        print(f"Augmented dataset saved to {out_file}")
#
#        ## --- Generate some GIFs using paraphrased labels ---
#        #MAX_GIFS = 3
#        #dataset_iter = tfds.load(RAW_DATASET_NAME, data_dir=DATA_DIR, split="train")
#        #
#        #for ep_idx, episode in enumerate(dataset_iter):
#        #    if ep_idx >= MAX_GIFS:
#        #        break
#        #
#        #    ep_key = f"episode_{ep_idx}"
#        #    if ep_key not in all_augmented_data[RAW_DATASET_NAME]:
#        #        continue
#        #
#        #    ep_data = all_augmented_data[RAW_DATASET_NAME][ep_key]
#        #    paraphrased_labels = []
#        #    for i in range(len(ep_data)):
#        #        for _ in range(CHUNK_SIZE):
#        #            paraphrased_labels.append(ep_data[f"step_{i}"]["paraphrased"])
#        #
#        #    steps = list(episode["steps"].as_numpy_iterator())
#        #    task_instruction = steps[0]["language_instruction"].decode()
#        #
#        #    visualize(
#        #        steps=steps,
#        #        aug_labels=paraphrased_labels,
#        #        task_instruction=task_instruction,
#        #        stride=1,
#        #        filename=str(OUTPUT_DIR / f"{RAW_DATASET_NAME}_{ep_key}.gif")
#        #    )
#        #    print(f"Saved GIF for {RAW_DATASET_NAME} {ep_key}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Google API key for Gemini")
    args = parser.parse_args()
    #asyncio.run(process_dataset(args.api_key, num_episodes=3))
    #asyncio.run(process_dataset(args.api_key, num_episodes=100))
    asyncio.run(process_dataset(args.api_key))
