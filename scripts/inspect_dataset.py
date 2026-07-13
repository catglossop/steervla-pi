import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# builder = tfds.builder(name="simplified_reasoning_dataset", data_dir="/raid/datasets/steervla")
builder = tfds.builder(name="simlingo_dataset_acceleration_negative1_img512_1116", data_dir="/raid/datasets/steervla")

ds = builder.as_dataset(split="train")
for example in ds:
    steps = example["steps"]
    for step in steps:
        print("Gemini Refined Label: ", step["gemini_refined_label"])
        print("Routing Command: ", step["routing_command"])
        print("Commentary: ", step["commentary"])
        print("Answer: ", step["answer"])
        print("Prompt: ", step["prompt"])
        breakpoint()
        # plot the actions
        actions = step["action"]
        waypoints_t = np.cumsum(np.array(actions["future_10_xy_delta_t"]), axis=0)
        waypoints_space = np.cumsum(np.array(actions["future_10_xy_delta_space"]), axis=0)
        breakpoint()
        plt.figure(figsize=(10, 10))
        plt.plot(waypoints_t[:, 0], waypoints_t[:, 1], label="Waypoints T")
        plt.plot(waypoints_space[:, 0], waypoints_space[:, 1], label="Waypoints Space")
        plt.legend()
        plt.savefig(f"waypoints_{step['gemini_refined_label']}.png")
        
        
    break

builder = tfds.builder(name="simplified_reasoning_dataset", data_dir="/raid/datasets/steervla")


ds = builder.as_dataset(split="train")
for example in ds:
    steps = example["steps"]
    for step in steps:
        print("Gemini Refined Label: ", step["gemini_refined_label"])
        print("Routing Command: ", step["routing_command"])
        print("Commentary: ", step["commentary"])
        print("Answer: ", step["answer"])
        print("Prompt: ", step["prompt"])
        breakpoint()
        break
    break





