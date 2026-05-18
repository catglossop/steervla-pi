import tensorflow_datasets as tfds

# builder = tfds.builder(name="simplified_reasoning_dataset", data_dir="/raid/datasets/steervla")
builder = tfds.builder(name="simlingo_dataset_all_img512_1116", data_dir="/raid/datasets/steervla")

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





