## Automatic Prompt Evaluation and Improvement for Temporal Analysis of User Queries.

This repository contains information for both date extraction from a single query, as well as a prompt self improvement loop

# Example usage for pure date extraction

from Pure_Date_Extraction import describeVideo

user_prompt = "Find articles written by Will Dunham about Donald Trump over the past few weeks"
date_info = describeVideo(user_prompt)
print(date_info)


# Examples usage for the self improvement loop

from date_extraction_loop import prompt_loop

prompt_loop(
    dataset_path="your_dataset.json",           # Path to your dataset
    initial_prompt="...",                       # The initial system prompt (see script for sample)
    max_iterations=3,                           # Number of improvement rounds
    delay=0.5,                                  # Delay between API calls (seconds)
    output_prefix="prompt_loop_results",        # Prefix for output files
    max_workers=5                               # Number of parallel workers
)
