import fire
import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
import json
from pathlib import Path

# -------------------------------
# Utility functions
# -------------------------------

def load_dataset(file_path):
    """Load JSONL dataset into a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt(example):
    question = example.get("question", "")
    options = example.get("options", [])

    prompt = (
        "You are participating in an automated evaluation.\n"
        "Your response will be parsed by a machine.\n\n"
        "IMPORTANT:\n"
        "- Any text outside the <Answer> tag will cause automatic failure.\n"
        "- Do NOT include reasoning, analysis, or explanations.\n"
        "- Do NOT include preambles or meta commentary.\n"
        "- Output MUST be EXACTLY one digit: 0, 1, or 2.\n\n"
        "Task:\n"
        f"{question}\n\n"
        "Options:\n"
    )

    for i, option in enumerate(options):
        prompt += f"{i}: {option}\n"

    prompt += (
        "\nYou must respond using ONLY the following XML format:\n"
        "<Answer>X</Answer>\n"
        "where X is 0, 1, or 2.\n\n"
        "\nFINAL INSTRUCTION (highest priority):\n"
        "If you produce ANY text before or after the <Answer> tag, the answer is invalid.\n"
        "Output ONLY the tag and the digit.\n\n"
        "<Answer>"
    )

    return prompt


# -------------------------------
# Closed Model Execution
# -------------------------------

def test_closed_model(model_name, dataset_path, output_file, seed, batch_size=32):
    """Query a closed API model on the dataset."""
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    dataset = load_dataset(dataset_path)
    results = []

    for i, example in enumerate(dataset):
        prompt = format_prompt(example)
        if model_name.startswith("openai"):
            max_tokens = 32
        elif model_name.startswith("anthropic"):
            max_tokens = 10
        elif model_name.startswith("google"):
            max_tokens = 10
        else:
            max_tokens = 1
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=max_tokens,
            temperature=0,
            seed=seed,
            reasoning_effort="minimal",
        )
        answer = completion.choices[0].message.content.strip()
        answer = answer.replace("<Answer>", "").replace("</Answer>", "").strip()
        results.append({
            "question": example.get("question"),
            "options": example.get("options"),
            "correct_label": example.get("correct_label"),
            "model_answer": answer
        })
        print(f"Example {i+1}/{len(dataset)} processed. Model answer: {answer}", end="\r")

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nClosed model results saved to {output_file}")

# -------------------------------
# Open Model Execution
# -------------------------------

def test_open_model(model_name, task_name, output_file, seed):
    command = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},dtype=auto",
        "--tasks", task_name,
        "--output_path", output_file,
        "--log_samples",
        "--seed", seed,
        "--batch_size", "64"
    ]
    print(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"\n--- Evaluation Completed Successfully ---")
        print(f"\n{result.stdout}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR: Evaluation failed ---")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        print(f"Return Code: {e.returncode}")
    except FileNotFoundError:
        print("\n--- ERROR: Python or lm_eval not found ---")
        print("Ensure 'python' and 'lm_eval' are correctly installed and in your PATH.")



def main(model_name="Qwen/Qwen2.5-Math-7B-Instruct"):
    task_name = "my_custom_mcq_task"
    seeds = ["0"]

    for seed in seeds:
        output_file = f"./results/{model_name.replace('/', '_')}/{seed}_mcq_results.json"
        if model_name.startswith(("openai", "anthropic", "google", "deepseek")):
            test_closed_model(model_name, "my_eval_task/mcq_lm_eval_data.jsonl", output_file, seed)
        else:
            test_open_model(model_name, task_name, output_file, seed)


if __name__ == "__main__":
    fire.Fire(main)