import json
import time
from datetime import datetime
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import concurrent.futures

def system_loop(user_prompt, system_prompt):
    
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    response = client.chat.completions.create(
        model="hj",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_response": response.choices[0].message.content}

def improve_system(previous_prompt, inaccurate_cases):
    
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    user_message = (
        "Here are some queries where the date extraction failed:\n"
        + "\n".join(f"-{case['query']}: expected {case['actual_date']}, got {case.get('extracted_start')} to {case.get('extracted_end')}" for case in inaccurate_cases)
        + "\n\nPlease rewrite the following date extraction prompt to improve extraction accuracy for these cases:\n"
        + previous_prompt
    )
    response = client.chat.completions.create(
        model="hj",
        messages=[
            {"role": "system", "content": "You are an expert prompt engineer"},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()

def process_query(args):
    q, system_prompt = args
    extracted = system_loop(q["query"], system_prompt)
    try:
        start_date = datetime.strptime(extracted.get("start_date"), "%Y-%m-%d")
        end_date = datetime.strptime(extracted.get("end_date"), "%Y-%m-%d")
        actual_date = datetime.strptime(q["actual_date"], "%Y-%m-%d")
        is_accurate = start_date <= actual_date <= end_date
    except Exception:
        is_accurate = False
    return {
        "query": q["query"],
        "actual_date": q["actual_date"],
        "extracted_start": extracted.get("start_date"),
        "extracted_end": extracted.get("end_date"),
        "is_accurate": is_accurate
    }

def prompt_loop(
        dataset_path,
        initial_prompt,
        max_iterations=3,
        delay=0.5,
        output_prefix="prompt_loop",
        max_workers=5
):
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    queries = []
    for entry in dataset:
        if "generated_query" in entry:
            for q in entry["generated_query"]:
                queries.append({"query": q, "actual_date": entry["date"]})
        elif "query" in entry:
            queries.append({"query": entry["query"], "actual_date": entry["actual_date"]})

    system_prompt = initial_prompt
    prompt_history = [system_prompt]
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            args = [(q, system_prompt) for q in queries]
            for result in executor.map(process_query, args):
                results.append(result)
                time.sleep(delay)
        with open(f"{output_prefix}_iter{iteration + 1}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        accuracy = sum(r["is_accurate"] for r in results) / len(results) if results else 0
        print(f"Accuracy after iteration {iteration + 1}: {accuracy:.2%}")
        inaccurate = [r for r in results if not r["is_accurate"]]
        if not inaccurate:
            print("No inaccurate queries left. Stopping early.")
            break
        system_prompt = improve_system(system_prompt, inaccurate)
        prompt_history.append(system_prompt)
    with open(f"{output_prefix}_prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompt_history, f, ensure_ascii=False, indent=2)
    print("Process Complete.")