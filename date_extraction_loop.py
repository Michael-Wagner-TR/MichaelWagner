import json
import time
from datetime import datetime
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import concurrent.futures


#Set up loop for improving the system prompt.
def system_loop(user_prompt, system_prompt, max_retries=3):
    load_dotenv()
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        api_version = "2024-12-01-preview",
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model = "hj",
                messages = [
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_prompt}
                ],
                response_format = {"type":"json_object"}
                
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)

            if "start_date" not in parsed or "end_date" not in parsed:
                raise ValueError("Missing required fields in response")
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON parsing error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {"error":"json_parse_error","raw_response":content}
        except Exception as e:
            error_str = str(e)
            #Skip through any API content filter errors
            if "content_filter" in error_str or "ResponsibleAIPolicyViolation" in error_str:
                print(f"Content filter triggered for query: {user_prompt[:100]}...")
                return {
                    "start_date":"1900-01-01",
                    "end_date":"1900-01-01",
                    "error":"content_filter",
                    "original_query": user_prompt
                }
            elif attempt == max_retries -1:
                return {"error":error_str,"raw_response":""}
            time.sleep(2 ** attempt)
    return {"error":"max_retries_exceeded"}


#Function for extracting the date from a user query
def process_query(args):
    q, system_prompt = args
    extracted = system_loop(q["query"], system_prompt)
    try:
        start_date = datetime.strptime(extracted.get("start_date"), "%Y-%m-%d")
        end_date = datetime.strptime(extracted.get("end_date"), "%Y-%m-%d")
        actual_date = datetime.strptime(q["actual_date"], "%Y-%m-%d")
        is_accurate = start_date <= actual_date <= end_date
    except Exception as e:
        is_accurate = False
    return {
        "query": q["query"],
        "actual_date": q["actual_date"],
        "extracted_start": extracted.get("start_date"),
        "extracted_end": extracted.get("end_date"),
        "is_accurate": is_accurate
    }
#Function that improves the system prompt for the date extraction. First load in the LLM. 
#Use only 15 examples to help the LLM improve the system prompt as to not overload it. 
def improve_system(previous_prompt,inaccurate_cases,max_examples = 15):
    load_dotenv()
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        api_version = "2024-12-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    examples = inaccurate_cases[:max_examples]

    error_examples = []
    for case in examples:
        error_examples.append(
            f"Query: '{case['query']}'\n"
            f"Expected: {case['actual_date']}'\n"
            f"Got:{case.get('extracted_start','None')} to {case.get('extracted_end','None')}\n"
        )
#This prompt instructs the LLM how to improve the system prompt of the date extraction. The LLM is instructed to add information and instruction to the initial date extraction prompt.
#Instead of rewriting the entire extraction prompt, the LLM uses the inaccurate examples to add information to the extraction prompt. 
    user_message = f"""
    Analyze these {len(examples)} failed date extraction cases:

    {chr(15).join(error_examples)}

    Current prompt:
    {previous_prompt}
    Please improve the prompt by adding more instruction and information to it. Do not remove any information already in the prompt. 
    Your added instruction should try to include the following criteria:
1. Common error patterns you observe
2. Specific improvements to handle these cases
3. Clearer instructions for edge cases
Again, the improved prompt should contain everything that was in the previous prompt along with any new instruction you put.
Return only the improved prompt.
"""
    try:
        response = client.chat.completions.create(
            model = "hj",
            messages = [
                {"role":"system","content":"You are an expert prompt engineer specializing in date extraction tasks."},
                {"role":"user","content":user_message}
            ]
            
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error improving prompt: {e}")
        return previous_prompt
#Set up the loop to continuously use the LLM to improve the extraction prompt over as many iterations as possible. 
#In this case only 3 iterations are used. 
def prompt_loop(
        dataset_path,
        initial_prompt,
        max_iterations=3,
        delay=0.5,
        output_prefix="prompt_loop_yon",
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
    #Saves the system prompt for each iteration
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

today = datetime.now().strftime('%Y-%m-%d')
#This is the initial extraction prompt
initial_prompt = f"""You are a specialized date extraction assistant. Your only job is to identify and convert time references in user queries into specific dates in YYYY-MM-DD format.

TASK:
- Extract ALL time references from the user's query
- Convert relative time references (like "3 weeks ago", "last month") to absolute dates
- For more abstract time references like "recently", "these days", or "lately", convert it to a time range of the past week or month depending on context
- With phrases like, "over the past few (time range), it's best to give a date range rather than a specific date.
- For each identified time reference, output the original reference and its corresponding absolute date
- Use today's date ({today}) as the reference point for relative dates
- If a publication date is provided (as metadata or in the text), and it differs from other dates mentioned in the title or content, always prioritize and output the publication date.
- If both a publication date and a different date are found, ONLY output the publication date
-For the phrase last week, start from the Monday of the previous week. If today is Wednesday June 18th and the query has 'last week', you should start from Monday June 9th.
-The same also goes for two weeks ago, three weeks ago, etc. If today is June 20th, and the query has two weeks ago in it, the range should be from June 2nd the end of that week. Repeat this for 3 weeks ago, 4 weeks etc.
-The same goes for last month, always start from the first day of the previous month. If it says 2 months ago, make it the first day of the month before that.
-For the phrase 'this year', you should also go back to January first until today. So if the prompt has this year and today is June 20th 2025, your range should be from January 1st 2025 to June 20th.
-Your outputted dates should never be from the future. If the month is June and the query asks for something from November, it should be from November of the previous year.
For mentions of seasons in queries, these can overlap. Winter should be between December and March, Spring should be between March and June, Summer should be between June and September, and Fall/Autumn should be from September to December. Again these dates can overlap.

- If no time reference is found, respond with "No date reference found"

OUTPUT FORMAT:

You must respond with valid JSON only. 
{{
    "end_date":"YYYY-MM-DD",
    "start_date":"YYYY-MM-DD",
}}
"""
