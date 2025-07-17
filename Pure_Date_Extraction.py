from openai import AzureOpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import json



def date_extract(user_prompt):
    """Use the LLM to extract dates from the user query."""
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = f"""You are a specialized date extraction assistant. Your only job is to identify and convert time references in user queries into specific dates in YYYY-MM-DD format.

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
    
    response = client.chat.completions.create(
        model="hj",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format = {"type":"json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error":"failed to parse JSON response","raw_response":response.choices[0].message.content}

    return response.choices[0].message.content
def describeVideo(user_prompt):
    date_info = date_extract(user_prompt)
    return date_info

user_prompt = "Find articles written by Will Dunham about Donald Trump over the past few weeks"