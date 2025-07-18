from openai import AzureOpenAI
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field
from typing import Optional

class DateRange(BaseModel):
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")

def date_extract(user_prompt):
    """Use the LLM to extract dates from the user query."""
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_monday_str = last_monday.strftime("%Y-%m-%d")

    system_prompt = f"""You are a specialized date extraction assistant. Your only job is to identify and convert time references in user queries into specific dates in YYYY-MM-DD format.
TASK:
- Extract ALL time references from the user's query
- Convert relative time references (like "3 weeks ago", "last month") to absolute dates
- For more abstract time references like "recently", "these days", or "lately", convert it to a time range of the past week or month depending on context
- With phrases like, "over the past few (time range), it's best to give a date range rather than a specific date.
- For each identified time reference, output the original reference and its corresponding absolute date
- Use today's date ({today_str}) as the reference point for relative dates
- If a publication date is provided (as metadata or in the text), and it differs from other dates mentioned in the title or content, always prioritize and output the publication date.
- If both a publication date and a different date are found, ONLY output the publication date
- For the phrase 'last week', the range should start from the Monday of the previous week. For example, if today is Wednesday, July 17, 2025, and the query has 'last week', your start date should be July 7, 2025 (Monday of last week).
- The same also goes for 'two weeks ago', 'three weeks ago', etc. If today is July 17, 2025, and the query has 'two weeks ago', the range should be from July 1, 2025 (Monday two weeks ago) to July 7, 2025 (Sunday of that same week). Repeat this for 3 weeks ago, 4 weeks etc.
- The same goes for 'last month', always start from the first day of the previous month. If it says '2 months ago', make it the first day of the month before that.
- For the phrase 'this year', you should also go back to January first until today. So if the prompt has 'this year' and today is July 17, 2025, your range should be from January 1, 2025 to July 17, 2025.
- Your outputted dates should never be from the future. If the month is June and the query asks for something from November, it should be from November of the previous year.
- For mentions of seasons in queries, these can overlap. Winter should be between December and March, Spring should be between March and June, Summer should be between June and September, and Fall/Autumn should be from September to December. Again these dates can overlap.
- If no time reference is found, your output should be a JSON object with "start_date" and "end_date" both set to null.

OUTPUT FORMAT:
You must respond with valid JSON only. Your JSON should adhere to the following schema:
{{
    "end_date":"YYYY-MM-DD",
    "start_date":"YYYY-MM-DD"
}}
Where "end_date" and "start_date" are optional fields. If no date is found, both should be null.
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
        
        date_data = DateRange.model_validate_json(response.choices[0].message.content)
        
        return date_data.model_dump()
    except Exception as e: 
        print(f"Error parsing LLM response: {e}")
        
        return DateRange().model_dump()

def describeVideo(user_prompt):
    date_info = date_extract(user_prompt)
    return date_info