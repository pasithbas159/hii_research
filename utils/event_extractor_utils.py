import os
import json
import glob
from typing import List
from dotenv import load_dotenv
from datetime import date
from langsmith import Client
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from prompt_utils import json_parsers_template

class EventOutput(BaseModel):
    eventid: str = Field(descriptoin="must be unique among events. include the report file name and running number from 0")
    name: str = Field(description="name of the event should be a very comprehensive description of the event and not include causation or date (for example น้ำล้นตลิ่ง, ฝนฟ้าคะนอง, ลมกรรโชกแรง, ความกดอากาศสูง)")
    eventdate: date = Field(description="date of the event from report. if you not sure you can use report file name date instead")
    place: str = Field(description="place where event occur")
    
    
class JSONOutput(BaseModel):
    events: List[EventOutput] = Field(description="List of events that happened from report")
    relationships: List[str] = Field(description="List of relationships between eventid in the following format eventid|CAUSES|eventid")
    

def event_extractor_from_markdown(llm, input_dir, output_dir): 
    """
    Input file path then get the extractor for creating GraphRAG

    Args:
        llm: large language module in langchain
        input_file (str): input file
        output_path (str): output directory
    """
    with open(input_dir, "r", encoding="utf-8") as f:
        report = f.read()
        file_name = os.path.basename(input_dir)

    json_parser = PydanticOutputParser(pydantic_object=JSONOutput)
    
    json_parser_tmpl = PromptTemplate(
        template=json_parsers_template,
        input_variables=["file_name", "report"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )
    
    parser_and_model = json_parser_tmpl | llm
    
    response = parser_and_model.invoke(file_name + ' \n ' + report)
    
    raw_response = response.content
    
    start = raw_response.find("{")
    end = raw_response.rfind("}") + 1
    json_str = raw_response[start:end]
    
    try:
        data = json.loads(json_str)
        
        output_file = f"{output_dir}{file_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return f"JSON data has been saved to {output_file}"
    except json.JSONDecodeError as e:
        return f"Failed to decode JSON: {e}"

if __name__ == '__main__':
    
    load_dotenv()

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"HII - data extractor"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # test_file = 'documents/markdown/1000_20210419_Predict_SendRid.md'
    # input_dir = 'documents/markdown/'
    
    
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2046,
        max_retries=5,
    )
    
    # Initialize LangSmith client to trace the code
    client = Client()
    
    for file in list(glob.glob('documents/markdown/*.md')):
        print(file)
        event_extractor_from_markdown(llm=llm, input_dir=file, output_dir='documents/json/')