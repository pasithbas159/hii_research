import os
import streamlit as st
from langsmith import Client
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
import chromadb
import torch 
import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from dotenv import load_dotenv

from utils.rag_utils import initialize_neo4j_rag
from utils.prompt_utils import instruction_prompt

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"HII - graph retriever"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

load_dotenv()

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
LANGCHAIN_API_KEY_GRAPH=os.environ.get('LANGCHAIN_API_KEY_GRAPH')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize LangSmith client to trace the code
client = Client(api_key=LANGCHAIN_API_KEY_GRAPH)


@tool
def get_cypher_chain(message: str) -> str:
    """
    Query information from graph database to see cause and effect.
    Which means that you need to start with the form of the following:
    MATCH (e1: Event) -[:CAUSES]->(e2:Event)
    WHERE ...your conditions here...
    RETURN e1, e2
    """
    
    graph = initialize_neo4j_rag()
    
    graph.refresh_schema()

    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm = ChatOpenAI(model="gpt-4o",max_tokens=2046,max_retries=2,),
        qa_llm = ChatOpenAI(model="gpt-4o",max_tokens=2046,max_retries=2,),
        graph=graph,
        verbose=True, 
        allow_dangerous_requests=True
    )
    res = cypher_chain.invoke({"query": message})
    
    return res['result']

@st.cache_resource
def get_agent_executor():
    
    llm = ChatOpenAI(model="gpt-4o",max_tokens=2046,max_retries=2,temperature=0.1)

    prompt = hub.pull("hwchase17/react")

    prompt.template = instruction_prompt + '\n' + prompt.template

    tools = [get_cypher_chain]

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor


def chat_with_graphrag(): 
    
    st.title("HII Water LLM Event App")
    
    agent_executor = get_agent_executor()

    with st.form("my_form"):
        
        st.text("Your input. For example:")
        st.caption("ลมตะวันออกและลมตะวันออกเฉียงใต้พัดปกคลุมภาคใต้ก่อให้เกิดผลกระทบอะไรบ้าง?")
        st.caption("สาเหตุที่ทำให้เกิดฝนตกในประเทศไทยตอนบน")
        st.caption("หย่อมความกดอากาศต่ำเนื่องจากความร้อนในเดือนมีนาคมก่อให้เกิดอะไรบ้าง?")
        st.caption("ฝนตกหนักถึงหนักมากในเดือน 2024-07 ทำให้เกิดผลกระทบอะไร?")
        
        text = st.text_area(
            "",
            "ลมตะวันออกและลมตะวันออกเฉียงใต้พัดปกคลุมภาคใต้ก่อให้เกิดผลกระทบอะไรบ้าง?",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            with st.spinner('Generating the answer...'):
                res = agent_executor.invoke({"input": text})
                st.info(res['output'])
                
def clip_image_search(collection_name): 
    
    chroma_client = chromadb.PersistentClient(path="./chroma")
    
    image_collection = chroma_client.get_collection(name=collection_name)
    
    if collection_name == "image": 
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    elif collection_name == "new_image": 
        processor = CLIPProcessor.from_pretrained("./models/hii-finetuned-clip-v1-processor")
        model = CLIPModel.from_pretrained("./models/hii-finetuned-clip-v1-model")

    with st.form("image_form"):
        st.text("Your input. For example:")
        st.caption("แผนที่แสดงปริมาณฝนสะสมรายสัปดาห์")
        st.caption("กราฟแสดงปริมาณน้ำระบายสะสมตั้งแต่ต้นปีในแต่ละภาค")
        st.caption("ภาพแผนที่คลื่นความร้อนฝั่งอ่าวไทยและอันดามัน")
        st.caption("ภาพแผนที่คลื่นอากาศ")
        
        text_input = st.text_area(
            "Enter text to search image related to your keyword.",
            "กราฟแสดงปริมาณน้ำระบายสะสมตั้งแต่ต้นปีในแต่ละภาค",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            inputs = processor(
                    text=text_input,
                    return_tensors="pt", 
                    truncation=True
            )
            text_features = model.get_text_features(**inputs).detach().cpu().numpy()
            result = image_collection.query(text_features, n_results=1)
            with st.spinner('Searching the image...'):
                for i in result['metadatas'][0]:
                    img = Image.open(i['path'])
                    st.image(img)
        
        
def main():
    st.set_page_config(
        page_title="HII Water App",
        layout="wide"
    )
    
    st.sidebar.title("Application")
    page = st.sidebar.selectbox(
        "Choose an example",
        ["Chat with GraphRAG", "CLIP Image Search"]
    )
    
    if page == "Chat with GraphRAG":
        chat_with_graphrag()
    elif page == "CLIP Image Search":
        st.sidebar.title("Model")
        models = st.sidebar.selectbox(
            "Choose a model",
            ["Fine-tuned CLIP", "Original CLIP"]
        )
        
        if models == "Fine-tuned CLIP":
            clip_image_search(collection_name="new_image")
        elif models == "Original CLIP":
            clip_image_search(collection_name="image")
        
        
if __name__ == "__main__":
    main()