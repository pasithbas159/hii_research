from dotenv import load_dotenv
import glob
import json
import os
import chromadb

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

def generate_cypher_from_json(json_data):
    with open(json_data, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    events = data.get("events", [])
    relationships = data.get("relationships", [])
    cause_effect_dict = {}
    
    # Start building the Cypher query
    cypher_query = []
    
    # Create nodes for events
    for event in events:
        cypher_query.append(f"""
        MERGE (e:Event {{eventid: '{event["name"]}'}})
        ON CREATE SET
            e.name = '{event["name"]}',
            e.eventdate = '{event["eventdate"]}',
            e.place = '{event["place"]}';
        """)
        cause_effect_dict[event["eventid"]] = event["name"]
    
    # Create relationships
    for relationship in relationships:
        parts = relationship.split("|")
        if len(parts) == 3:
            source, rel_type, target = parts
            cypher_query.append(f"""
            MATCH (e1:Event {{eventid: '{cause_effect_dict[source]}'}}), (e2:Event {{eventid: '{cause_effect_dict[target]}'}})
            MERGE (e1)-[:{rel_type}]->(e2);
            """)
    
    # Combine all Cypher queries into a single script
    return "\n".join(cypher_query)

def initialize_neo4j_rag(): 
    
    load_dotenv()

    NEO4J_URI=os.environ.get("NEO4J_URI")
    NEO4J_USERNAME=os.environ.get("NEO4J_USERNAME")
    NEO4J_PASSWORD=os.environ.get("NEO4J_PASSWORD")

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        enhanced_schema=True,
    )
    
    return graph

def initialize_chroma_rag(collection_name): 
    
    chroma_client = chromadb.PersistentClient(path="./chroma")
    
    try: 
        image_collection = chroma_client.get_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    except: 
        image_collection = chroma_client.create_collection(name=collection_name , metadata={"hnsw:space": "cosine"})
    
    return image_collection

if __name__ == '__main__':
    
    # # For graph RAG
    # graph = initialize_neo4j_rag()

    # # # for removing graphs
    # delete_query = """MATCH (n) DETACH DELETE n"""
    # graph.query(delete_query)

    # # In case you want to loop through files
    # for file in list(glob.glob('documents/json/*.json')):
        
    #     cypher_query = generate_cypher_from_json(file)
        
    #     for query in cypher_query.split(';')[:-1]:
    #         graph.query(query)

    # graph.refresh_schema()

    # print(graph.schema)
    
    # For image RAG
    from glob import glob
    from pathlib import Path
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    
    # # finetune_image: hii-finetuned-clip-v1-model / image: clip-vit-base-patch32
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # image_collection = initialize_chroma_rag("image")

    # finetune_image: hii-finetuned-clip-v1-model / image: clip-vit-base-patch32
    processor = CLIPProcessor.from_pretrained("./models/hii-finetuned-clip-v1-processor")
    model = CLIPModel.from_pretrained("./models/hii-finetuned-clip-v1-model")
    image_collection = initialize_chroma_rag("new_image")  
    
    image_path = Path("./images")
    images = glob(str(image_path / "*"))
    
    image_embedding = []
    ids_index = 1

    for i in images:
        inputs = processor(
            images=Image.open(i),
            return_tensors="pt"
        )
        image_features = model.get_image_features(**inputs).detach().cpu().numpy().tolist()
        image_embedding.append(image_features)
        
        ids_index += 1
        
        image_collection.add(ids= str(ids_index + 1) ,embeddings=image_features, metadatas={"path": i , "name": os.path.basename(i)})