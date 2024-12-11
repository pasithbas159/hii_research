
# HII Reserach Application


## Installation Guide

#### 1. Download model file and save to models/...

#### 2. Have .env file with the following configuration

```http
  LLAMA_CLOUD_API_KEY=
  ANTHROPIC_API_KEY=
  OPENAI_API_KEY=
  LANGCHAIN_API_KEY=
  LANGCHAIN_API_KEY_GRAPH=
  NEO4J_URI=
  NEO4J_USERNAME=
  NEO4J_PASSWORD=
  AURA_INSTANCEID=
  AURA_INSTANCENAME=
```

#### 3. Install python requirements

```http
  pip install -r requirements.txt
```

#### 4. Run streamlit via main.py

```http
  streamlit run main.py
```