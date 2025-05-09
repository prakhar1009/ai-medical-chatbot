import os
import streamlit as st
import json
import hashlib
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from PIL import Image
from io import BytesIO
import requests
from typing import Optional

# Gemini fallback function
def call_gemini_api(user_question: str, gemini_api_key: Optional[str] = None) -> str:
    """
    Calls Gemini API with a strict, fact-based, medical-only prompt. If the question is not medical, Gemini should reply with a guidance message.
    """
    if gemini_api_key is None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    # Strong anti-hallucination, medical-only prompt
    gemini_prompt = f"""
    You are a highly reliable medical assistant. ONLY answer questions that are strictly related to medical, healthcare, or scientific reference topics. If the user's question is not related to medicine, healthcare, or scientific reference, do not answer it—instead, politely tell the user: 'This application only answers medical or healthcare-related questions based on trusted references. Please ask a relevant question.'
    
    When you answer, use only factual, verifiable, and up-to-date information. NEVER make up facts or hallucinate. If you are unsure or cannot answer factually, respond: 'I do not know the answer to that based on trusted medical sources.'
    
    User's question: {user_question}
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": gemini_prompt}]}]
    }
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Gemini returns candidates list
        if "candidates" in data and data["candidates"]:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return "[Gemini did not return a valid answer.]"
    except Exception as e:
        return f"[Error fetching Gemini response: {e}]"

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
USER_DB_PATH = "vectorstore/user_db.json"
CHAT_HISTORY_PATH = "vectorstore/chat_history"

# Available models
MODELS = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-2-7B": "meta-llama/Llama-2-7b-chat-hf",
    "GPT-3.5": "openai/gpt-3.5-turbo"
}

# Medical entity categories for extraction
MEDICAL_ENTITIES = [
    "Disease", "Symptom", "Medication", "Treatment", "Procedure", 
    "Body_Part", "Medical_Condition", "Diagnostic_Test"
]

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_user_db():
    if not os.path.exists(USER_DB_PATH):
        os.makedirs(os.path.dirname(USER_DB_PATH), exist_ok=True)
        with open(USER_DB_PATH, 'w') as f:
            json.dump({}, f)
    
    with open(USER_DB_PATH, 'r') as f:
        return json.load(f)

def save_user_db(user_db):
    with open(USER_DB_PATH, 'w') as f:
        json.dump(user_db, f)

def authenticate(username, password):
    user_db = load_user_db()
    if username in user_db and user_db[username]['password'] == hash_password(password):
        return True
    return False

def register_user(username, password, email):
    user_db = load_user_db()
    if username in user_db:
        return False
    
    user_db[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.datetime.now().isoformat(),
        'chat_history_file': f"{CHAT_HISTORY_PATH}/{username}.json"
    }
    save_user_db(user_db)
    
    # Create user chat history file
    os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
    if not os.path.exists(f"{CHAT_HISTORY_PATH}/{username}.json"):
        with open(f"{CHAT_HISTORY_PATH}/{username}.json", 'w') as f:
            json.dump([], f)
    
    return True

# Chat history functions
def save_chat_history(username, messages):
    user_db = load_user_db()
    if username in user_db:
        chat_file = user_db[username]['chat_history_file']
        with open(chat_file, 'w') as f:
            json.dump(messages, f)

def load_chat_history(username):
    user_db = load_user_db()
    if username in user_db:
        chat_file = user_db[username]['chat_history_file']
        if os.path.exists(chat_file):
            with open(chat_file, 'r') as f:
                return json.load(f)
    return []

def load_llm(model_name, temperature=0.5, max_length=512):
    """Load language model based on selection"""
    HF_TOKEN = os.environ.get("HF_TOKEN")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    if "openai" in model_name.lower():
        return ChatOpenAI(
            model_name=model_name.split('/')[-1],
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        return HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            temperature=temperature,
            model_kwargs={
                "token": HF_TOKEN,
                "max_length": str(max_length)
            }
        )

# Advanced RAG with contextual compression
def get_advanced_retriever(vectorstore, llm):
    """Create an advanced retriever with contextual compression"""
    base_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compression_retriever

# Medical entity extraction
def extract_medical_entities(text, llm):
    """Extract medical entities from text using LLM"""
    entity_prompt = PromptTemplate(
        template="""Extract all medical entities from the following text and categorize them. 
        Return the result as a JSON object with entity types as keys and lists of entities as values.
        Entity types to extract: Disease, Symptom, Medication, Treatment, Procedure, Body_Part, Medical_Condition, Diagnostic_Test
        
        Text: {text}
        
        JSON Output:""",
        input_variables=["text"]
    )
    
    chain = entity_prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"text": text})
        # Find JSON in the response
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            entities = json.loads(json_str)
            return entities
        return {}
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return {}

def main():
    # Page configuration and styling
    st.set_page_config(page_title="MediBot - Advanced Medical Assistant", page_icon="💊", layout="wide")
    
    # Apply custom CSS for medical theme
    st.markdown("""
    <style>
    /* Main app theme */
    .main {background-color: #f0f8ff;}
    .stApp {background-color: #f0f8ff;}
    
    /* Input fields styling */
    .stTextInput>div>div>input {background-color: #ffffff; color: #000000; border: 2px solid #1e90ff; border-radius: 5px; padding: 8px 12px;}
    .stTextInput>label {font-weight: bold; color: #0066cc;}
    
    /* Button styling */
    .stButton>button {background-color: #1e90ff; color: white; font-weight: bold; border-radius: 5px; padding: 8px 16px; transition: all 0.3s;}
    .stButton>button:hover {background-color: #0056b3; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    
    /* Sidebar styling */
    .stSidebar {background-color: #e6f2ff;}
    .css-1d391kg {background-color: #e6f2ff;}
    
    /* Headings */
    h1 {color: #0066cc; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);}
    h2 {color: #0066cc; font-weight: bold;}
    h3 {color: #0066cc;}
    
    /* Login/Register container styling */
    .auth-container {background-color: #1a1a1a; border-radius: 10px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); max-width: 500px; margin: 0 auto;}
    .auth-container h2 {color: #ffffff; text-align: center; margin-bottom: 20px;}
    .auth-container .stTextInput>div>div>input {background-color: #333333; color: #ffffff; border: 1px solid #555555;}
    .auth-container .stTextInput>label {color: #ffffff;}
    .auth-container .stButton>button {width: 100%; margin-top: 10px;}
    .auth-tabs {margin-bottom: 20px;}
    .auth-message {padding: 10px; border-radius: 5px; margin-top: 10px;}
    .auth-success {background-color: rgba(40, 167, 69, 0.2); border: 1px solid #28a745; color: #28a745;}
    .auth-error {background-color: rgba(220, 53, 69, 0.2); border: 1px solid #dc3545; color: #dc3545;}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "Mistral-7B"
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.5
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'use_advanced_rag' not in st.session_state:
        st.session_state.use_advanced_rag = True
    if 'extract_entities' not in st.session_state:
        st.session_state.extract_entities = True
    
    # Authentication section
    if not st.session_state.authenticated:
        st.markdown("<h1 style='text-align: center;'>🏥 MediBot - Advanced Medical Assistant</h1>", unsafe_allow_html=True)
        
        # Display medical logo or image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                response = requests.get("https://img.freepik.com/free-vector/gradient-medical-logo-template_23-2149613549.jpg")
                img = Image.open(BytesIO(response.content))
                st.image(img, width=300)
            except:
                st.markdown("### 🏥 AI-Powered Medical Assistant")
        
        # Create a container with the dark theme for authentication
        st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
        
        # Login/Register tabs with custom styling
        st.markdown("<div class='auth-tabs'>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.markdown("<h2>Welcome Back</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: #cccccc; text-align: center; margin-bottom: 20px;'>Log in to access your medical assistant</p>", unsafe_allow_html=True)
            
            login_username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            # Remember me checkbox
            col1, col2 = st.columns([1, 1])
            with col1:
                remember_me = st.checkbox("Remember me", key="remember_me")
            with col2:
                st.markdown("<p style='text-align: right; color: #1e90ff;'><a href='#' style='color: #1e90ff; text-decoration: none;'>Forgot password?</a></p>", unsafe_allow_html=True)
            
            login_button = st.button("Sign In", use_container_width=True)
            
            if login_button:
                if authenticate(login_username, login_password):
                    st.session_state.authenticated = True
                    st.session_state.username = login_username
                    st.session_state.messages = load_chat_history(login_username)
                    st.experimental_rerun()
                else:
                    st.markdown("<div class='auth-message auth-error'>Invalid username or password</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<h2>Create Account</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: #cccccc; text-align: center; margin-bottom: 20px;'>Join us to get personalized medical information</p>", unsafe_allow_html=True)
            
            reg_username = st.text_input("Username", key="reg_username", placeholder="Choose a username")
            reg_email = st.text_input("Email", key="reg_email", placeholder="Enter your email address")
            reg_password = st.text_input("Password", type="password", key="reg_password", placeholder="Create a password")
            reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password", placeholder="Confirm your password")
            
            # Terms and conditions checkbox
            terms_agree = st.checkbox("I agree to the Terms and Conditions", key="terms_agree")
            
            reg_button = st.button("Sign Up", use_container_width=True)
            
            if reg_button:
                if not terms_agree:
                    st.markdown("<div class='auth-message auth-error'>Please agree to the Terms and Conditions</div>", unsafe_allow_html=True)
                elif reg_password != reg_confirm_password:
                    st.markdown("<div class='auth-message auth-error'>Passwords do not match</div>", unsafe_allow_html=True)
                elif not reg_username or not reg_email or not reg_password:
                    st.markdown("<div class='auth-message auth-error'>All fields are required</div>", unsafe_allow_html=True)
                else:
                    if register_user(reg_username, reg_password, reg_email):
                        st.markdown("<div class='auth-message auth-success'>Registration successful! Please login.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='auth-message auth-error'>Username already exists</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close auth-tabs div
        st.markdown("</div>", unsafe_allow_html=True)  # Close auth-container div
    
    else:  # User is authenticated
        # Sidebar for settings and options
        with st.sidebar:
            st.markdown(f"### Welcome, {st.session_state.username}! 👋")
            st.divider()
            
            st.subheader("Model Settings")
            st.session_state.current_model = st.selectbox(
                "Select AI Model", 
                list(MODELS.keys()),
                index=list(MODELS.keys()).index(st.session_state.current_model)
            )
            
            st.session_state.temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.temperature,
                step=0.1
            )
            
            st.divider()
            st.subheader("Advanced Features")
            st.session_state.show_sources = st.checkbox(
                "Show Source Documents", 
                value=st.session_state.show_sources
            )
            
            st.session_state.use_advanced_rag = st.checkbox(
                "Use Advanced RAG", 
                value=st.session_state.use_advanced_rag,
                help="Uses contextual compression for better retrieval"
            )
            
            st.session_state.extract_entities = st.checkbox(
                "Extract Medical Entities", 
                value=st.session_state.extract_entities,
                help="Extracts and visualizes medical entities from responses"
            )
            
            st.divider()
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                save_chat_history(st.session_state.username, [])
                st.experimental_rerun()
            
            if st.button("Logout"):
                # Save chat history before logout
                save_chat_history(st.session_state.username, st.session_state.messages)
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.messages = []
                st.experimental_rerun()
        
        # Main chat interface
        st.markdown("<h1 style='text-align: center;'>🏥 MediBot - Advanced Medical Assistant</h1>", unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
                # Display entity visualization if available and this is an assistant message
                if message['role'] == 'assistant' and 'entities' in message and message['entities']:
                    st.divider()
                    st.markdown("### 📊 Medical Entities Detected")
                    
                    # Create a DataFrame for visualization
                    entity_data = []
                    for entity_type, entities in message['entities'].items():
                        if entities:  # Only include non-empty entity types
                            for entity in entities:
                                entity_data.append({"Type": entity_type, "Entity": entity})
                    
                    if entity_data:
                        df = pd.DataFrame(entity_data)
                        
                        # Display as table
                        st.dataframe(df, use_container_width=True)
                        
                        # Create a bar chart of entity counts
                        entity_counts = df['Type'].value_counts()
                        if not entity_counts.empty:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            entity_counts.plot(kind='bar', ax=ax, color='skyblue')
                            plt.title('Medical Entity Distribution')
                            plt.xlabel('Entity Type')
                            plt.ylabel('Count')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
        
        # Chat input
        prompt = st.chat_input("Ask me about any medical topic...")
        
        if prompt:
            # Add user message to chat
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display assistant "thinking" message
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🤔 Thinking...")
                
                try:
                    # Get selected model
                    model_name = MODELS[st.session_state.current_model]
                    
                    # Load vectorstore
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        raise Exception("Failed to load the vector store")
                    
                    # Load LLM
                    llm = load_llm(
                        model_name=model_name,
                        temperature=st.session_state.temperature
                    )
                    
                    # Custom prompt template
                    CUSTOM_PROMPT_TEMPLATE = """
                    You are MediBot, an advanced medical assistant with expertise in healthcare and medicine.
                    Use the pieces of information provided in the context to answer the user's question.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    Only provide information that is supported by the given context.
                    
                    Context: {context}
                    Question: {question}
                    
                    Start the answer directly. Be concise but thorough. Use medical terminology appropriately.
                    """
                    
                    # Get retriever (basic or advanced)
                    if st.session_state.use_advanced_rag:
                        retriever = get_advanced_retriever(vectorstore, llm)
                    else:
                        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
                    
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )
                    
                    # Get response
                    start_time = time.time()
                    response = qa_chain.invoke({"query": prompt})
                    end_time = time.time()
                    
                    # Extract result and sources
                    result = response["result"]
                    source_documents = response["source_documents"]
                    
                    # Format source documents nicely
                    sources_text = ""
                    if st.session_state.show_sources and source_documents:
                        sources_text = "\n\n**Sources:**\n"
                        for i, doc in enumerate(source_documents):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'Unknown')
                            sources_text += f"- Source {i+1}: {os.path.basename(source)}, Page {page}\n"
                    
                    # Extract medical entities if enabled
                    entities = {}
                    if st.session_state.extract_entities:
                        entities = extract_medical_entities(result, llm)
                    
                    # Compile final result
                    result_to_show = f"{result}{sources_text}\n\n*Response time: {end_time - start_time:.2f} seconds*"

                    # If RAG answer is empty, 'I do not know', or similar, use Gemini fallback
                    fallback_phrases = [
                        "i do not know", "don't know", "do not know", "sorry", "unable to answer", "cannot answer", "no answer", "not sure", "not found", "unknown", "n/a", "out of the given context"
                    ]
                    use_gemini = False
                    if not result.strip() or any(phrase in result.lower() for phrase in fallback_phrases):
                        use_gemini = True
                    # If no source documents are found, also fallback
                    if not source_documents:
                        use_gemini = True
                    if use_gemini:
                        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
                        gemini_response = call_gemini_api(prompt, gemini_api_key)
                        result_to_show = f"{gemini_response}\n\n*Response from Gemini (fallback model)*"
                        entities = extract_medical_entities(gemini_response, llm) if st.session_state.extract_entities else {}

                    # Update placeholder with result
                    message_placeholder.markdown(result_to_show)
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": result_to_show,
                        "entities": entities
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Save chat history
                    save_chat_history(st.session_state.username, st.session_state.messages)
                    
                    # Display entity visualization if entities were extracted
                    if entities:
                        st.divider()
                        st.markdown("### 📊 Medical Entities Detected")
                        
                        # Create a DataFrame for visualization
                        entity_data = []
                        for entity_type, entity_list in entities.items():
                            if entity_list:  # Only include non-empty entity types
                                for entity in entity_list:
                                    entity_data.append({"Type": entity_type, "Entity": entity})
                        
                        if entity_data:
                            df = pd.DataFrame(entity_data)
                            
                            # Display as table
                            st.dataframe(df, use_container_width=True)
                            
                            # Create a bar chart of entity counts
                            entity_counts = df['Type'].value_counts()
                            if not entity_counts.empty:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                entity_counts.plot(kind='bar', ax=ax, color='skyblue')
                                plt.title('Medical Entity Distribution')
                                plt.xlabel('Entity Type')
                                plt.ylabel('Count')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    message_placeholder.markdown(f"❌ {error_msg}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()