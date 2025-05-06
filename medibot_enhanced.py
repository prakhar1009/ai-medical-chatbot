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
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Check if GEMINI_API_KEY is available
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Gemini API integration
def call_gemini_api(user_question: str, gemini_api_key: Optional[str] = None, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Calls Gemini API with a medical prompt and conversation history.
    """
    if gemini_api_key is None:
        # Use the global GEMINI_API_KEY variable
        gemini_api_key = GEMINI_API_KEY
        
        # Check if key exists and is not empty
        if not gemini_api_key:
            return "❌ Error: Gemini API key not found. Please add your GEMINI_API_KEY to the .env file. See the sample.env file for reference."
    
    # Use the model name parameter if provided, otherwise use a default
    # We'll get the actual model from session state when the function is called from within the app
    model_name = "gemini-pro"  # Default fallback model
    
    # Use the correct API endpoint format for the selected model
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"
    
    # Convert message history to Gemini format
    gemini_messages = []
    if history:
        for msg in history[-10:]:  # Only use last 10 messages for context
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
    
    # Add the current question
    gemini_messages.append({
        "role": "user",
        "parts": [{"text": user_question}]
    })
    
    # Strong anti-hallucination, medical-only prompt
    system_prompt = """
    You are MediBot, a highly reliable medical assistant. ONLY answer questions that are strictly related to medical, healthcare, or scientific reference topics. 
    If the user's question is not related to medicine, healthcare, or scientific reference, politely tell the user: 
    'This application only answers medical or healthcare-related questions based on trusted references. Please ask a relevant question.'
    
    When you answer, use only factual, verifiable, and up-to-date information. NEVER make up facts or hallucinate. 
    If you are unsure or cannot answer factually, respond: 'I do not know the answer to that based on trusted medical sources.'
    
    Format your responses in markdown for better readability. Use bullet points, headers, and emphasis where appropriate.
    """
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": system_prompt}]
            }
        ] + gemini_messages,
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 2048
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI  # Added for Gemini integration

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
USER_DB_PATH = "vectorstore/user_db.json"
CHAT_HISTORY_PATH = "vectorstore/chat_history"

# Available models
MODELS = {
    "Gemini Pro": "gemini-pro",
    "Gemini Flash": "gemini-2.0-flash",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-2-7B": "meta-llama/Llama-2-7b-chat-hf"
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
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    if "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY,
            convert_system_message_to_human=True
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

# Function to display suggested questions
def display_suggested_questions():
    st.markdown("### Start with a question:")
    
    # Custom CSS for better-looking suggestion buttons
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: auto;
        padding: 15px;
        text-align: left;
        background-color: #2d2d2d;
        color: white;
        border: none;
        border-radius: 10px;
        margin: 5px 0;
    }
    div.stButton > button:hover {
        background-color: #3d3d3d;
        border-left: 4px solid #4a6bdf;
    }
    div.stButton > button p {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    suggestions = [
        {"icon": "🩺", "text": "What are the symptoms of diabetes?", "color": "#4a6bdf"},
        {"icon": "💊", "text": "How do antibiotics work?", "color": "#28a745"},
        {"icon": "🧠", "text": "Explain how vaccines prevent disease", "color": "#dc3545"},
        {"icon": "❤️", "text": "What causes high blood pressure?", "color": "#fd7e14"}
    ]
    
    cols = st.columns(2)
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % 2]:
            button_label = f"{suggestion['icon']}  {suggestion['text']}"
            if st.button(button_label, key=f"suggestion_{idx}"):
                st.session_state["chat_input"] = suggestion['text']
                st.session_state["suggestion_triggered"] = True
                # Force a rerun to immediately process the suggestion
                st.experimental_rerun()

# Helper to process suggestion click
def process_suggestion():
    if st.session_state.get("suggestion_triggered", False):
        # Get the suggested question from session state
        prompt = st.session_state.get("chat_input", "")
        if prompt:
            # Add user message to chat
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process the suggestion as a user input
            process_user_input(prompt)
            
        # Reset the suggestion trigger
        st.session_state["suggestion_triggered"] = False
        
# Helper to process user input (used by both direct input and suggestions)
def process_user_input(prompt):
    # Display assistant "thinking" message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")
        
        try:
            # Get selected model
            model_name = MODELS[st.session_state.current_model]
            
            # For Gemini models, use direct API call
            if "gemini" in model_name.lower():
                # Get chat history for context
                history = [msg for msg in st.session_state.messages[:-1]]  # Exclude the most recent user message
                
                # Call Gemini API
                result = call_gemini_api(prompt, GEMINI_API_KEY, history)
                
                # Extract medical entities if enabled
                entities = {}
                if st.session_state.extract_entities:
                    # Load LLM for entity extraction
                    llm = load_llm(model_name, st.session_state.temperature)
                    entities = extract_medical_entities(result, llm)
                
                # Update placeholder with result
                message_placeholder.markdown(result)
                
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": result,
                    "entities": entities
                }
                st.session_state.messages.append(assistant_message)
                
                # Save chat history
                save_chat_history(st.session_state.username, st.session_state.messages)
            
            # For other models, use RAG with LangChain
            else:
                # Get vectorstore
                vectorstore = get_vectorstore()
                
                # Load LLM
                llm = load_llm(model_name, st.session_state.temperature)
                
                # Create retriever based on settings
                if st.session_state.use_advanced_rag:
                    retriever = get_advanced_retriever(vectorstore, llm)
                else:
                    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
                
                # Set up custom prompt template
                template = """Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Always cite your sources by indicating which document (Doc 1, Doc 2, etc.) contains the information.
                
                {context}
                
                Question: {question}
                
                Answer in a detailed, medically accurate way:"""
                
                # Create QA chain
                qa_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | set_custom_prompt(template)
                    | llm
                    | StrOutputParser()
                )
                
                # Run chain
                result = qa_chain.invoke(prompt)
                
                # Extract medical entities if enabled
                entities = {}
                if st.session_state.extract_entities:
                    entities = extract_medical_entities(result, llm)
                
                # Update placeholder with result
                message_placeholder.markdown(result)
                
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": result,
                    "entities": entities
                }
                st.session_state.messages.append(assistant_message)
                
                # Save chat history
                save_chat_history(st.session_state.username, st.session_state.messages)
        
        except Exception as e:
            error_message = f"Error: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            save_chat_history(st.session_state.username, st.session_state.messages)

def main():
    # Page configuration and styling
    st.set_page_config(page_title="MediBot - Advanced Medical Assistant", page_icon="💊", layout="wide")

    # Warn if GEMINI_API_KEY is missing
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY is missing. Please set it in your .env file or environment variables.")
    
    # Apply custom CSS for modern theme (similar to financial assistant)
    st.markdown("""
    <style>
    /* Main app theme - dark mode */
    .main {background-color: #121212;}
    .stApp {background-color: #121212; color: #ffffff;}
    
    /* Input fields styling */
    .stTextInput>div>div>input {background-color: #2d2d2d; color: #ffffff; border: 1px solid #444444; border-radius: 8px; padding: 10px 14px;}
    .stTextInput>label {font-weight: bold; color: #ffffff;}
    
    /* Button styling */
    .stButton>button {background-color: #4a6bdf; color: white; font-weight: bold; border-radius: 8px; padding: 10px 18px; transition: all 0.3s;}
    .stButton>button:hover {background-color: #3a5bcf; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
    
    /* Sidebar styling */
    .stSidebar {background-color: #1e1e1e;}
    
    /* Headings */
    h1 {color: #ffffff; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);}
    h2 {color: #ffffff; font-weight: bold;}
    h3 {color: #ffffff;}
    
    /* Chat container styling */
    .chat-container {max-width: 800px; margin: 0 auto;}
    
    /* Message styling */
    .stChatMessage {border-radius: 15px; margin-bottom: 10px;}
    .stChatMessage.user {background-color: #4a6bdf;}
    .stChatMessage.assistant {background-color: #2d2d2d;}
    
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
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
        st.session_state.current_model = "Gemini Flash"
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
        st.markdown("<h1 style='text-align: center;'>💊 MediBot - Advanced Medical Assistant</h1>", unsafe_allow_html=True)
        
        # Display logo with glowing effect
        st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 30px;">
            <div style="background-color: #4a6bdf; width: 120px; height: 120px; border-radius: 50%; 
            display: flex; justify-content: center; align-items: center; box-shadow: 0 0 30px #4a6bdf;">
                <span style="font-size: 60px;">💊</span>
            </div>
        </div>
        <h2 style="text-align: center; margin-bottom: 30px;">Your AI Medical Assistant</h2>
        """, unsafe_allow_html=True)
        
        # If no messages yet, show suggested questions
        if not st.session_state.messages:
            display_suggested_questions()
            process_suggestion()
        
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
                            entity_counts.plot(kind='bar', ax=ax, color='#4a6bdf')
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
            
            # Process the user input
            process_user_input(prompt)

if __name__ == "__main__":
    main()
