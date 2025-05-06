# MediBot - Advanced Medical Assistant

An AI-powered medical chatbot that provides reliable medical information using Google's Gemini API and Retrieval-Augmented Generation (RAG) technology.

![MediBot Banner](https://img.freepik.com/free-vector/gradient-medical-logo-template_23-2149613549.jpg)

## Features

- **Modern UI**: Dark theme with a sleek interface and intuitive design
- **Gemini API Integration**: Fast, accurate medical responses powered by Google's Gemini models
- **RAG (Retrieval-Augmented Generation)**: Evidence-based answers from a trusted medical knowledge base
- **Medical Entity Extraction**: Automatic identification and visualization of medical entities
- **User Authentication**: Secure login and registration system
- **Chat History Management**: Persistent conversation history for each user
- **Suggested Questions**: Interactive question cards for easy interaction
- **Advanced Visualization**: Charts and tables for medical entity analysis
- **Multiple LLM Support**: Choose between Gemini models and open-source alternatives

## Demo

![MediBot Demo](https://i.imgur.com/example.png)

## Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/app/apikey))
- HuggingFace token (optional, for Mistral and Llama models)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/prakhar1009/ai-medical-chatbot.git
cd ai-medical-chatbot
```

2. **Create a virtual environment**

```bash
python -m venv .venv
```

3. **Activate the virtual environment**

On Windows:
```bash
.venv\Scripts\activate
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements_enhanced.txt
```

5. **Set up environment variables**

Create a `.env` file in the project root with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

6. **Generate the vector database**

```bash
python memory.py
```

7. **Run the application**

```bash
streamlit run medibot_enhanced.py
```

## Usage

1. **Register or Login**
   - Create a new account or log in to an existing one
   - Your chat history will be saved to your account

2. **Configure Settings**
   - Select your preferred AI model (Gemini Flash, Gemini Pro, Mistral-7B, etc.)
   - Adjust temperature for more creative or deterministic responses
   - Enable/disable advanced features like RAG, entity extraction, etc.

3. **Ask Medical Questions**
   - Use the suggestion buttons for common medical queries
   - Type your own questions in the chat input
   - View detailed responses with source citations

4. **Analyze Medical Entities**
   - See extracted medical entities categorized by type
   - View visualizations of entity distribution
   - Explore relationships between medical concepts

## Features in Detail

### 1. Retrieval-Augmented Generation (RAG)

MediBot uses RAG technology to provide evidence-based answers from "The GALE ENCYCLOPEDIA of MEDICINE." This ensures responses are grounded in reliable medical information rather than hallucinated.

### 2. Medical Entity Extraction

The application automatically identifies and categorizes medical entities in responses:
- Diseases
- Symptoms
- Medications
- Treatments
- Procedures
- Body Parts
- Medical Conditions
- Diagnostic Tests

### 3. Multiple Model Support

Choose from various language models:
- **Gemini Flash**: Fast responses for general medical queries
- **Gemini Pro**: More detailed and nuanced medical information
- **Mistral-7B**: Open-source alternative (requires HuggingFace token)
- **Llama-2-7B**: Another open-source option (requires HuggingFace token)

### 4. User Authentication

Secure login and registration system with:
- Password hashing
- Session management
- Persistent chat history

## Project Structure

```
ai-medical-chatbot/
├── data/                      # Medical PDF data
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── vectorstore/               # Vector database and user data
│   ├── db_faiss/              # FAISS vector database
│   ├── user_db.json           # User database
│   └── chat_history/          # User chat histories
├── .env                       # Environment variables (API keys)
├── medibot.py                 # Basic version of the chatbot
├── medibot_enhanced.py        # Enhanced version with Gemini integration
├── memory.py                  # PDF processing and vector creation
├── requirements.txt           # Basic dependencies
├── requirements_enhanced.txt  # Enhanced version dependencies
└── README_enhanced.md         # This documentation
```

## Troubleshooting

### Common Issues

#### "Failed to load the vector store" Error

This usually means the vector database hasn't been properly initialized.
- Ensure the `vectorstore/db_faiss` directory exists
- Run the `memory.py` script to generate the vector database

#### Authentication Issues

If you're having trouble logging in:
- Make sure you're using the correct username and password
- Check if the `vectorstore/user_db.json` file exists
- Try registering a new account

#### API Key Errors

If you see "Gemini API key not found" error:
- Make sure your `.env` file contains the correct `GEMINI_API_KEY` value
- Ensure the API key is valid and has not expired
- Check that the `python-dotenv` package is installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- The GALE ENCYCLOPEDIA of MEDICINE for the medical knowledge base
- Google for the Gemini API
- HuggingFace for providing access to state-of-the-art language models
- LangChain for the RAG framework
