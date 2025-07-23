# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
import os
import re # Import the regular expression module

# Initialize FastAPI app
app = FastAPI()

# Mount static files (for CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates for serving HTML
templates = Jinja2Templates(directory="templates")

# --- Configure Google Generative AI API ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual API key if running locally outside Canvas.
# For deployment, it's best practice to use environment variables.
# You can get an API key from Google AI Studio: https://aistudio.google.com/
# For Canvas, the API key is automatically provided if left as an empty string.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBLV5TuW5-aJubAR-GGsaZzrxlFriYhSg8") # Keep as empty string for Canvas
genai.configure(api_key=GEMINI_API_KEY)

# --- Initialize the Generative Model with improved settings ---
# Using gemini-2.0-flash for faster responses, suitable for conversational bots.
model = genai.GenerativeModel(
    'gemini-2.0-flash',
    # System instruction defines the bot's persona and how it should respond
    system_instruction=(
        "You are a helpful and encouraging Study Mate bot designed to assist students with their academic queries. "
        "Provide clear, concise, and easy-to-understand explanations. "
        "Break down complex topics into simple steps or bullet points using standard Markdown (`*` or `-` for lists, `**text**` for bold). "
        "Avoid redundant numbering or excessive use of asterisks for emphasis. " # Added explicit instruction
        "Offer practical study tips and resources. "
        "Keep your responses focused on the student's learning needs and avoid overly long or conversational tangents. "
        "Be supportive and positive."
    ),
    # generation_config helps control the output style (e.g., creativity, length)
    generation_config=genai.types.GenerationConfig(
        candidate_count=1,          # Request only one response candidate
        max_output_tokens=300,      # Limit the length of the response for conciseness
        temperature=0.7,            # Controls creativity; 0.7 is a good balance for informative tasks
        top_p=0.95,                 # Controls diversity; higher means more diverse
        top_k=60                    # Controls diversity; higher means more diverse
    ),
    # safety_settings help filter out potentially harmful content
    safety_settings={
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
)

# Pydantic model for incoming chat messages
class ChatMessage(BaseModel):
    message: str
    # chat_history will be a list of dictionaries, e.g.,
    # [{"role": "user", "parts": "Hello"}, {"role": "model", "parts": "Hi there!"}]
    chat_history: list = []

# --- Helper function to clean up bot responses ---
def clean_bot_response(text: str) -> str:
    """
    Cleans up common formatting issues from Gemini API responses.
    Removes redundant asterisks and numerical prefixes that might appear.
    """
    # Remove leading numbers like "**1." or "**2."
    text = re.sub(r'\*\*\d+\.\s*', '', text)
    # Replace multiple asterisks with single ones for bolding, then remove any remaining single asterisks
    text = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', text) # Convert **text** to <b>text</b> for HTML bolding
    text = text.replace('*', '') # Remove any remaining single asterasterisks
    return text

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML page for the chat interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """
    Handles incoming chat messages, sends them to the Gemini model,
    and returns the model's response.
    """
    try:
        # Prepare the chat history for the model
        # The API expects history in a specific format:
        # [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]
        formatted_history = []
        for entry in chat_message.chat_history:
            # Ensure 'parts' is always a list of dictionaries with 'text' key
            # This handles cases where 'parts' might just be a string from the frontend
            if isinstance(entry["parts"], str):
                formatted_history.append({"role": entry["role"], "parts": [{"text": entry["parts"]}]})
            else: # Assume it's already in the correct list of dicts format
                formatted_history.append({"role": entry["role"], "parts": entry["parts"]})


        # Start a new chat session with the model, providing the history
        chat_session = model.start_chat(history=formatted_history)

        # Send the new user message to the chat session
        response = chat_session.send_message(chat_message.message)

        # Extract the text from the response
        bot_response = response.text

        # --- Apply cleaning to the bot response ---
        cleaned_response = clean_bot_response(bot_response)
        # -------------------------------------------

        return {"response": cleaned_response}
    except Exception as e:
        # --- IMPORTANT DEBUGGING LINE ADDED HERE ---
        print(f"Error processing chat: {e}")
        # -------------------------------------------
        return {"response": "Oops! Something went wrong. Please try again."}

# To run this FastAPI application locally:
# 1. Save this code as `main.py`.
# 2. Create a folder named `templates` in the same directory.
# 3. Inside `templates`, create an `index.html` file (we'll provide this next).
# 4. Create a folder named `static` in the same directory.
# 5. Open your terminal in the project directory.
# 6. Install dependencies: `pip install fastapi uvicorn google-generativeai python-multipart jinja2`
# 7. Run the app: `uvicorn main:app --reload`
# 8. Open your browser to `http://127.0.0.1:8000`
