"""
app.py - Visual Math Evaluation Tool with Async API Calls
"""

import asyncio
import base64
import glob
import hmac
import json
import logging
import os
import uuid
from typing import List, Optional, Tuple, Dict, Any

import streamlit as st
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
NUM_QUESTIONS = 5

# Predefined question parts count
QUESTION_PARTS = {
    1: 1,
    2: 3,
    3: 2,
    4: 2,
    5: 2
}

MODELS = [
    # "openai/o1",
    "openai/gpt-4o-mini-2024-07-18",
    "openai/gpt-4o-2024-11-20",
    "google/gemini-2.0-pro-exp-02-05:free",
    "google/gemini-2.0-flash-001",
    "google/learnlm-1.5-pro-experimental:free",
    "anthropic/claude-3.7-sonnet"
]

DEFAULT_SYSTEM_PROMPT = """
Analyse the given Singapore Secondary School math question and any relevant diagram(s) and provide:

1. your understanding of what the question is about and asking for
2. your step-by-step workings to solve this question
3. your final answer (remember to include units, if applicable)
4. your explanation of your approach/workings/answer to a student who may not understand this question
5. a brief hint to a student who may be stuck on this question

Please think step by step.
""".strip()

MODEL_DESCRIPTION = """
- `o1`: OpenAI's main reasoning model, optimized for reasoning (note: `o3-mini` does not support visual inputs)
- `gpt-4o-mini-2024-07-18`: OpenAI's fast/cheap model, same model used for SLS currently
- `gpt-4o-2024-11-20`: OpenAI's latest non-reasoning model, good for most tasks
- `gemini-2.0-pro-exp-02-05:free`: Google's best model
- `gemini-2.0-flash-001`: Google's fast/cheap model (much newer than gpt-4o-mini)
- `learnlm-1.5-pro-experimental:free`: Google's experimental model, fine-tuned for following padegogical instructions given by the user
- `claude-3.7-sonnet`: Anthropic's latest model, good for most tasks. Reasoning is available in the model, but not enabled for this demo yet.
""".strip()

# PASSWORD = os.getenv("PASSWORD")
PASSWORD = os.getenv("PASSWORD")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AsyncOpenAI client
async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# ---------------------------------------------------------
# Data Models
# ---------------------------------------------------------
class WorkStep(BaseModel):
    step: str = Field(..., description="A summary of the step and calculations, or reasoning")

class MathAnalysis(BaseModel):
    understanding: str = Field(..., description="Your understanding of what the question is about and asking for")
    workings: List[WorkStep] = Field(..., description="Your detailed workings")
    answer: str = Field(..., description="Your final answer")
    explanation: str = Field(..., description="Your explanation about this question to a student who may not understand this question")
    hint: str = Field(..., description="A brief hint to help the student get started")

# Dynamic creation of MathAnswer class based on number of parts
def create_math_answer_model(num_parts: int) -> type:
    """Create a dynamic MathAnswer model based on the number of question parts"""
    return create_model(
        'MathAnswer',
        answer_to_each_part=(List[MathAnalysis], Field(..., description=f"The answers to all {num_parts} parts of the question")),
        __base__=BaseModel
    )

# ---------------------------------------------------------
# Authentication Functions
# ---------------------------------------------------------
def check_password() -> bool:
    """Validate user password"""
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], PASSWORD):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Return True if already validated
    if st.session_state.get("password_correct", False):
        return True

    # Show password input
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
        
    return False

# ---------------------------------------------------------
# Image Processing Functions
# ---------------------------------------------------------
def convert_to_base64(image_path: str) -> str:
    """Convert an image file to base64 encoding"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file and return the path"""
    # Create a unique filename
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"uploads/{uuid.uuid4()}{file_extension}"
    
    # Save the file
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return unique_filename

def load_question_data(question_number: int) -> Tuple[str, Optional[str], str, Optional[str], Optional[str], int]:
    """Load question images and text for the specified question number"""
    question_base = f"{question_number:02d}"
    
    # Find the computer-generated image
    computer_image_path = glob.glob(f"questions/{question_base}_*_computer.png")[0]
    
    # Try to find hand-drawn image (if available)
    try:
        hand_image_path = glob.glob(f"questions/{question_base}_*_hand.png")[0]
    except:
        hand_image_path = None
        
    # Try to find text description (if available)
    try:
        text_path = glob.glob(f"questions/{question_base}_*_hand_text.txt")[0]
        with open(text_path) as f:
            question_text = f.read().strip()
    except:
        question_text = None
        
    # Convert images to base64
    computer_image = convert_to_base64(computer_image_path)
    hand_image = convert_to_base64(hand_image_path) if hand_image_path else None
    
    # Get number of parts for this question
    num_parts = QUESTION_PARTS.get(question_number, 1)
    
    return computer_image_path, hand_image_path, computer_image, hand_image, question_text, num_parts

# ---------------------------------------------------------
# Async LLM Functions
# ---------------------------------------------------------
def prepare_llm_args(system_prompt: str, question_image: str, model: str, question_text: Optional[str] = None) -> Dict[str, Any]:
    """Prepare arguments for the LLM API call"""
    user_message = f"{question_text + '. Based on this question and with reference to the above image,' if question_text else ''} Please solve this.".strip()
    
    args = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{question_image}"},
                },
                {
                    "type": "text",
                    "text": user_message
                }          
            ]}
        ]
    }
    
    if model != "openai/o1":
        args["temperature"] = 0
    
    if model == "openai/o1":
        args["reasoning_effort"] = "low"
        
    return args

async def process_openai_model_async(llm_args: Dict[str, Any], math_answer_model: type) -> BaseModel:
    """Process question with OpenAI models asynchronously"""
    llm_args["response_format"] = math_answer_model
    completion = await async_client.beta.chat.completions.parse(**llm_args)
    logger.info(f"Completed API call for model: {llm_args['model']}")
    return completion.choices[0].message.parsed

async def process_other_model_async(llm_args: Dict[str, Any], math_answer_model: type) -> BaseModel:
    """Process question with Anthropic & Google models asynchronously"""
    schema = math_answer_model.model_json_schema()
    updated_system_prompt = f"""
    {llm_args['messages'][0]['content']}

    Please reply in JSON, without codeblocks.
    Use this schema:
    {schema}
    PLEASE REPLY IN VALID JSON.
    """.strip()
    
    llm_args["messages"][0]["content"] = updated_system_prompt
    completions = await async_client.chat.completions.create(**llm_args)
    text = completions.choices[0].message.content.replace("```json", "").replace("```", "")
    
    try:
        return math_answer_model(**json.loads(text))
    except json.JSONDecodeError:
        logger.error(f"JSON decode error from model {llm_args['model']}, raw response: {text[:200]}...")
        # Return a default answer with error info if JSON parsing fails
        default_parts = []
        for _ in range(get_num_parts_from_model(math_answer_model)):
            default_parts.append(
                MathAnalysis(
                    understanding="Error parsing model response",
                    workings=[WorkStep(step="The model did not return valid JSON")],
                    answer="Error",
                    explanation=f"The model {llm_args['model']} did not return a valid JSON response",
                    hint="Please try again or select a different model"
                )
            )
        
        return math_answer_model(answer_to_each_part=default_parts)

def get_num_parts_from_model(math_answer_model: type) -> int:
    """Extract the number of parts from the model's field description"""
    description = math_answer_model.model_fields["answer_to_each_part"].description
    try:
        # Parse something like "The answers to all 3 parts of the question"
        return int(description.split("all ")[1].split(" parts")[0])
    except:
        return 1  # Default to 1 part if parsing fails

async def answer_question_async(system_prompt: str, question_image: str, model: str, math_answer_model: type, question_text: Optional[str] = None) -> BaseModel:
    """Process a math question using the specified LLM model asynchronously"""
    llm_args = prepare_llm_args(system_prompt, question_image, model, question_text)
    
    try:
        if "openai" in model:
            return await process_openai_model_async(llm_args, math_answer_model)
        else:
            return await process_other_model_async(llm_args, math_answer_model)
    except Exception as e:
        logger.error(f"Error processing model {model}: {str(e)}")
        # Return a default answer with error info
        default_parts = []
        for _ in range(get_num_parts_from_model(math_answer_model)):
            default_parts.append(
                MathAnalysis(
                    understanding=f"Error calling model: {model}",
                    workings=[WorkStep(step=f"Error: {str(e)}")],
                    answer="Error",
                    explanation="An error occurred while processing this question",
                    hint="Please try again or select a different model"
                )
            )
        
        return math_answer_model(answer_to_each_part=default_parts)

async def process_all_models_async(system_prompt: str, models: List[str], 
                                  computer_image: str, hand_image: Optional[str], 
                                  question_text: Optional[str], num_parts: int) -> Dict[str, Dict[str, BaseModel]]:
    """Process all selected models concurrently"""
    tasks = []
    
    # Create dynamic MathAnswer model based on number of parts
    math_answer_model = create_math_answer_model(num_parts)
    
    # Create tasks for all models and both image types
    for model in models:
        # Computer image task
        tasks.append((model, "computer", asyncio.create_task(
            answer_question_async(system_prompt, computer_image, model, math_answer_model, question_text)
        )))
        
        # Hand image task (if available)
        if hand_image:
            tasks.append((model, "hand", asyncio.create_task(
                answer_question_async(system_prompt, hand_image, model, math_answer_model, question_text)
            )))
    
    # Wait for all tasks to complete
    results = {}
    for model, image_type, task in tasks:
        if model not in results:
            results[model] = {}
        
        # Await the task result
        try:
            results[model][image_type] = await task
        except Exception as e:
            logger.error(f"Task failed for model {model}, image {image_type}: {str(e)}")
            default_parts = []
            for _ in range(num_parts):
                default_parts.append(
                    MathAnalysis(
                        understanding=f"Task error for model {model}",
                        workings=[WorkStep(step=f"Error: {str(e)}")],
                        answer="Error",
                        explanation="An error occurred while processing this question",
                        hint="Please try again or select a different model"
                    )
                )
            
            results[model][image_type] = math_answer_model(answer_to_each_part=default_parts)
    
    return results

# ---------------------------------------------------------
# UI Display Functions
# ---------------------------------------------------------
def display_question_images(computer_image_path: str, hand_image_path: Optional[str]) -> None:
    """Display the question images side by side"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(computer_image_path, caption="Computer Generated Figure")
        
    with col2:
        if hand_image_path:
            st.image(hand_image_path, caption="Hand Drawn Figure")
            st.write("Note: We also pass the question's text directly to the model, instead of an image")
        else:
            st.write("No hand drawn figure available")

def display_model_response(response: BaseModel) -> None:
    """Display a single model's response"""
    # Display each part of the answer
    for index, part in enumerate(response.answer_to_each_part):
        st.markdown(f"## Part {index+1} of the question")
        
        st.markdown("### 1: Model's Understanding of the Question")
        st.info(part.understanding)
        
        st.markdown("### 2: Model's Workings")
        for step_index, step in enumerate(part.workings):
            st.markdown(f"#### Step {step_index+1}")
            st.info(f"{step.step}")
        
        st.markdown("### 3: Answer")
        st.success(part.answer)
        
        st.markdown("### 4: Explanation")
        st.info(part.explanation)
        
        st.markdown("### 5: Hint")
        st.info(part.hint)

def display_all_model_responses(selected_models: List[str], results: Dict[str, Dict[str, BaseModel]]) -> None:
    """Display responses from all models"""
    st.title("Model Comparisons")
    
    for index, model in enumerate(selected_models):
        st.markdown(f"# {index+1}. {model}")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("## Computer Generated Figure")
            if "computer" in results[model]:
                display_model_response(results[model]["computer"])
                
        with cols[1]:
            if "hand" in results[model]:
                st.markdown("## Hand Drawn Figure")
                display_model_response(results[model]["hand"])

def upload_custom_question() -> Tuple[str, Optional[str], str, Optional[str], Optional[str], int]:
    """UI for uploading a custom question"""
    st.header("Upload Your Own Question")
    
    # Number of parts in the question
    num_parts = st.number_input("Number of parts in this question", min_value=1, max_value=10, value=1)
    
    # Upload computer-generated image (required)
    computer_image_file = st.file_uploader("Upload Computer Generated Figure (Required)", type=["png", "jpg", "jpeg"])
    
    # Upload hand-drawn image (optional)
    hand_image_file = st.file_uploader("Upload Hand Drawn Figure (Optional)", type=["png", "jpg", "jpeg"])
    
    # Question text (optional)
    question_text = st.text_area("Question Text (Optional) - Given together with the hand-drawn figure", height=100)
    
    # Check if required files are uploaded
    if computer_image_file is None:
        st.warning("Please upload a computer-generated figure (required)")
        return None, None, None, None, None, num_parts
    
    # Save uploaded files
    computer_image_path = save_uploaded_file(computer_image_file)
    hand_image_path = save_uploaded_file(hand_image_file) if hand_image_file else None
    
    # Convert to base64
    computer_image_base64 = convert_to_base64(computer_image_path)
    hand_image_base64 = convert_to_base64(hand_image_path) if hand_image_path else None
    
    return computer_image_path, hand_image_path, computer_image_base64, hand_image_base64, question_text, num_parts

def show_main_ui() -> None:
    """Display the main application UI"""
    # Set up the page
    st.set_page_config(page_title="Visual Math Eval", page_icon=":math:", layout="wide")
    st.title("Visual Math Eval Tool")

    with st.expander("About each model"):
        st.write(MODEL_DESCRIPTION)
    
    # Tabs for choosing between pre-loaded and custom questions
    tab1, tab2 = st.tabs(["Pre-loaded Questions", "Upload Your Own Question"])
    
    with tab1:
        # Input controls
        system_prompt = st.text_area(
            "Enter your system prompt", 
            value=DEFAULT_SYSTEM_PROMPT, 
            height=300,
            key="preloaded_system_prompt"
        )
        
        selected_question = st.selectbox(
            "Select a question", 
            range(1, NUM_QUESTIONS + 1)
        )
        
        st.info(f"Question {selected_question} has {QUESTION_PARTS.get(selected_question, 1)} part(s)")
        
        selected_models = st.multiselect(
            "Select models", 
            MODELS
        )
        
        # Process when submit button is clicked
        if st.button("Submit Pre-loaded Question"):
            # Load question data
            computer_image_path, hand_image_path, computer_image, hand_image, question_text, num_parts = (
                load_question_data(selected_question)
            )
            
            # Display question images
            display_question_images(computer_image_path, hand_image_path)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Starting API calls to models...")
            
            # Process all models concurrently
            with st.spinner("Processing all models simultaneously..."):
                # Run the async function in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    process_all_models_async(
                        system_prompt,
                        selected_models,
                        computer_image,
                        hand_image,
                        question_text,
                        num_parts
                    )
                )
                loop.close()
            
            # Update progress to 100%
            progress_bar.progress(100)
            status_text.text("All model responses received!")
            
            # Display all model responses
            display_all_model_responses(selected_models, results)
    
    with tab2:
        system_prompt = st.text_area(
            "Enter your system prompt", 
            value=DEFAULT_SYSTEM_PROMPT, 
            height=200,
            key="custom_system_prompt"
        )
        
        # Upload custom question UI
        custom_question_data = upload_custom_question()
        computer_image_path, hand_image_path, computer_image, hand_image, question_text, num_parts = custom_question_data
        
        selected_models = st.multiselect(
            "Select models", 
            MODELS,
            key="custom_models"
        )
        
        # Process when submit button is clicked
        if st.button("Submit Custom Question"):
            if computer_image is None:
                st.error("Please upload the required computer-generated figure")
            else:
                # Display question images
                display_question_images(computer_image_path, hand_image_path)
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Starting API calls to models...")
                
                # Process all models concurrently
                with st.spinner("Processing all models simultaneously..."):
                    # Run the async function in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        process_all_models_async(
                            system_prompt,
                            selected_models,
                            computer_image,
                            hand_image,
                            question_text,
                            num_parts
                        )
                    )
                    loop.close()
                
                # Update progress to 100%
                progress_bar.progress(100)
                status_text.text("All model responses received!")
                
                # Display all model responses
                display_all_model_responses(selected_models, results)

# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
def main():
    # Check authentication
    if not check_password():
        st.stop()
    
    # Show main UI if authenticated
    show_main_ui()

if __name__ == "__main__":
    main()
