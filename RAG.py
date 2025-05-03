import google.generativeai as genai
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import time
from tenacity import retry, stop_after_attempt, wait_exponential

def setup_gemini_api(api_key):
    """Initialize Gemini Flash 1.5 API"""
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DB_FILE = "vector_database.pkl"
TEXT_DATA_FILE = "text_data.pkl"
NUTRITION_DATA_FILE = "nutrition_data.pkl"

def load_nutrition_and_exercise_data(nutrition_path, exercise_path, physiotherapy_path):
    """Load nutrition, exercise, and physiotherapy data from CSV/XLSX files."""
    try:
        # Load nutrition data
        nutrition_df = pd.read_excel(nutrition_path)
        nutrition_texts = nutrition_df.apply(
            lambda x: f"{x['name']}: {x['calories']} calories, {x['total_fat']}g fat, {x['protein']}g protein, {x['carbohydrate']}g carbs per {x['serving_size']}g", 
            axis=1).tolist()
        
        # Load exercise data
        exercise_df = pd.read_csv(exercise_path)
        exercise_texts = exercise_df.apply(
            lambda x: f"{x['name']}: {x['muscle_group']}, {x['exercise_type']}, {x['difficulty']}, {x['equipment']}. Instructions: {x['instructions']}", 
            axis=1).tolist()    
        
        # Load physiotherapy exercises data
        physiotherapy_df = pd.read_csv(physiotherapy_path)
        physiotherapy_texts = physiotherapy_df.apply(
            lambda x: f"{x['Name']}: {x['Description']}, Muscles Involved: {x['Muscles Involved']}, Related Conditions: {x['Related Conditions']}, Structures Involved: {x['Structures Involved']}",
            axis=1).tolist()
        
        # Save nutrition data separately
        with open(NUTRITION_DATA_FILE, "wb") as f:
            pickle.dump(nutrition_df, f)
        
        return nutrition_texts + exercise_texts + physiotherapy_texts
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []

def load_nutrition_data():
    """Load nutrition data from file."""
    try:
        if os.path.exists(NUTRITION_DATA_FILE):
            with open(NUTRITION_DATA_FILE, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading nutrition data: {str(e)}")
    return None

def build_vector_database(text_data):
    """Create a FAISS vector database from text data and save it."""
    try:
        embeddings = embedding_model.encode(text_data, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save vector database and text data
        with open(VECTOR_DB_FILE, "wb") as f:
            pickle.dump(index, f)
        with open(TEXT_DATA_FILE, "wb") as f:
            pickle.dump(text_data, f)
        
        return index
    except Exception as e:
        print(f"Error building vector database: {str(e)}")
        return None

def load_vector_database():
    """Load FAISS vector database from file."""
    try:
        if os.path.exists(VECTOR_DB_FILE) and os.path.exists(TEXT_DATA_FILE):
            with open(VECTOR_DB_FILE, "rb") as f:
                index = pickle.load(f)
            with open(TEXT_DATA_FILE, "rb") as f:
                text_data = pickle.load(f)
            return index, text_data
    except Exception as e:
        print(f"Error loading vector database: {str(e)}")
    return None, None

def retrieve_relevant_entry(query, text_data, index, top_n=2):
    """Retrieve relevant entries using FAISS vector search"""
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_n)
        results = [(text_data[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results
    except Exception as e:
        print(f"Error retrieving entries: {str(e)}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_gemini_flash(user_query, text_data, index):
    """Retrieve relevant data and generate a response using Gemini Flash 1.5"""
    try:
        # Step 1: Retrieve relevant results
        retrieved_data = retrieve_relevant_entry(user_query, text_data, index, top_n=3)  # Increased to 3 results
        if not retrieved_data:
            return "Unable to retrieve relevant information. Please try again."
        
        # Step 2: Load nutrition data only if needed
        nutrition_df = None
        relevant_foods = ""
        if any(keyword in user_query.lower() for keyword in ["calorie", "nutrition", "food", "diet", "meal"]):
            nutrition_df = load_nutrition_data()
            if nutrition_df is not None:
                # Increased to 8 relevant food items
                relevant_foods = nutrition_df[["name", "calories", "protein", "total_fat", "carbohydrate"]].head(8).to_string(index=False)
        
        # Step 3: Construct more detailed prompt
        context_data = "\n".join([f"- {item[0]}" for item in retrieved_data])
        prompt = f"""User Query: "{user_query}"

        Data Provided:
        - General Nutrition Information: {context_data}
        - Detailed Nutrition Values for Each Food Item: {relevant_foods}

        Task:
        Using the provided nutrition data, generate a personalized diet plan that is tailored to the user's query and dietary goals. The diet plan should:

        1. **Directly Address the Query:** Clearly respond to the user's specific dietary needs or goals (e.g., weight loss, muscle gain, improved energy, managing health conditions).
        2. **Leverage Nutrition Data:** Incorporate the detailed nutrition values of each food item to create balanced meals, ensuring appropriate distribution of macronutrients (proteins, carbohydrates, fats) and micronutrients.
        3. **Meal Structure:** Outline a daily meal plan, including suggestions for breakfast, lunch, dinner, and snacks if applicable. Provide portion recommendations where possible.
        4. **Key Details & Explanations:** Explain the rationale behind food choices and meal timing. Highlight how the nutritional content supports the user's goals.
        5. **Practical Recommendations:** Include actionable advice such as preparation tips, alternatives for dietary restrictions or allergies, and suggestions for maintaining a balanced diet.
        6. **Important Considerations:** Address any necessary precautions (e.g., managing caloric intake, monitoring sodium levels, or allergen information) and note any assumptions made about the user's dietary requirements.

        Keep the response informative, concise, and directly relevant to both the user's query and the nutritional data provided."""

        
        # Step 4: Call Gemini Flash
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    except genai.types.generation_types.BlockedPromptException:
        return "I apologize, but I cannot provide information about that topic."
    except Exception as e:
        if "quota" in str(e).lower():
            time.sleep(2)  # Added delay for quota errors
            return "API quota exceeded. Please try again in a few minutes."
        return f"Error: {str(e)}. Please try again."


def initialize_system(api_key):
    """Initialize the system with all necessary components"""
    try:
        setup_gemini_api(api_key)
        index, text_data = load_vector_database()
        
        if index is None or text_data is None:
            print("Building new vector database...")
            text_data = load_nutrition_and_exercise_data(
                "nutrition.xlsx",
                "exercise_data_cleaned.csv",
                "physiotherapy_exercises.csv"
            )
            if not text_data:
                raise Exception("Failed to load data files")
            
            index = build_vector_database(text_data)
            if index is None:
                raise Exception("Failed to build vector database")
        
        return index, text_data
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        return None, None

def main():
    """Main function to run the fitness assistant"""
    # Initialize system
    api_key = "AIzaSyDniam4TLId5RAR8eDcglYwk4zaBYmsRo8"  # Replace with your actual API key
    index, text_data = initialize_system(api_key)
    
    if index is None or text_data is None:
        print("Failed to initialize system. Please check your data files and try again.")
        return
    
    print("Fitness Assistant initialized successfully!")
    print("Ask questions about exercise, nutrition, or physiotherapy.")
    print("Type 'exit' to quit.")
    
    query_count = 0
    last_query_time = time.time()
    
    # Interactive loop
    while True:
        try:
            current_time = time.time()
            # Add dynamic rate limiting based on query count
            if query_count > 5:
                sleep_time = max(2, min(5, current_time - last_query_time))
                time.sleep(sleep_time)
                query_count = 0
            
            user_query = input("\nEnter your question: ").strip()
            if user_query.lower() == 'exit':
                print("Goodbye!")
                break
            
            if not user_query:
                print("Please enter a valid question.")
                continue
            
            # Get and print response
            response = query_gemini_flash(user_query, text_data, index)
            print("\nResponse:", response)
            print("-" * 40)
            
            # Update query tracking
            query_count += 1
            last_query_time = current_time
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try again in a moment.")
            time.sleep(2)

if __name__ == "__main__":
    main()