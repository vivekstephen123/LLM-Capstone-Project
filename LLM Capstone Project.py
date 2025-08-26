import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import faiss
import torch

# Set page config
st.set_page_config(page_title="Smarter Car Recommender", page_icon="🚗", layout="wide")

# Title and description
st.title("Smarter LLM-Based Car Recommendation System")
st.markdown("""
This app recommends cars based on your preferences. **Now updated with a 'Vehicle Type' filter to ensure category accuracy (e.g., SUV vs. Sedan).**
- **Hard Filters**: Price, Year, Fuel, Transmission, and **Vehicle Type**.
- **Smart Search**: Semantic search finds the best matches *within* your filtered results.
- **AI Explanations**: The LLM explains why each car is a good fit.
""")

# --- 1. FEATURE ENGINEERING: Add a function to classify vehicle types ---
def get_vehicle_type(name):
    """Classifies vehicle type based on model name."""
    name = name.lower()
    # High-confidence SUV names
    if any(suv in name for suv in ['fortuner', 'land cruiser', 'vitara brezza', 's cross', 'etios cross']):
        return 'SUV'
    # High-confidence MPV names
    if any(mpv in name for mpv in ['innova', 'ertiga']):
        return 'MPV'
    # High-confidence Sedan names
    if any(sedan in name for sedan in ['corolla', 'ciaz', 'sx4', 'dzire', 'etios g', 'etios gd', 'camry']):
        return 'Sedan'
    # Filter out motorcycles which are in the dataset
    if any(bike in name for bike in ['royal enfield', 'bajaj', 'tvs', 'honda', 'hero', 'yamaha', 'ktm']):
        return 'Motorcycle'
    # Default to Hatchback for most other small cars
    if any(hatch in name for hatch in ['ritz', 'swift', 'wagon r', 'alto', 'baleno', 'ignis', 'omni', 'liva', '800']):
        return 'Hatchback'
    return 'Other' # Fallback for anything not caught

# Load car dataset from online CSV
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sumit0072/Car-Price-Prediction-Project/main/car%20data.csv"
    df = pd.read_csv(url)
    
    # --- Apply the vehicle type classification ---
    df['Vehicle_Type'] = df['Car_Name'].apply(get_vehicle_type)
    
    # --- Filter out Motorcycles and 'Other' categories for cleaner recommendations ---
    df = df[~df['Vehicle_Type'].isin(['Motorcycle', 'Other'])]
    
    df['description'] = df.apply(lambda row: f"{row['Year']} {row['Car_Name']} ({row['Vehicle_Type']}) with {row['Fuel_Type']} fuel, {row['Transmission']} transmission, {row['Kms_Driven']} kms driven.", axis=1)
    df = df.dropna()
    return df

cars = load_data()

# Load embedding model from Hugging Face
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# Precompute car embeddings and build FAISS index
@st.cache_data
def get_car_index(_cars_df):
    embeddings = embedder.encode(_cars_df['description'].tolist(), convert_to_tensor=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.cpu().numpy().astype(np.float32))
    return index, embeddings.cpu().numpy()

index, car_embeddings = get_car_index(cars)

# LLM setup with an instruction-tuned model
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=250,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# Prompt Template (no changes needed here)
prompt_template = """
You are a helpful car recommendation assistant. Based on the user's preferences and the details of the car provided, generate a concise, engaging explanation. Structure your response exactly like this:

*Why This Car?*
[Short intro sentence matching the car to the user's preferences.]
- *Pros*:
  - [Pro 1]
  - [Pro 2]
- *Cons*:
  - [Con 1]
- *Summary*: [Engaging one-sentence recommendation.]

---
User preferences: {query}
Car details:
{description}
---

Output only the structured explanation.
"""
prompt = PromptTemplate(input_variables=["query", "description"], template=prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)

def clean_explanation(raw_output):
    start_marker = "*Why This Car?*"
    start_index = raw_output.find(start_marker)
    return raw_output[start_index:].strip() if start_index != -1 else raw_output.strip()

# --- 2. UI AND FILTERING LOGIC: Add the new filter ---
st.sidebar.header("Filter Options")
vehicle_types = st.sidebar.multiselect("Vehicle Type", options=cars['Vehicle_Type'].unique(), default=cars['Vehicle_Type'].unique())
min_price, max_price = st.sidebar.slider("Price Range (₹ Lakhs)", float(cars['Selling_Price'].min()), 35.0, (0.0, 35.0))
min_year, max_year = st.sidebar.slider("Year Range", int(cars['Year'].min()), int(cars['Year'].max()), (int(cars['Year'].min()), int(cars['Year'].max())))
fuel_types = st.sidebar.multiselect("Fuel Type", options=cars['Fuel_Type'].unique(), default=cars['Fuel_Type'].unique())
transmissions = st.sidebar.multiselect("Transmission", options=cars['Transmission'].unique(), default=cars['Transmission'].unique())

# Apply all filters, including the new Vehicle_Type
filtered_cars = cars[
    (cars['Vehicle_Type'].isin(vehicle_types)) &
    (cars['Selling_Price'] >= min_price) & (cars['Selling_Price'] <= max_price) &
    (cars['Year'] >= min_year) & (cars['Year'] <= max_year) &
    (cars['Fuel_Type'].isin(fuel_types)) &
    (cars['Transmission'].isin(transmissions))
].copy()

if filtered_cars.empty:
    st.warning("No cars match your filters. Please adjust them to see recommendations.")
    st.stop()

# Rebuild FAISS index for the filtered cars
filtered_indices_original = filtered_cars.index.tolist()
filtered_embeddings = car_embeddings[filtered_indices_original]
if filtered_embeddings.shape[0] == 0:
    st.warning("No cars found for the selected filter combination.")
    st.stop()
    
filtered_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
filtered_index.add(filtered_embeddings.astype(np.float32))

# Chat interface
st.subheader("Chat with the Recommender")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("e.g., 'A reliable family SUV for under ₹10 Lakhs'")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Finding the best matches..."):
            try:
                query_embedding = embedder.encode(query).astype(np.float32).reshape(1, -1)
                
                # Search FAISS index for top 3 matches WITHIN the filtered group
                k = min(3, len(filtered_cars))
                distances, indices_in_filtered = filtered_index.search(query_embedding, k=k)
                
                response = ""
                scores, recommended_cars = [], []
                
                for i, dist in zip(indices_in_filtered[0], distances[0]):
                    car = filtered_cars.iloc[i]
                    recommended_cars.append(car['Car_Name'])
                    similarity_score = 1 / (1 + dist)
                    scores.append(similarity_score)
                    
                    raw_explanation = chain.run({
                        "query": query,
                        "description": car['description']
                    })
                    
                    explanation = clean_explanation(raw_explanation)
                    
                    response += f"### {car['Car_Name']} ({car['Year']}) - ₹{car['Selling_Price']} Lakhs (Similarity: {similarity_score:.2f})\n"
                    response += f"**Type**: {car['Vehicle_Type']} | **Fuel**: {car['Fuel_Type']} | **Transmission**: {car['Transmission']} | **Kms**: {car['Kms_Driven']}\n\n"
                    response += explanation + "\n\n---\n"
                
                if scores:
                    score_df = pd.DataFrame({'Car': recommended_cars, 'Similarity Score': scores})
                    st.bar_chart(score_df.set_index('Car'))
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}. Please try again.")