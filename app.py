import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit setup
st.set_page_config(page_title="🤖 AI Product Ideator", page_icon="🚀")
st.title("🚀 AI Product Ideator")

# Initialize model
llm = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

# Prompt for generating product ideas
idea_prompt = PromptTemplate(
    input_variables=["interest"],
    template="""
You are a product strategist.

Given this area of interest or problem: "{interest}",

Suggest 3 creative and feasible AI product ideas. Be specific and concise.

Format:
1. ...
2. ...
3. ...
"""
)
idea_chain = LLMChain(llm=llm, prompt=idea_prompt)

# Prompt for generating tech stack
detail_prompt = PromptTemplate(
    input_variables=["idea_description"],
    template="""
You are a senior AI engineer.

Given this AI product idea: "{idea_description}",

Suggest:
- The ideal tech stack (frameworks, libraries, models, tools)
- Key implementation details or tips for building it
"""
)
detail_chain = LLMChain(llm=llm, prompt=detail_prompt)

# Input field for interest
interest = st.text_input("💡 Enter a domain or problem area you're passionate about:")

# State variable for storing ideas
if "idea_result" not in st.session_state:
    st.session_state.idea_result = ""

# Step 1: Generate ideas
if interest and st.button("Generate AI Ideas"):
    with st.spinner("Thinking of product ideas..."):
        st.session_state.idea_result = idea_chain.run(interest)

# Step 2: Show ideas if available
if st.session_state.idea_result:
    st.subheader("✨ AI Product Ideas")
    st.text_area("Here are 3 AI product ideas:", st.session_state.idea_result, height=200)

    idea_number = st.selectbox("Select an idea number to explore:", ["", "1", "2", "3"], index=0)

    if idea_number:
        # Parse idea from result
        idea_lines = st.session_state.idea_result.strip().split("\n")
        selected_line = next((line for line in idea_lines if line.strip().startswith(f"{idea_number}.")), None)

        if selected_line:
            with st.spinner("Generating technical insights..."):
                detail_result = detail_chain.run(selected_line)
            st.subheader("🛠️ Tech Stack & Build Guide")
            st.text_area(f"Details for Idea #{idea_number}", detail_result, height=300)
        else:
            st.warning("Couldn't extract the idea properly.")
