import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

data=pd.read_csv("new_data.csv")
# Streamlit UI
st.title("Research Paper Abstract Matcher")

# Input abstract
user_abstract = st.text_area("Enter your abstract here:", height=200)

# Fetch unique categories from DB for dropdown
categories = data['category'].unique()
selected_category = st.selectbox("Select a research domain (category):", categories)

if st.button("Find Similar Papers") and user_abstract and selected_category:
    with st.spinner("Fetching and comparing papers..."):

        # Fetch papers from selected category
        papers = list(data[data['category']==selected_category].to_dict(orient='records'))
        if not papers:
            st.warning("No papers found for this category.")
        else:
            summaries = [paper["summary"] for paper in papers]

            # Embed user abstract and all summaries
            user_embedding = model.encode([user_abstract])
            paper_embeddings = model.encode(summaries)

            # Calculate cosine similarity
            similarities = cosine_similarity(user_embedding, paper_embeddings)[0]

            # Combine results
            similar_papers = []
            for i, score in enumerate(similarities):
                if score > 0.5:  # You can adjust threshold
                    paper = papers[i]
                    similar_papers.append({
                        "Title": paper.get("title", "N/A"),
                        "Authors": ", ".join(paper.get("authors", [])),
                        "Year": paper.get("year", "N/A"),
                        "Citations": paper.get("n_citation", 0),
                        "Similarity": round(score, 3),
                        "Summary": paper.get("summary", "")[:300] + "..."
                    })

            # Sort and display
            if similar_papers:
                df = pd.DataFrame(similar_papers).sort_values(by="Similarity", ascending=False)
                st.success(f"Found {len(df)} similar papers:")
                st.dataframe(df)
            else:
                st.info("No highly similar papers found.")

