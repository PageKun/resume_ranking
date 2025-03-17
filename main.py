import streamlit as st
import pdfplumber  # Changed from PyPDF2
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load English language model for spaCy
nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(file):
    """Extract text from PDF file using pdfplumber"""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

def preprocess_text(text):
    """Clean and preprocess text using spaCy"""
    if not text:
        return ""
    
    # Basic cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Advanced NLP processing
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def rank_resumes(job_description, resumes):
    """Calculate cosine similarity scores between job description and resumes"""
    try:
        documents = [job_description] + resumes
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return similarity_scores
    except Exception as e:
        st.error(f"Error in ranking process: {str(e)}")
        return None

# Streamlit UI
st.title("AI Resume Ranking System")
st.markdown("Upload job description and resumes to get ranking")

# Job Description Input
jd = st.text_area("Paste Job Description:", height=200)

# Resume Upload
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF only)", 
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process Resumes") and jd and uploaded_files:
    with st.spinner("Processing resumes..."):
        # Process job description
        processed_jd = preprocess_text(jd)
        
        # Process resumes
        resumes = []
        resume_data = []
        
        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            if raw_text:
                cleaned_text = preprocess_text(raw_text)
                resumes.append(cleaned_text)
                resume_data.append({
                    'Filename': file.name,
                    'Raw Text': raw_text,
                    'Processed Text': cleaned_text
                })
        
        if len(resumes) > 0:
            # Calculate scores
            scores = rank_resumes(processed_jd, resumes)
            
            # Create results dataframe
            results_df = pd.DataFrame(resume_data)
            results_df['Similarity Score'] = scores
            results_df['Rank'] = results_df['Similarity Score'].rank(ascending=False)
            results_df = results_df.sort_values('Rank')
            
            # Display results
            st.subheader("Ranking Results")
            st.dataframe(
                results_df[['Filename', 'Similarity Score', 'Rank']]
                .sort_values('Rank')
                .style.format({'Similarity Score': '{:.2%}'})
                .bar(subset=['Similarity Score'], color='#5fba7d'),
                height=400
            )
        else:
            st.warning("No valid resumes found in uploaded files")