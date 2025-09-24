# Resume Screening with AI (TF-IDF + Cosine Similarity + Streamlit UI)

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx2txt

# --- Helper Functions --- #
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        text = docx2txt.process(docx_file)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def clean_text(text):
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

# --- Main Application --- #
def main():
    st.title("AI Resume Screening System")
    st.write("Upload job description and resumes to find the best matches!")
    
    # Job Description Input
    st.subheader("Job Description")
    job_desc = st.text_area("Paste the job description here:", height=200)
    
    # Resume Upload
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF or DOCX)", 
        type=['pdf', 'docx'], 
        accept_multiple_files=True
    )
    
    if st.button("Screen Resumes") and job_desc and uploaded_files:
        # Process job description
        cleaned_job_desc = clean_text(job_desc)
        
        # Process resumes
        resumes_data = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
            
            if text:
                cleaned_text = clean_text(text)
                resumes_data.append({
                    'filename': file.name,
                    'text': cleaned_text
                })
        
        if not resumes_data:
            st.error("No valid resumes found!")
            return
        
        # Create TF-IDF Matrix
        documents = [cleaned_job_desc] + [resume['text'] for resume in resumes_data]
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Create results dataframe
            results = []
            for i, resume in enumerate(resumes_data):
                results.append({
                    'Filename': resume['filename'],
                    'Similarity Score': round(cosine_similarities[i] * 100, 2)
                })
            
            # Sort by similarity score
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Similarity Score', ascending=False)
            
            # Display results
            st.subheader("Screening Results")
            st.dataframe(results_df)
            
            # Show top candidate
            if not results_df.empty:
                top_candidate = results_df.iloc[0]
                st.success(f"Top Match: {top_candidate['Filename']} "
                          f"({top_candidate['Similarity Score']}% match)")
                
        except Exception as e:
            st.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()