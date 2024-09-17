import base64
import io
import os
import re
import fitz  # PyMuPDF
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Set API key for Google Generative AI
API_KEY = "addyourapikey"
genai.configure(api_key=API_KEY)

def get_gemini_response(input_text, pdf_content, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
        response = model.generate_content([input_text, pdf_content, prompt])
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def input_pdf_setup(uploaded_file):
    try:
        if uploaded_file is not None:
            pdf_text = ""
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                pdf_text += page.get_text()
            doc.close()

            if pdf_text:
                return pdf_text
            else:
                raise ValueError("No text found in the PDF.")
        else:
            raise FileNotFoundError("No file uploaded")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def calculate_ats_score(job_description, resume_text):
    # Create TF-IDF vectors for job description and resume text
    vectorizer = TfidfVectorizer().fit([job_description, resume_text])
    vectors = vectorizer.transform([job_description, resume_text])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    score = cosine_sim * 100  # Convert similarity to percentage
    
    # Get missing keywords
    job_keywords = set(re.findall(r'\w+', job_description.lower()))
    resume_keywords = set(re.findall(r'\w+', resume_text.lower()))
    missing_keywords = list(job_keywords - resume_keywords)
    
    return score, missing_keywords

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert", page_icon=":bar_chart:")
st.title("ATS Tracking System")

st.markdown("""
Welcome to the ATS Tracking System! Upload your resume and provide the job description to get insights on how well your resume matches the job requirements. 
You can choose to receive a professional evaluation of your resume or get a percentage match along with missing keywords.

### How to Use
1. **Job Description**: Paste the job description into the text area.
2. **Upload Resume**: Upload your resume in PDF format.
3. **Submit**: Click on the buttons to get the desired results.
""")

input_text = st.text_area("**Job Description**:", key="input", help="Paste the job description here.")
uploaded_file = st.file_uploader("**Upload Your Resume (PDF)**:", type=["pdf"], help="Upload your resume in PDF format.")

col1, col2 = st.columns([1, 2])

with col1:
    submit1 = st.button("Evaluate Resume")

with col2:
    submit3 = st.button("Calculate Percentage Match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First, the output should come as percentage, then keywords missing, and finally, your thoughts.
"""

if submit1:
    with st.spinner("Processing your request..."):
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            if pdf_content is not None:
                response = get_gemini_response(input_text, pdf_content, input_prompt1)
                st.subheader("Professional Evaluation:")
                st.write(response)
        else:
            st.error("Please upload the resume before submitting.")

elif submit3:
    with st.spinner("Calculating ATS score..."):
        if uploaded_file is not None:
            resume_text = input_pdf_setup(uploaded_file)
            if resume_text is not None:
                # Calculate ATS score
                score, missing_keywords = calculate_ats_score(input_text, resume_text)

                st.subheader("ATS Score")
                st.write(f"**ATS Score: {score:.2f}%**")

                if missing_keywords:
                    st.write("### Missing Keywords:")
                    st.write(", ".join(missing_keywords))
                else:
                    st.write("### No Missing Keywords! The resume covers all necessary terms.")

                thoughts_msg = (
                    f"The resume matches the job description with an ATS score of {score:.2f}%. "
                    f"{'However, there are some keywords missing that are important for the role.' if missing_keywords else 'Great job! The candidate has tailored their resume well.'}"
                )
                st.write("### Thoughts:")
                st.write(thoughts_msg)
        else:
            st.error("Please upload the resume before submitting.")
