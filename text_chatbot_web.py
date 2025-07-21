import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Page Layout ---
st.set_page_config(page_title="AI Chatbot - Text Q&A", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot - Chat with Your Text")

# --- Text Area to Paste Content ---
text_data = st.text_area("📋 Paste the knowledge text below (rules, syllabus, policy, etc.):", height=300)

# Process the text if entered
if text_data:
    # Split into chunks
    chunks = [text_data[i:i+500] for i in range(0, len(text_data), 500)]

    # Generate vector embeddings
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    st.success("✅ Text processed! Ask your questions below:")

    # Ask question
    user_input = st.text_input("❓ Ask a question about the text:")
    
    if user_input:
        # Search similar chunk
        q_vec = model.encode([user_input])
        D, I = index.search(np.array(q_vec), k=1)
        answer = chunks[I[0][0]]

        # Display answer
        st.markdown("### 💡 Answer:")
        st.info(answer)
