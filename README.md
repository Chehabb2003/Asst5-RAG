Video QA Retrieval-Augmented Generation (RAG) App

  This application utilizes a Retrieval-Augmented Generation (RAG) system to enable users to query specific segments from a video. It integrates both textual (transcripts) and visual (image frames) modalities for enhanced query understanding.

Files and their Usage

  app.py: Main Streamlit application file for user interaction, retrieval logic, and UI.

  frame_to_chunk.csv: Maps video frame IDs to their respective video segments.

  image_embeddings.npy: Precomputed visual embeddings for frames using CLIP.

  text_embeddings.npy: Precomputed textual embeddings for video transcript segments.

  transcript_chunks.csv: Contains transcript segments with their start and end timestamps.

  transcript_with_index.csv: Transcript segments linked to indices used in embeddings.

  requirements.txt: Python dependencies needed to run the application.

How to Run

  Link GitHub Repository:

  Connect your GitHub account to Streamlit Cloud.

  Select this repository to deploy the Streamlit application.

Environment Setup:

  Ensure requirements.txt is correctly recognized and installed automatically by Streamlit Cloud.

Running the Application:

  Once deployed, open the provided Streamlit app URL.

  Enter natural language queries related to the video content.

  The app will respond by either playing the relevant video segment or indicating if the answer isn't present.

