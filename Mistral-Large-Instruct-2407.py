from huggingface_hub import InferenceClient
from huggingface_hub import snapshot_download
from pathlib import Path
import streamlit as st
import time

token = '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'

mistral_models_path = Path.home().joinpath('mistral_models', 'Large')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-Large-Instruct-2407", 
                  allow_patterns=["params.json", "consolidated-*.safetensors", "tokenizer.model.v3"], 
                  local_dir=mistral_models_path, 
                  token=token)


# Initialize the HuggingFace client with your token
client = InferenceClient(
    "mistralai/Mistral-Nemo-Instruct-2407",
    token=token,  # Replace with your actual Hugging Face token
)

# Streamlit app setup
st.title("ChatGPT Replica")

# Input area for the user
prompt = st.text_input("Enter your question:", "")

if prompt:
    st.write("Processing...")
    response = ""

    try:
        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=True,
        ):
            content = message.choices[0].delta.content
            response += content
            # Display the response as it is generated
            st.write(response)
            # Add a slight delay to simulate streaming
            time.sleep(0.1)
    except Exception as e:
        st.write("An error occurred:", str(e))

# Display the final response
if response:
    st.write("Response:", response)
