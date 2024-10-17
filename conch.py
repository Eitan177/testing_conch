import streamlit as st
from PIL import Image
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

import torch
from huggingface_hub import login
login(st.secrets['hf'])
# Load the pretrained model and transforms
#model = create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
# Load the model
@st.cache_resource
def load_model():
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
    return model, preprocess

model, preprocess = load_model()

st.title("CONCH - Image Captioning and Retrieval")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and get image embeddings
    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_embs = model.encode_image(image, proj_contrast=True, normalize=True)

    st.write("Image embeddings generated successfully.")

# Text retrieval example
# Number of inputs to display (can be dynamic)
num_inputs = st.number_input("How many text inputs?", min_value=1, max_value=10, value=3)

# List to store user inputs
input_list = []

# Dynamically generate the text input fields
for i in range(num_inputs):
    user_input = st.text_input(f"Input Text {i+1}")
    input_list.append(user_input)

populated_status = ["Populated" if text.strip() else "Empty" for text in input_list]
if "Populated" in populated_status:
  # Tokenize the text
  tokenizer = get_tokenizer() # load tokenizer
  text_tokens = tokenize(texts=input_list, tokenizer=tokenizer) # tokenize the text
  text_embs = model.encode_text(text_tokens)


  #with torch.no_grad():
  #      text_embs = model.encode_text(tokens, proj_contrast=True, normalize=True)

  st.write("Text embeddings generated successfully.")

    # Perform similarity check
  similarity = torch.cosine_similarity(image_embs, text_embs)
  st.write("Similarity check completed.")
  st.write(similarity)
