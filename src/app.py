import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import zipfile
import os

REGISTER_URL = "http://127.0.0.1:8000/register/"
IDENTIFY_URL = "http://127.0.0.1:8000/identify/"
REGISTERED_PERSONS_URL = "http://127.0.0.1:8000/registered_persons/"
REMOVE_PERSON_URL = "http://127.0.0.1:8000/remove_person/"
REGISTER_BATCH_URL = "http://127.0.0.1:8000/register_batch/"

# Define a fixed size for all images
IMAGE_SIZE = (150, 150)

def fetch_registered_persons():
    response = requests.get(REGISTERED_PERSONS_URL)
    if response.status_code == 200:
        return response.json().get("registered_persons", [])
    return []

st.title("🔍 Face Recognition App")

st.sidebar.header("Registered Persons")
registered_persons = fetch_registered_persons()
if registered_persons:
    for person in registered_persons:
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        with col1:
            st.write(person)
        with col2:
            if st.button("❌", key=f"remove_{person}"):
                response = requests.delete(REMOVE_PERSON_URL, params={"name": person})
                if response.status_code == 200:
                    st.sidebar.success(f"{person} has been removed.")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to remove person.")
else:
    st.sidebar.write("No persons registered yet.")

tab1, tab2 = st.tabs(["Register Face", "Identify Person"])

with tab1:
    st.header("📸 Register a New Person")

    uploaded_files = st.file_uploader("Upload face images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    name = st.text_input("Enter the person's name")

    if uploaded_files:
        st.write(f"✅ {len(uploaded_files)} images uploaded.")

        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).resize(IMAGE_SIZE)
            with cols[i % 4]:
                st.image(image, caption=f"Image {i + 1}", use_container_width=True)

        if st.button("Submit Registration") and name:
            images_data = []
            for uploaded_file in uploaded_files:
                img_byte_arr = io.BytesIO()
                image = Image.open(uploaded_file).resize(IMAGE_SIZE)
                image.save(img_byte_arr, format="PNG")
                images_data.append((uploaded_file.name, img_byte_arr.getvalue()))

            files = [("files", (name, data, "image/png")) for _, data in images_data]

            with st.spinner("Registering face..."):
                response = requests.post(REGISTER_URL, files=files, data={"name": name})

            if response.status_code == 200:
                st.success("Face registered successfully!")
            else:
                st.error("Failed to register face")

    # Batch registration using folder structure (as ZIP)
    st.header("📦 Batch Register People")
    uploaded_batch = st.file_uploader("Upload batch of face images (ZIP)", type=["zip"])

    if uploaded_batch:
        st.write("✅ ZIP file uploaded.")
        if st.button("Submit Batch Registration"):
            with st.spinner("Registering faces in batch..."):
                files = {"files": uploaded_batch}
                response = requests.post(REGISTER_BATCH_URL, files=files)

            if response.status_code == 200:
                st.success("Batch registration completed successfully!")
            else:
                st.error("Failed to register faces")

with tab2:
    st.header("🔍 Identify a Person")

    option = st.radio("Choose an option:", ("Upload Image", "Capture from Webcam"))
    image_data = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image to identify", type=["jpg", "png", "jpeg"], key="identify_uploader")
        if uploaded_file:
            image = Image.open(uploaded_file).resize(IMAGE_SIZE)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            image_data = img_byte_arr.getvalue()

    elif option == "Capture from Webcam":
        cam_image = st.camera_input("Take a picture")
        if cam_image:
            image = Image.open(cam_image)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            image_data = img_byte_arr.getvalue()

    if image_data and st.button("Identify Person"):
        with st.spinner("Identifying face..."):
            response = requests.post(IDENTIFY_URL, files={"file": image_data})

        if response.status_code == 200:
            result = response.json()
            st.success(f"Identified as: {result['name']} (Similarity: {result['similarity']:.2f})")
        else:
            st.error("Failed to identify face")
