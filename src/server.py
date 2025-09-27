import os
import torch
import uvicorn
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from torchvision import transforms
from PIL import Image
import io
from pathlib import Path
from SiameseNetwork import SiameseNetwork
import zipfile

app = FastAPI()

# Directories for storing model, embeddings, and uploads
EMBEDDINGS_FILE = "../data/face_embeddings.pkl"
MODEL_PATH = "../model/best.pth"
uploads_dir = Path("./uploads")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        face_embeddings = pickle.load(f)
else:
    face_embeddings = {}

@app.post("/register/")
async def register_face(name: str = Form(...), files: List[UploadFile] = File(...)):
    global face_embeddings

    embeddings_list = []
    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(image_tensor).cpu().numpy().flatten()
        embeddings_list.append(embedding)

    if not embeddings_list:
        return {"error": "No valid images uploaded."}

    mean_embedding = np.mean(embeddings_list, axis=0)
    face_embeddings[name] = mean_embedding

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(face_embeddings, f)

    return {"message": f"Face registered for {name}", "embeddings_saved": 1}

@app.get("/registered_persons/")
def get_registered_persons():
    """Returns a list of all registered persons."""
    return {"registered_persons": list(face_embeddings.keys())}

@app.delete("/remove_person/")
def remove_person(name: str):
    """Removes a registered person from the system."""
    global face_embeddings
    if name in face_embeddings:
        del face_embeddings[name]
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(face_embeddings, f)
        return {"message": f"{name} has been removed."}
    else:
        raise HTTPException(status_code=404, detail="Person not found.")

@app.post("/register_batch/")
async def register_batch(files: List[UploadFile] = File(...)):
    global face_embeddings

    # Create the uploads directory if it doesn't exist
    if not uploads_dir.exists():
        uploads_dir.mkdir(parents=True)

    for file in files:
        # Ensure that the file path is correctly formed
        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Handle zip files
        if file_path.suffix == '.zip':
            try:
                # Extract the contents of the zip file into a folder named after the zip file
                extracted_folder = uploads_dir / file.filename.split('.')[0]
                if not extracted_folder.exists():
                    extracted_folder.mkdir(parents=True)

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder)

                # Process each folder inside the extracted ZIP
                for person_folder in extracted_folder.iterdir():
                    if person_folder.is_dir():
                        embeddings_list = []
                        for image_file in person_folder.glob("*.jpg"):
                            image = Image.open(image_file).convert("RGB")
                            image_tensor = transform(image).unsqueeze(0).to(device)

                            with torch.no_grad():
                                embedding = model(image_tensor).cpu().numpy().flatten()
                            embeddings_list.append(embedding)

                        if embeddings_list:
                            mean_embedding = np.mean(embeddings_list, axis=0)
                            face_embeddings[person_folder.name] = mean_embedding
                        else:
                            raise HTTPException(status_code=400,
                                                detail=f"No valid images found in folder {person_folder.name}.")
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP file.")

        else:
            raise HTTPException(status_code=400, detail="Uploaded file is not a ZIP file.")

    # Save the updated face embeddings to the file
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(face_embeddings, f)

    return {"message": "Batch registration completed.", "embeddings_saved": len(face_embeddings)}

@app.post("/identify/")
async def identify_face(file: UploadFile = File(...)):
    if not face_embeddings:
        return {"error": "No registered faces. Please register some faces first."}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(image_tensor).cpu().numpy().flatten()

    similarities = {}
    for name, embedding in face_embeddings.items():
        similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities[name] = similarity

        # Print each similarity to the console
        print(f"Similarity with {name}: {similarity:.2f}")

    best_match = max(similarities, key=similarities.get)
    best_similarity = similarities[best_match]

    THRESHOLD = 0.8
    recognized_name = best_match if best_similarity >= THRESHOLD else "Unknown"

    return {"name": recognized_name, "similarity": float(best_similarity)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
