import io
import requests  # Make sure to run: pip install requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image
from sentence_transformers import SentenceTransformer

# 1. Initialize FastAPI and pre-load the multi-modal CLIP model
app = FastAPI(title="Product Image Vectorizer Service")

# This downloads the standard OpenAI CLIP model (outputs a 512-dimensional vector)
model = SentenceTransformer('clip-ViT-B-32')

@app.get("/")
def read_root():
    return {"status": "online", "model": "clip-ViT-B-32", "dimensions": 512}

@app.post("/vectorize")
async def vectorize_image(
    image: UploadFile = File(None),  # Changed from File(...) to File(None) to make it optional
    url: str = Query(None)           # Added to support Cloudinary URLs
):
    try:
        # Scenario A: Download image from Cloudinary URL if provided
        if url:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image from Cloudinary URL.")
            pil_image = Image.open(io.BytesIO(response.content))
            filename = url.split('/')[-1]
            
        # Scenario B: Fallback to direct file upload if no URL is provided
        elif image:
            # Validate that the uploaded file is actually an image
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
            
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            filename = image.filename
            
        else:
            raise HTTPException(status_code=400, detail="You must provide either an image file or a url parameter.")
        
        # 4. Generate the embedding vector array
        vector = model.encode(pil_image)
        vector_list = vector.tolist()
        
        # 5. Return clean JSON response back to Laravel
        return {
            "success": True,
            "filename": filename,
            "dimensions": len(vector_list),
            "vector": vector_list
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector generation failed: {str(e)}")
