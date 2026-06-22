# Omnichannel Visual Search Engine: Image Vectorization Microservice

A high-performance, container-ready Python microservice built using FastAPI and the `sentence-transformers` execution framework. This service abstracts the standard OpenAI CLIP (`clip-ViT-B-32`) vision transformer model to generate mathematically precise, 512-dimensional float vector coordinate arrays from both direct binary multi-part uploads and remote cloud asset URLs.

This service acts as the core artificial intelligence spoke for the `Omnichannel-HQ` ecosystem, driving automated computer vision, semantic discovery matching, and sub-millisecond reverse image lookups inside a PostgreSQL database cluster.

---

## 🚀 Key Architectural Features

*   **Multi-Modal Ingestion Engine**: Dynamically accepts either an uploaded binary image stream (`multipart/form-data`) or a remote, secure Cloudinary URL string (`GET query parameter`) within a unified endpoint block.
*   **Volatile Memory URL Streaming**: Fetches remote cloud image data direct to an in-memory buffer channel (`io.BytesIO`) via fast HTTP request timeouts. This bypasses disk space writes, keeping performance blazing fast and RAM footprint lightweight.
*   **CLIP Feature Extraction Tracking**: Leverages `SentenceTransformer` vector encodings (`.encode()`) to produce standard float coordinate arrays mapping complex visual traits, spatial color maps, and shapes seamlessly.

---

## 🛠️ System Requirements & Dependencies

Ensure your execution layer handles **Python 3.10+** before initializing dependency allocations.

Core requirements profile:
*   `fastapi` & `uvicorn` (ASGI server utility layer)
*   `sentence-transformers` (CLIP vector abstraction layer)
*   `pillow` (PIL visual byte data manipulation engine)
*   `requests` (Remote cloud media content streamer)

---

## 📦 Getting Started & Environment Setup

### 1. Initialize Virtual Workspace
Clone this service profile repository onto your machine node and run an isolated virtual environment shell track:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Dependency Allocations
Install the image processing and networking dependencies using pip:
```bash
pip install fastapi uvicorn sentence-transformers pillow requests
```
> **Note on First Run Execution**: The very first time a vectorized task is processed, the `sentence-transformers` model engine will automatically download the pre-trained CLIP binary weights (`clip-ViT-B-32`) from the remote repository cache. This may take a moment depending on network bandwidth speeds.

### 3. Boot Up the ASGI Microservice Daemon
Launch the network socket listener on your designated environmental port mapping (configured to match your Laravel core `.env` service communication configs):
```bash
uvicorn main:app --host 127.0.0.1 --port 5000 --reload
```

---

## 📡 API Specification Snapshot

### Core Status Verification
*   **Route**: `GET /`
*   **Payload Response**: `{"status": "online", "model": "clip-ViT-B-32", "dimensions": 512}`

### Image Vectorization Mapping
*   **Endpoint Route**: `POST /vectorize`
*   **Accepted Content-Types**: `application/json` (when passing URL strings) OR `multipart/form-data` (when uploading file binary blocks)

#### Scenario A: Cloudinary URL Query Pass-Through (Recommended for Production)
*   **Endpoint URL Layout**: `POST /vectorize?url=https://cloudinary.com`

#### Scenario B: Direct Binary Upload Fallback
*   **Request Payload**: An image file binary block mapped under the parameter form-data key string field `image`.

#### Successful Response Layout Matrix (`200 OK`)
```json
{
  "success": true,
  "filename": "sample.jpg",
  "dimensions": 512,
  "vector": [
    0.01234567,
    -0.04567891,
    0.00891234,
    "...Truncated remaining 509 floating-point coordinates..."
  ]
}
```