# 📓Raster Image → HTML (OCR) Converter

Turns page photos or PDF images into interactive, searchable HTML using OCR (client–server application).

## Features

- PDF to HTML conversion with OCR text extraction
- Interactive HTML viewer with theme selection
- Font scaling and page navigation
- Image popup functionality
- Single PDF processing with background tasks
- Auto-cleanup of temporary files

## Project Structure

```
pdf2html-ocr/
├── app/
│   └── server.py          # FastAPI server for PDF processing
├── src/
│   └── client.py          # Client for uploading PDFs and downloading results
└── README.md
```

## Installation

1. Clone the DotsOCR repository:
```bash
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
```

2. Install dependencies:
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install modelscope
```

3. Download the OCR model:
```bash
python3 tools/download_model.py --type modelscope
```

4. Rename the repository folder from `dots.ocr` to `dots_ocr`

5. Place the server script in the `dots_ocr` folder

## Usage

### Start the Server
```bash
python app/server.py
```
The server will run on `http://localhost:8000`

### Use the Client
```bash
python src/client.py path/to/your/document.pdf
```

### Interactive Mode
```bash
python src/client.py
```

## API Endpoints

- `GET /` - Health check and API information
- `POST /upload` - Upload PDF file for processing
- `GET /status/{job_id}` - Check processing status
- `GET /download/{job_id}` - Download HTML results
- `GET /jobs` - List all jobs
- `DELETE /cleanup` - Manual cleanup
- `DELETE /jobs/{job_id}` - Delete specific job

## Features

- **Theme Selection**: Choose from White, Gray, or Dark Green themes
- **Font Scaling**: Adjust text size with A+/A- buttons
- **Page Navigation**: Navigate through pages with arrow keys or buttons
- **Image Popups**: Click on images for enlarged view
- **Smart Layout**: Automatic text positioning to avoid overlapping with figures
- **MathJax Support**: Proper rendering of mathematical formulas
- **Auto Cleanup**: Temporary files are cleaned up after 2 minutes when idle

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support
- DotsOCR model weights

## Output

The converted HTML file includes:
- Interactive page viewer
- Positioned text elements matching the original PDF layout
- Clickable images and formulas
- Theme and font customization options
- Keyboard navigation support
