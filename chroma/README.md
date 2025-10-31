# ChromaDB Database

This directory contains the pre-built ChromaDB vector database with Korean press ethics similar cases.

## Database File

The `chroma.sqlite3` file (167MB) is too large for standard Git storage and requires Git LFS.

### Option 1: Download from Hugging Face (Recommended)

The complete database is available in the Hugging Face Space:
https://huggingface.co/spaces/jonghhhh/press_ethics

Clone the space to get the database:
```bash
git clone https://huggingface.co/spaces/jonghhhh/press_ethics
cp press_ethics/chroma/chroma.sqlite3 ./chroma/
```

### Option 2: Rebuild from Source

If you have the original ethics case data, you can rebuild the database:
```python
# TODO: Add rebuild script
```

## Directory Structure

```
chroma/
├── chroma.sqlite3          (167MB - main database, requires Git LFS)
├── 4f0d0b85-04bb-4afc-9e18-51530c937a3b/  (index files)
└── README.md               (this file)
```

## Note

For deployment to platforms like Render.com or Streamlit Cloud, you'll need to either:
1. Use the Hugging Face Space version which includes the full database
2. Download the database file separately during build/deployment
3. Set up Git LFS if your platform supports it
