from typing import List, Optional, Dict, Any
from pathlib import Path
from PIL import Image

from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    OcrAutoOptions,
)
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import tiktoken

from .models import DocumentChunk

from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    OcrAutoOptions,
)


def build_pipeline_options() -> ThreadedPdfPipelineOptions:
    return ThreadedPdfPipelineOptions(
        do_ocr=True,          # OCR when text is missing
        ocr_options=OcrAutoOptions(lang=["eng"]),
        generate_page_images=True,
        generate_picture_images=True,
    )
 
def build_converter() -> DocumentConverter:
    return DocumentConverter()  

def build_hybrid_chunker(
    max_tokens: int = 768,
) -> HybridChunker:
    # Use a tokenizer encoding compatible with ollama's nomic-embed-text
    try:
        # Prefer nomic-embed-text encoding when available
        encoding = tiktoken.encoding_for_model("nomic-embed-text")
    except Exception:
        # Fallback to a generic encoding compatible with many embedding models
        encoding = tiktoken.get_encoding("cl100k_base")

    tokenizer = OpenAITokenizer(tokenizer=encoding, max_tokens=max_tokens)

    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,   # strongly recommended for RAG
    )
    
def save_image(
    image: Image.Image,
    source_file: str,
    page_number: int,
    image_index: int,
    output_dir: str = "images",
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_path = (
        Path(output_dir)
        / f"{source_file}_p{page_number}_img{image_index}.png"
    )

    image.save(image_path)
    return str(image_path)

def process_document(
    file_path: str,
    max_tokens: int = 768,
) -> List[DocumentChunk]:
    source_file = Path(file_path).name

    # ---- Build components ----
    converter = build_converter()
    pipeline_options = build_pipeline_options()
    # Attach pipeline options to PDF format so DocumentConverter uses them
    converter.format_to_options[InputFormat.PDF].pipeline_options = pipeline_options

    chunker = build_hybrid_chunker(max_tokens=max_tokens)

    # ---- Convert document (OCR + parsing) ----
    conversion_result = converter.convert(
        file_path,
    )

    document = conversion_result.document

    # ---- Hybrid chunking ----
    docling_chunks = list(chunker.chunk(dl_doc=document))

    output_chunks: List[DocumentChunk] = []
    image_counter = 0

    for chunk in docling_chunks:
        page_numbers, bbox, has_ocr, language = _extract_meta_fields(chunk)

        # ---- TEXT / TABLE / CAPTION ----
        if getattr(chunk, "text", None) and chunk.text.strip():
            output_chunks.append(
                DocumentChunk.create(
                    text=chunk.text,
                    metadata={
                        "source_file": source_file,
                        "page_numbers": page_numbers,
                        "chunk_type": None,
                        "bbox": bbox,
                        "has_ocr": has_ocr,
                        "language": language,
                        "image_path": None,
                    },
                )
            )

        # ---- IMAGE ----
        img_obj = getattr(chunk, "image", None)
        # If underlying doc_item contains an image ref, it may be available via meta export as a dict
        if img_obj is None:
            meta_dict = chunk.meta.export_json_dict() if hasattr(chunk.meta, "export_json_dict") else {}
            doc_items = meta_dict.get("doc_items", []) or []
            if doc_items and isinstance(doc_items[0], dict):
                img_ref = doc_items[0].get("image_ref")
                if img_ref:
                    # image_ref.uri might be a file path or data url; skip complex handling for now
                    img_obj = None

        if img_obj is not None:
            page_number = page_numbers[0] if page_numbers else -1

            image_path = save_image(
                image=img_obj,
                source_file=source_file,
                page_number=page_number,
                image_index=image_counter,
            )

            image_counter += 1

            output_chunks.append(
                DocumentChunk.create(
                    text="",
                    metadata={
                        "source_file": source_file,
                        "page_numbers": page_numbers,
                        "chunk_type": "image",
                        "bbox": bbox,
                        "has_ocr": False,
                        "language": None,
                        "image_path": image_path,
                    },
                )
            )

    return output_chunks
    