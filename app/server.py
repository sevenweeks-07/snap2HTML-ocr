#!/usr/bin/env python3
"""
FastAPI PDF to HTML Converter Server
Based on DotsOCR Enhanced Positioned HTML Converter
- Upload PDF files
- Get HTML pages as ZIP download
- Single PDF processing only
- Auto-cleanup temp files 2 minutes after completion when server is idle
- No index.html generation
"""

import os
import json
import shutil
import asyncio
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import tempfile
import uuid
import re
from PIL import Image
import re

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# DotsOCR imports
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.image_utils import PILimage_to_base64
from dots_ocr.utils.doc_utils import load_images_from_pdf

_pdf_image_cache = {}

def load_images_cached(pdf_path, dpi):
    key = (pdf_path, dpi)
    imgs = _pdf_image_cache.get(key)
    if imgs is None:
        imgs = load_images_from_pdf(pdf_path, dpi=dpi)
        _pdf_image_cache[key] = imgs
    return imgs

def union_bboxes(bbox1, bbox2):
    """Calculate union of two bounding boxes"""
    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    return [x1, y1, x2, y2]


def bboxes_overlap(bbox1, bbox2, threshold=0.1):
    """Check if two bounding boxes overlap significantly"""
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x1 >= x2 or y1 >= y2:
        return False
    
    # Calculate areas
    intersection_area = (x2 - x1) * (y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Check if intersection is significant
    min_area = min(bbox1_area, bbox2_area)
    overlap_ratio = intersection_area / min_area if min_area > 0 else 0
    
    return overlap_ratio > threshold


def is_mergeable_element(category):
    """Check if element type should be considered for merging/union operations"""
    mergeable_types = [
        'Picture', 'Figure', 
        'Section-header', 'Page-header', 'Page-footer',
        'Caption', 'Title'
    ]
    return category in mergeable_types


def adjust_text_bbox_for_figure(text_bbox, figure_bbox, image_padding=5):
    """
    ENHANCED: Create larger keep-away zones around figures
    Think of it like parking spaces - you need buffer room!
    """
    tx1, ty1, tx2, ty2 = text_bbox
    fx1, fy1, fx2, fy2 = figure_bbox

    # Create BIGGER padded zone (like a safety bubble)
    padded_fx1 = fx1 - image_padding
    padded_fy1 = fy1 - image_padding  
    padded_fx2 = fx2 + image_padding
    padded_fy2 = fy2 + image_padding
    
    padded_figure_bbox = [padded_fx1, padded_fy1, padded_fx2, padded_fy2]
    
    # If no overlap, we're good!
    if not bboxes_overlap(text_bbox, padded_figure_bbox, threshold=0.05):
        return text_bbox
    
    # Find the BEST way to move text (biggest remaining area)
    adjustments = []
    extra_gap = 1  # Additional safety margin
    
    # Option 1: Move up (text ends before figure starts)
    if padded_fy1 > ty1:
        top_bbox = [tx1, ty1, tx2, padded_fy1 - extra_gap]
        if top_bbox[3] > top_bbox[1] and (top_bbox[3] - top_bbox[1]) > 15:  # Minimum height
            area = (top_bbox[2] - top_bbox[0]) * (top_bbox[3] - top_bbox[1])
            adjustments.append(('top', top_bbox, area))
    
    # Option 2: Move down (text starts after figure ends)
    if padded_fy2 < ty2:
        bottom_bbox = [tx1, padded_fy2 + extra_gap, tx2, ty2]
        if bottom_bbox[3] > bottom_bbox[1] and (bottom_bbox[3] - bottom_bbox[1]) > 15:
            area = (bottom_bbox[2] - bottom_bbox[0]) * (bottom_bbox[3] - bottom_bbox[1])
            adjustments.append(('bottom', bottom_bbox, area))
    
    # Choose adjustment with LARGEST remaining area
    if adjustments:
        best_adjustment = max(adjustments, key=lambda x: x[2])
        return best_adjustment[1]
    
    # If no good adjustment, return MUCH smaller bbox to avoid overlap
    # Better to have small text than overlapping text
    return [tx1, ty1, min(tx2, padded_fx1 - extra_gap), min(ty2, padded_fy1 - extra_gap)]


def create_better_figure_zones(matched_elements, scale_factor):
    """
    STEP 1: Process ALL figures first to create no-go zones
    Like marking all the furniture before placing decorations
    """
    figure_zones = []
    
    for element in matched_elements:
        category = element['category']
        if category in ['Picture', 'Figure', 'Formula','Section-header','Caption']:
            bbox = element['bbox']
            # Scale the bbox
            scaled_bbox = [
                bbox[0] * scale_factor,
                bbox[1] * scale_factor, 
                bbox[2] * scale_factor,
                bbox[3] * scale_factor
            ]
            # Add large padding zone
            padding = 5  # Bigger safety zone
            padded_zone = [
                scaled_bbox[0] - padding,
                scaled_bbox[1] - padding,
                scaled_bbox[2] + padding, 
                scaled_bbox[3] + padding
            ]
            figure_zones.append(padded_zone)
    
    return figure_zones


def avoid_all_figure_zones(text_bbox, figure_zones):
    """
    STEP 2: Check text against ALL figure zones at once
    Like checking if a new furniture piece fits without hitting anything
    """
    current_bbox = text_bbox
    
    for figure_zone in figure_zones:
        if bboxes_overlap(current_bbox, figure_zone, threshold=0.05):
            # Adjust to avoid this figure zone
            current_bbox = adjust_text_bbox_for_figure(current_bbox, figure_zone, image_padding=0)
            
            # If bbox becomes too small, skip this text
            width = current_bbox[2] - current_bbox[0]
            height = current_bbox[3] - current_bbox[1]
            if width < 30 or height < 20:
                return None  # Text too small to be useful
    
    return current_bbox


def merge_overlapping_elements(layout_data):
    """Merge overlapping elements, but ONLY for mergeable types (figures, headers, captions, titles)"""
    merged_elements = []
    used_indices = set()
    
    for i, element in enumerate(layout_data):
        if i in used_indices:
            continue
            
        current_element = element.copy()
        current_bbox = element['bbox']
        current_category = element.get('category', 'Text')
        
        # Only consider merging if current element is mergeable
        if not is_mergeable_element(current_category):
            # Don't merge text, list-items, etc. - keep them as separate OCR text
            merged_elements.append(current_element)
            used_indices.add(i)
            continue
        
        # Look for overlapping elements (but only among mergeable types)
        overlapping_indices = []
        for j, other_element in enumerate(layout_data):
            if i != j and j not in used_indices:
                other_category = other_element.get('category', 'Text')
                # Only consider overlaps with other mergeable elements
                if is_mergeable_element(other_category) and bboxes_overlap(current_bbox, other_element['bbox']):
                    overlapping_indices.append(j)
        
        # If there are overlapping mergeable elements, merge them
        if overlapping_indices:
            # Determine priority (Picture > Section-header > Caption > Title)
            priority_order = ['Picture', 'Figure', 'Section-header', 'Page-header', 'Page-footer', 'Caption', 'Title', 'List-item','Text']
            all_elements = [current_element] + [layout_data[j] for j in overlapping_indices]
            
            # Sort by priority
            def get_priority(elem):
                category = elem.get('category', 'Text')
                try:
                    return priority_order.index(category)
                except ValueError:
                    return len(priority_order)
            
            sorted_elements = sorted(all_elements, key=get_priority)
            primary_element = sorted_elements[0]
            
            # Calculate union bbox
            union_bbox = current_bbox
            for j in overlapping_indices:
                union_bbox = union_bboxes(union_bbox, layout_data[j]['bbox'])
            
            # Create merged element with primary category and union bbox
            merged_element = primary_element.copy()
            merged_element['bbox'] = union_bbox
            merged_element['merged_from'] = [elem.get('category', 'Text') for elem in all_elements]
            
            merged_elements.append(merged_element)
            used_indices.add(i)
            used_indices.update(overlapping_indices)
        else:
            # Single mergeable element with no overlaps
            merged_elements.append(current_element)
            used_indices.add(i)
    
    return merged_elements


def clean_text(text: str) -> str:
    """Clean and escape text for HTML, but preserve subscript/superscript and math expressions"""
    if not text:
        return ""
    
    text = text.strip()
    
    # STEP 1: Protect math expressions by replacing them with placeholders
    math_placeholders = {}
    placeholder_counter = 0
    
    def protect_math(match):
        nonlocal placeholder_counter
        placeholder = f"__MATH_PLACEHOLDER_{placeholder_counter}__"
        math_placeholders[placeholder] = match.group(0)
        placeholder_counter += 1
        return placeholder
    
    # Protect display math ($$...$$) - multiline
    text = re.sub(r'\$\$.*?\$\$', protect_math, text, flags=re.DOTALL)
    
    # Protect inline math ($...$)
    text = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', protect_math, text)
    
    # STEP 2: Protect subscript and superscript tags
    sub_sup_placeholders = {}
    
    def protect_sub_sup(match):
        nonlocal placeholder_counter
        placeholder = f"__SUBSUP_PLACEHOLDER_{placeholder_counter}__"
        sub_sup_placeholders[placeholder] = match.group(0)
        placeholder_counter += 1
        return placeholder
    
    # Protect <sub> and <sup> tags
    text = re.sub(r'<(sub|sup)>(.*?)</(sub|sup)>', protect_sub_sup, text, flags=re.IGNORECASE)
    
    # STEP 3: Handle bold formatting FIRST (before removing any asterisks)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'""([^"]+)""', r'<strong>\1</strong>', text)
    
    # STEP 4: Remove bullet symbols
    text = re.sub(r'^[•\-»]\s*', '', text)
    text = re.sub(r'\s*[•\-»]+\s*$', '', text)
    
    # STEP 5: Clean up remaining asterisks
    text = re.sub(r'\*+', '', text)
    
    # STEP 6: Escape HTML characters (but preserve our formatting tags)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    # STEP 7: Restore our formatting tags
    text = text.replace('&lt;strong&gt;', '<strong>')
    text = text.replace('&lt;/strong&gt;', '</strong>')
    text = text.replace('&lt;em&gt;', '<em>')
    text = text.replace('&lt;/em&gt;', '</em>')
    
    # STEP 8: Restore subscript/superscript tags
    for placeholder, tag_content in sub_sup_placeholders.items():
        text = text.replace(placeholder, tag_content)
    
    # STEP 9: Restore math expressions (these should NOT be HTML escaped)
    for placeholder, math_expr in math_placeholders.items():
        text = text.replace(placeholder, math_expr)
    
    return text


def parse_markdown_content(markdown_content):
    """Parse markdown content to extract different types of elements"""
    lines = markdown_content.split('\n')
    parsed_elements = []
    current_element = {'type': 'text', 'content': ''}
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            if current_element['content']:
                parsed_elements.append(current_element)
                current_element = {'type': 'text', 'content': ''}
            i += 1
            continue
            
        # Detect headers
        if line.startswith('# '):
            if current_element['content']:
                parsed_elements.append(current_element)
            current_element = {'type': 'title', 'content': line[2:].strip()}
            parsed_elements.append(current_element)
            current_element = {'type': 'text', 'content': ''}
        elif line.startswith('## '):
            if current_element['content']:
                parsed_elements.append(current_element)
            current_element = {'type': 'section-header', 'content': line[3:].strip()}
            parsed_elements.append(current_element)
            current_element = {'type': 'text', 'content': ''}
        # Handle multi-line display math blocks
        elif line == '$$':
            if current_element['content']:
                parsed_elements.append(current_element)
            
            # Collect all lines until closing $$
            formula_lines = ['$$']
            i += 1
            while i < len(lines):
                formula_line = lines[i].strip()
                formula_lines.append(formula_line)
                if formula_line == '$$':
                    break
                i += 1
            
            # Join all formula lines
            formula_content = '\n'.join(formula_lines)
            current_element = {'type': 'formula', 'content': formula_content}
            parsed_elements.append(current_element)
            current_element = {'type': 'text', 'content': ''}
        # Handle single-line display math
        elif line.startswith('$$') and line.endswith('$$') and len(line) > 4:
            if current_element['content']:
                parsed_elements.append(current_element)
            current_element = {'type': 'formula', 'content': line}
            parsed_elements.append(current_element)
            current_element = {'type': 'text', 'content': ''}
        # Detect list items - clean them properly
        elif line.startswith('»') or line.startswith('•') or line.startswith('-'):
            if current_element['content']:
                parsed_elements.append(current_element)
            # Clean up the bullet point content - remove leading bullet symbols
            clean_content = re.sub(r'^[»•\-]\s*', '', line).strip()
            current_element = {'type': 'list-item', 'content': clean_content}
            parsed_elements.append(current_element)
            current_element = {'type': 'text', 'content': ''}
        # Detect images
        elif line.startswith('!['):
            if current_element['content']:
                parsed_elements.append(current_element)
            current_element = {'type': 'picture', 'content': line}
            parsed_elements.append(current_element)
            current_element = {'type': 'text', 'content': ''}
        
        else:
            # Regular text
            if current_element['content']:
                current_element['content'] += ' ' + line
            else:
                current_element['content'] = line
        
        i += 1
    
    # Add the last element
    if current_element['content']:
        parsed_elements.append(current_element)
    
    return parsed_elements


import re

def match_content_to_layout(markdown_elements, layout_data):
    """Match markdown content to JSON layout elements based on type and content similarity.
    Fixes:
      - Do NOT borrow text for OCR-empty elements (incl. headers/captions/titles).
      - Require both type match AND non-trivial overlap to match from markdown.
      - Prefer layout's own OCR text when present but no confident match exists.
      - Avoid assigning the same markdown element to multiple layout items.
      - Gentle reading-order fallback for leftover plain text only.
    """
    def norm_tokens(s: str):
        if not s:
            return set()
        # letters/numbers, 2+ chars, lowercase
        return set(t for t in re.findall(r"[A-Za-z0-9]+", s.lower()) if len(t) >= 2)

    def content_overlap(a: str, b: str) -> float:
        ta, tb = norm_tokens(a), norm_tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        return inter / max(len(ta), len(tb))

    # Helper: normalize category for matching while preserving original for output
    def cat_norm(c: str) -> str:
        return (c or "").strip().lower()

    # Build a working list of markdown elements and a used mask
    available_md = list(markdown_elements)  # shallow copy is fine
    md_used = [False] * len(available_md)

    # Map normalized layout categories to acceptable markdown types
    def types_compatible(layout_cat_norm: str, md_type: str) -> bool:
        if layout_cat_norm == "title":
            return md_type == "title"
        if layout_cat_norm == "section-header":
            return md_type == "section-header"
        if layout_cat_norm == "formula":
            return md_type == "formula"
        if layout_cat_norm == "list-item":
            return md_type == "list-item"
        if layout_cat_norm in ["text", "caption", "page-footer", "page-header"]:
            return md_type == "text"
        # pictures and tables are handled elsewhere
        return False

    matched_elements = []
    pending_for_fallback = []  # collect plain text items that might need order fallback

    # --- First pass: strict matching (type + overlap); keep OCR-empty elements empty ---
    for layout_item in layout_data:
        lcat_orig = layout_item.get("category", "")  # preserve original case for output
        lcat = cat_norm(lcat_orig)
        layout_text = (layout_item.get("text") or "").strip()

        me = {
            "bbox": layout_item["bbox"],
            "category": lcat_orig,
            "layout_text": layout_text,
            "content": "",
            "type": "text",
            "merged_from": layout_item.get("merged_from", []),
        }

        # 1) Pure visual categories: pictures handled without text matching
        if lcat in ["picture"]:
            me["type"] = "picture"
            me["content"] = layout_text  # usually empty; rendering uses cropped image
            matched_elements.append(me)
            continue

        # 2) If this is an OCR-empty text-like element, KEEP IT EMPTY (no stealing)
        if lcat in ["text", "list-item", "caption", "title", "section-header", "page-header", "page-footer"] and not layout_text:
            me["content"] = ""
            me["type"] = "empty"
            matched_elements.append(me)
            continue

        # 3) Try to find a SINGLE best markdown element (type-compatible + overlap >= threshold)
        best_idx = None
        best_score = 0.0
        OVERLAP_THRESHOLD = 0.30  # tuneable; 0.25–0.35 is a good range

        for i, md in enumerate(available_md):
            if md_used[i] or md is None:
                continue
            md_type = md.get("type", "text")
            md_content = md.get("content", "").strip()

            # skip obviously empty markdown
            if not md_content:
                continue

            if not types_compatible(lcat, md_type):
                continue

            # Require non-trivial token overlap against layout_text
            score = content_overlap(layout_text, md_content)
            if score >= OVERLAP_THRESHOLD and score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            me["content"] = available_md[best_idx]["content"]
            me["type"] = available_md[best_idx]["type"]
            md_used[best_idx] = True
            matched_elements.append(me)
            continue

        # 4) If no good match: prefer the layout's own OCR text when meaningful
        if layout_text and len(layout_text) >= 3:
            # For headers/captions/titles, we still trust OCR if present
            if lcat == "title":
                me["content"] = layout_text
                me["type"] = "title"
            elif lcat == "section-header":
                me["content"] = layout_text
                me["type"] = "section-header"
            elif lcat == "caption":
                me["content"] = layout_text
                me["type"] = "text"  # caption is usually rendered as text if not cropped
            elif lcat == "list-item":
                me["content"] = layout_text
                me["type"] = "list-item"
            elif lcat in ["page-header", "page-footer"]:
                me["content"] = layout_text
                me["type"] = "text"
            elif lcat == "formula":
                # For formulas, downstream rendering prefers cropped image; keep as formula with any text
                me["content"] = layout_text
                me["type"] = "formula"
            else:
                # plain text
                me["content"] = layout_text
                me["type"] = "text"
            matched_elements.append(me)

            # Collect plain text (not headers/captions) for a possible order-based fallback refinement
            if lcat == "text":
                pending_for_fallback.append((len(matched_elements) - 1, layout_item))
            continue

        # 5) Otherwise, leave empty
        me["content"] = ""
        me["type"] = "empty"
        matched_elements.append(me)

    leftover_md_indices = [
        i for i, md in enumerate(available_md)
        if not md_used[i] and md is not None and md.get("type") == "text" and (md.get("content") or "").strip()
    ]
    if leftover_md_indices and pending_for_fallback:
        # Sort leftover markdown by their original order (already ordered)
        md_queue = leftover_md_indices[:]

        # Sort pending layout items by reading order (top-to-bottom, then left-to-right)
        def layout_key(li):
            # li is the original layout item; bbox = [x1, y1, x2, y2]
            x1, y1, x2, y2 = li["bbox"]
            return (y1, x1, y2, x2)

        pending_for_fallback.sort(key=lambda t: layout_key(t[1]))

        for matched_idx, layout_item in pending_for_fallback:
            if not md_queue:
                break
            # Skip if element already changed type/content by previous steps
            if matched_elements[matched_idx]["type"] != "text":
                continue
            # Assign next markdown text to this layout element if it still has OCR content
            if matched_elements[matched_idx]["content"]:
                # Keep the existing OCR content; do not overwrite—this fallback is conservative
                continue
            # Assign one md text (reading-order heuristic)
            md_i = md_queue.pop(0)
            matched_elements[matched_idx]["content"] = available_md[md_i]["content"]
            matched_elements[matched_idx]["type"] = "text"
            md_used[md_i] = True

    return matched_elements



def create_combined_positioned_html(pages_data, pdf_name, page_width=1000):
    """
    Create a single HTML file with all pages and navigation controls
    Features: Theme selection, font scaling, page navigation
    """
    # Generate individual page content
    all_pages_html = []
    
    for page_data in pages_data:
        original_clean_image = page_data['image']
        layout_data = page_data['layout']
        markdown_content = page_data['markdown']
        page_num = page_data['page_num']
        
        page_content = create_single_page_content(
            original_clean_image, layout_data, markdown_content, 
            page_num, page_width
        )
        all_pages_html.append(page_content)
    
    total_pages = len(all_pages_html)
    
    # Create the complete HTML document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{pdf_name} - Interactive Viewer</title>
    
    <!-- MathJax v3 config -->
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']]
            }},
            chtml: {{
                scale: 1,
                minScale: 0.5,
                matchFontHeight: false
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }},
            startup: {{
                pageReady: () => {{
                    console.log('MathJax v3 loaded successfully');
                    return MathJax.startup.defaultPageReady();
                }}
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    
    <style>
        :root {{
            --bg-color: #f7fafc;
            --page-bg-color: white;
            --text-color: #2d3748;
            --header-color: #1a365d;
            --list-color: #4a5568;
            --caption-color: #718096;
            --font-scale: 1;
            --nav-bg: rgba(45, 55, 72, 0.9);
            --nav-color: white;
        }}
        
        [data-theme="gray"] {{
            --bg-color: #4a5568;
            --page-bg-color: #718096;
            --text-color: #f7fafc;
            --header-color: white;
            --list-color: #e2e8f0;
            --caption-color: #cbd5e0;
            --nav-bg: rgba(26, 32, 44, 0.9);
        }}
        
        [data-theme="dark-green"] {{
            --bg-color: #2d5a2d;
            --page-bg-color: #3d6b3d;
            --text-color: white;
            --header-color: #c6f6d5;
            --list-color: #e6fffa;
            --caption-color: #b2f5ea;
            --nav-bg: rgba(22, 44, 22, 0.9);
        }}
        
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            transition: all 0.3s ease;
            overflow-x: auto;
        }}
        
        .controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2000;
            background: var(--nav-bg);
            color: var(--nav-color);
            padding: 15px;
            border-radius: 12px;
            display: flex;
            gap: 15px;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .control-group label {{
            font-size: 12px;
            font-weight: 600;
            margin-right: 5px;
        }}
        
        select, button {{
            padding: 5px 10px;
            border: none;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.2);
            color: var(--nav-color);
            cursor: pointer;
            font-size: 12px;
        }}
        
        button:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}

        /* Fix dropdown options visibility */
        select option {{
            background: #2d3748;
            color: white;
            padding: 5px;
        }}
        
        select option:hover {{
            background: #4a5568;
            color: white;
        }}
        
        .page-container {{
            position: relative;
            width: {page_width}px;
            margin: 80px auto 40px;
            background-color: var(--page-bg-color);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            border-radius: 8px;
            overflow: visible;
            transition: all 0.3s ease;
        }}
        
        .page {{
            display: none;
            position: relative;
            min-height: 600px;
        }}
        
        .page.active {{
            display: block;
        }}
        
        .page-info {{
            position: absolute;
            top: 10px;
            left: 20px;
            background: var(--nav-bg);
            color: var(--nav-color);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            z-index: 100;
        }}
        
        .navigation {{
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 20px;
            z-index: 1500;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .navigation.show {{
            opacity: 1;
        }}
        
        .nav-btn {{
            background: var(--nav-bg);
            color: var(--nav-color);
            border: none;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        
        .nav-btn:hover {{
            transform: scale(1.1);
            background: rgba(45, 55, 72, 1);
        }}
        
        .nav-btn:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
            transform: none;
        }}
        
        /* Responsive font sizing for all text elements */
        .text-element {{
            font-size: calc(14px * var(--font-scale));
            line-height: 1.4;
            color: var(--text-color);
            word-wrap: break-word;
            white-space: normal;
            overflow-wrap: break-word;
            hyphens: auto;
            overflow: visible;
            height: auto !important;
            min-height: 100%;
        }}

        .list-element {{
            font-size: calc(13px * var(--font-scale));
            line-height: 1.4;
            color: var(--list-color);
            word-wrap: break-word;
            white-space: normal;
            overflow-wrap: break-word;
            overflow: visible;
            height: auto !important;
            min-height: 100%;
        }}

        .caption-element {{
            font-size: calc(12px * var(--font-scale));
            font-style: italic;
            color: var(--caption-color);
            word-wrap: break-word;
            white-space: normal;
            overflow-wrap: break-word;
            overflow: visible;
            height: auto !important;
            min-height: 100%;
        }}

        .title-element {{
            font-size: calc(24px * var(--font-scale));
            font-weight: bold;
            color: var(--header-color);
            line-height: 1.2;
            word-wrap: break-word;
            white-space: normal;
            overflow-wrap: break-word;
            overflow: visible;
            height: auto !important;
            min-height: 100%;
        }}

        .section-header-element {{
            font-size: calc(18px * var(--font-scale));
            font-weight: 600;
            color: var(--text-color);
            line-height: 1.3;
            word-wrap: break-word;
            white-space: normal;
            overflow-wrap: break-word;
            overflow: visible;
            height: auto !important;
            min-height: 100%;
        }}

        /* Beautiful scrollable formulas with fade effects */
        .formula-element {{
            position: relative;
            overflow: visible !important;
        }}

        .formula-element mjx-container {{
            font-size: calc(1em * var(--font-scale)) !important;
            max-width: 100% !important;
            overflow-x: auto !important;
            overflow-y: visible !important;
            display: block !important;
            white-space: nowrap !important;
            padding: 5px 0;
            
            /* Custom scrollbar styling */
            scrollbar-width: thin;
            scrollbar-color: rgba(0,0,0,0.3) transparent;
        }}

        /* Webkit scrollbar styling */
        .formula-element mjx-container::-webkit-scrollbar {{
            height: 6px;
        }}

        .formula-element mjx-container::-webkit-scrollbar-track {{
            background: rgba(0,0,0,0.1);
            border-radius: 3px;
        }}

        .formula-element mjx-container::-webkit-scrollbar-thumb {{
            background: rgba(0,0,0,0.3);
            border-radius: 3px;
        }}

        .formula-element mjx-container::-webkit-scrollbar-thumb:hover {{
            background: rgba(0,0,0,0.5);
        }}

                
        /* Responsive adjustments */
        @media (max-width: {page_width + 40}px) {{
            .page-container {{
                transform: scale(0.8);
                transform-origin: top center;
            }}
            .controls {{
                transform: scale(0.9);
                transform-origin: top right;
            }}
        }}
        
        @media (max-width: {int(page_width * 0.8) + 40}px) {{
            .page-container {{
                transform: scale(0.6);
                transform-origin: top center;
            }}
            .controls {{
                transform: scale(0.8);
                transform-origin: top right;
            }}
        }}

        /* Image popup styles */
        .popup-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 99999;
            cursor: pointer;
        }}
        
        .popup-image {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 95%;
            max-height: 95%;
            min-width: 400px;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }}
        
        .clickable-image {{
            cursor: pointer;
            transition: transform 0.2s ease;
        }}
        
        .clickable-image:hover {{
            transform: scale(1.05);
        }}

        /* Add spacing around images and formulas to prevent text overlap */
        .figure-element {{
            margin: 5px;
            padding: 3px;
        }}
        
        .formula-element {{
            margin: 5px;
            padding: 3px;
        }}
    </style>
</head>
<body data-theme="white">
    <div class="controls">
        <div class="control-group">
            <label>Theme:</label>
            <select id="themeSelect">
                <option value="white">White</option>
                <option value="gray">Gray</option>
                <option value="dark-green">Dark Green</option>
            </select>
        </div>
        <div class="control-group">
            <label>Font:</label>
            <button onclick="changeFontSize(-0.1)">A-</button>
            <button onclick="changeFontSize(0.1)">A+</button>
            <button onclick="resetFontSize()">Reset</button>
        </div>
        <div class="control-group">
            <span id="pageCounter">1 / {total_pages}</span>
        </div>
    </div>
    
    <div class="page-container">
        {''.join(f'<div class="page{" active" if i == 0 else ""}" id="page-{i+1}"><div class="page-info">Page {i+1} of {total_pages}</div>{page}</div>' for i, page in enumerate(all_pages_html))}
    </div>
    
    <div class="navigation" id="navigation">
        <button class="nav-btn" id="prevBtn" onclick="previousPage()">‹</button>
        <button class="nav-btn" id="nextBtn" onclick="nextPage()">›</button>
    </div>

    <div class="popup-overlay" id="popupOverlay">
        <img class="popup-image" id="popupImage" src="" alt="">
    </div>
    
    <script>
        let currentPage = 1;
        const totalPages = {total_pages};
        let fontScale = 1;
        
        // Theme switching
        document.getElementById('themeSelect').addEventListener('change', function(e) {{
            document.body.setAttribute('data-theme', e.target.value);
        }});
        
        // Font size controls
        function changeFontSize(delta) {{
            fontScale = Math.max(0.5, Math.min(2, fontScale + delta));
            document.documentElement.style.setProperty('--font-scale', fontScale);
            
            // Trigger MathJax re-render if available
            if (window.MathJax && window.MathJax.typesetPromise) {{
                window.MathJax.typesetPromise();
            }}
        }}
        
        function resetFontSize() {{
            fontScale = 1;
            document.documentElement.style.setProperty('--font-scale', fontScale);
            
            if (window.MathJax && window.MathJax.typesetPromise) {{
                window.MathJax.typesetPromise();
            }}
        }}
        
        // Page navigation
        function showPage(pageNum) {{
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.getElementById(`page-${{pageNum}}`).classList.add('active');
            
            currentPage = pageNum;
            document.getElementById('pageCounter').textContent = `${{currentPage}} / ${{totalPages}}`;
            
            // Update navigation buttons
            document.getElementById('prevBtn').disabled = currentPage === 1;
            document.getElementById('nextBtn').disabled = currentPage === totalPages;
        }}
        
        function nextPage() {{
            if (currentPage < totalPages) {{
                showPage(currentPage + 1);
            }}
        }}
        
        function previousPage() {{
            if (currentPage > 1) {{
                showPage(currentPage - 1);
            }}
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft') previousPage();
            if (e.key === 'ArrowRight') nextPage();
        }});
        
        // Show/hide navigation on mouse movement
        let mouseTimer;
        document.addEventListener('mousemove', function() {{
            document.getElementById('navigation').classList.add('show');
            clearTimeout(mouseTimer);
            mouseTimer = setTimeout(function() {{
                document.getElementById('navigation').classList.remove('show');
            }}, 2000);
        }});
        
        // Initialize
        showPage(1);
        
        // Show navigation initially, then hide after 3 seconds
        setTimeout(function() {{
            document.getElementById('navigation').classList.add('show');
            setTimeout(function() {{
                document.getElementById('navigation').classList.remove('show');
            }}, 3000);
        }}, 500);

        // Image popup functionality
        function initImagePopups() {{
            const clickableImages = document.querySelectorAll('.clickable-image');
            const popupOverlay = document.getElementById('popupOverlay');
            const popupImage = document.getElementById('popupImage');
            
            clickableImages.forEach(function(image) {{
                image.addEventListener('click', function(e) {{
                    e.stopPropagation();
                    popupImage.src = this.src;
                    popupImage.alt = this.alt;
                    popupOverlay.style.display = 'block';
                }});
            }});
            
            popupOverlay.addEventListener('click', function() {{
                popupOverlay.style.display = 'none';
            }});
            
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    popupOverlay.style.display = 'none';
                }}
            }});
        }}
        
        // Initialize popup functionality after page loads
        initImagePopups();
    </script>
    </body>
</html>"""
    
    return html_content


def create_single_page_content(original_clean_image, layout_data, markdown_content, page_num, page_width=1000, image_padding=2):
    """
    Create content for a single page with ENHANCED figure padding
    Returns only the positioned elements HTML, not a complete document
    """
    # Merge overlapping elements (only for mergeable types)
    merged_layout = merge_overlapping_elements(layout_data)
    
    # Parse markdown content
    markdown_elements = parse_markdown_content(markdown_content)
    
    # Match markdown content to layout elements
    matched_elements = match_content_to_layout(markdown_elements, merged_layout)

    # Get original image dimensions
    img_width, img_height = original_clean_image.size
    
    # Calculate scale factor
    scale_factor = page_width / img_width
    scaled_height = int(img_height * scale_factor)
    
    # Set the page height
    page_style = f"height: {scaled_height}px;"
    
    # Separate figures and text elements
    figure_elements = []
    text_elements = []
    
    for element in matched_elements:
        if 'bbox' not in element:
            continue
            
        category = element['category']
        
        if category in ['Picture','Formula']:
            figure_elements.append(element)
        else:
            text_elements.append(element)
    
    # NEW: Create figure zones FIRST (before processing any text)
    figure_zones = create_better_figure_zones(figure_elements, scale_factor)
    
    # Start building positioned elements
    html_elements = [f'<div style="{page_style}">']
    
    # STEP 1: Add figures first (includes formulas now)
    for i, element in enumerate(figure_elements):
        x1, y1, x2, y2 = element['bbox']
        category = element['category']
        
        # Scale coordinates
        scaled_x1 = int(x1 * scale_factor)
        scaled_y1 = int(y1 * scale_factor)
        scaled_x2 = int(x2 * scale_factor)
        scaled_y2 = int(y2 * scale_factor)
        
        width = scaled_x2 - scaled_x1
        height = scaled_y2 - scaled_y1
        
        # Handle formulas differently than regular figures
        if category == 'Formula':
            try:
                formula_crop = original_clean_image.crop((x1, y1, x2, y2))
                formula_base64 = PILimage_to_base64(formula_crop)
                
                html_element = f'''
                <div class="formula-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px; 
                        height: {height}px;
                        z-index: 50;
                        margin: 5px;
                        padding: 3px;">
                    <img src="{formula_base64}" 
                         class="clickable-image"
                         style="width: 100%; 
                                height: 100%; 
                                object-fit: contain;
                                border-radius: 4px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);"
                         alt="Formula {i+1}" />
                </div>
                '''
            except Exception:
                html_element = f'''
                <div class="formula-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px; 
                        height: {height}px;
                        background-color: var(--page-bg-color);
                        border: 2px dashed #ccc;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 4px;
                        z-index: 50;">
                    <span style="color: #666; font-size: 12px;">Formula {i+1}</span>
                </div>
                '''
        else:
            # Regular figure/picture processing
            try:
                image_crop = original_clean_image.crop((x1, y1, x2, y2))
                image_base64 = PILimage_to_base64(image_crop)
                
                html_element = f'''
                <div class="figure-element" style="position: absolute; 
                           left: {scaled_x1}px; 
                           top: {scaled_y1}px; 
                           width: {width}px; 
                           min-height: {height}px;
                           z-index: 50;">
                    <img src="{image_base64}" 
                         class="clickable-image"
                         style="width: 100%; 
                                height: 100%; 
                                object-fit: contain;
                                border-radius: 4px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);"
                         alt="Figure {i+1}" />
                </div>
                '''
            except Exception:
                html_element = f'''
                <div class="figure-element" style="position: absolute; 
                           left: {scaled_x1}px; 
                           top: {scaled_y1}px; 
                           width: {width}px; 
                           min-height: {height}px; 
                           background-color: var(--page-bg-color);
                           border: 2px dashed #ccc;
                           display: flex;
                           align-items: center;
                           justify-content: center;
                           border-radius: 4px;
                           z-index: 50;">
                    <span style="color: #666; font-size: 12px;">Image {i+1}</span>
                </div>
                '''
        
        html_elements.append(html_element)
    
    # STEP 2: Add text elements with ENHANCED overlap avoidance
    for i, element in enumerate(text_elements):
        x1, y1, x2, y2 = element['bbox']
        category = element['category']
        content = element['content']
        
        # Skip empty elements - don't render anything if no content
        if not content or content.strip() == '' or element.get('type') == 'empty':
            continue
        
        # Scale original bbox
        scaled_bbox = [
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor),
            int(y2 * scale_factor)
        ]
        
        # NEW: Check against ALL figure zones at once
        adjusted_bbox = avoid_all_figure_zones(scaled_bbox, figure_zones)
        
        # If text area too small after adjustment, skip it
        if adjusted_bbox is None:
            continue
            
        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = adjusted_bbox
        width = scaled_x2 - scaled_x1
        height = scaled_y2 - scaled_y1
        
        if width < 10 or height < 10:
            continue
        
        if category == 'Formula':
            # Crop formula as image instead of using MathJax
            try:
                formula_crop = original_clean_image.crop((x1, y1, x2, y2))
                formula_base64 = PILimage_to_base64(formula_crop)
                
                html_element = f'''
                <div class="formula-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px; 
                        height: {height}px;
                        z-index: 30;
                        margin: 5px;
                        padding: 3px;">

                    <img src="{formula_base64}" 
                         class="clickable-image"
                         style="width: 100%; 
                                height: 100%; 
                                object-fit: contain;
                                border-radius: 4px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);"
                         alt="Formula {i+1}" />
                </div>
                '''
            except Exception:
                # Fallback placeholder
                html_element = f'''
                <div class="formula-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px; 
                        height: {height}px;
                        background-color: var(--page-bg-color);
                        border: 2px dashed #ccc;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 4px;
                        z-index: 30;">
                    <span style="color: #666; font-size: 12px;">Formula {i+1}</span>
                </div>
                '''

        elif category in ['Section-header', 'Page-header', 'Page-footer']:
            # Try to crop as image first, fallback to text
            try:
                section_crop = original_clean_image.crop((x1, y1, x2, y2))
                section_base64 = PILimage_to_base64(section_crop)
                
                html_element = f'''
                <div style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px; 
                        height: {height}px;
                        z-index: 15;">
                    <img src="{section_base64}" 
                        style="width: 100%; 
                                height: 100%; 
                                object-fit: contain;
                                border-radius: 2px;"
                        alt="{category} {i+1}" />
                </div>
                '''
            except Exception:
                # Fallback to text rendering
                clean_content = clean_text(content)
                html_element = f'''
                <h2 class="section-header-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px;
                        min-height: {height}px;
                        margin: 0;
                        padding: 5px;
                        display: flex;
                        align-items: center;
                        overflow: visible;
                        background: var(--page-bg-color);
                        border-radius: 4px;
                        z-index: 5;">
                    {clean_content}
                </h2>
                '''

        elif category == 'Caption':
            # Try to crop as image first, fallback to text
            try:
                caption_crop = original_clean_image.crop((x1, y1, x2, y2))
                caption_base64 = PILimage_to_base64(caption_crop)
                
                html_element = f'''
                <div style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px; 
                        height: {height}px;
                        z-index: 15;">
                    <img src="{caption_base64}" 
                        style="width: 100%; 
                                height: 100%; 
                                object-fit: contain;
                                border-radius: 2px;"
                        alt="Caption {i+1}" />
                </div>
                '''
            except Exception:
                # Fallback to text rendering
                clean_content = clean_text(content)
                html_element = f'''
                <div class="caption-element" style="position: absolute; 
                            left: {scaled_x1}px; 
                            top: {scaled_y1}px; 
                            width: {width}px;
                            min-height: {height}px;
                            padding: 2px 5px;
                            text-align: center;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            overflow: visible;
                            background: var(--page-bg-color);
                            border-radius: 4px;
                            z-index: 5;">
                    {clean_content}
                </div>
                '''
                
        elif category == 'Title':
            # Crop titles as images if they're mergeable, otherwise render as text
            if is_mergeable_element(category):
                try:
                    title_crop = original_clean_image.crop((x1, y1, x2, y2))
                    title_base64 = PILimage_to_base64(title_crop)
                    
                    html_element = f'''
                    <div style="position: absolute; 
                            left: {scaled_x1}px; 
                            top: {scaled_y1}px; 
                            width: {width}px; 
                            height: {height}px;
                            z-index: 5;">
                        <img src="{title_base64}" 
                            style="width: 100%; 
                                    height: 100%; 
                                    object-fit: contain;
                                    border-radius: 2px;"
                            alt="Title {i+1}" />
                    </div>
                    '''
                except Exception:
                    # Fallback to text rendering
                    clean_content = clean_text(content)
                    html_element = f'''
                    <h1 class="title-element" style="position: absolute; 
                            left: {scaled_x1}px; 
                            top: {scaled_y1}px; 
                            width: {width}px;
                            min-height: {height}px;
                            margin: 0;
                            padding: 5px;
                            display: flex;
                            align-items: center;
                            overflow: visible;
                            background: var(--page-bg-color);
                            border-radius: 4px;
                            z-index: 5;">
                        {clean_content}
                    </h1>
                    '''
            else:
                # Render as text
                clean_content = clean_text(content)
                html_element = f'''
                <h1 class="title-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px;
                        min-height: {height}px;
                        margin: 0;
                        padding: 5px;
                        display: flex;
                        align-items: center;
                        overflow: visible;
                        background: var(--page-bg-color);
                        border-radius: 4px;
                        z-index: 5;">
                    {clean_content}
                </h1>
                '''
                
        elif category == 'List-item':
            clean_content = clean_text(content)
            html_element = f'''
            <div class="list-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px;
                        min-height: {height}px;
                        padding: 2px 5px;
                        display: flex;
                        align-items: flex-start;
                        overflow: visible;
                        background: var(--page-bg-color);
                        border-radius: 4px;
                        z-index: 5;">
                <span style="margin-right: 8px; flex-shrink: 0;">•</span>
                <span>{clean_content}</span>
            </div>
            '''
            
        elif category == 'Table':
            # Crop table from original image instead of rendering as HTML text
            try:
                x1, y1, x2, y2 = element['bbox']  # Fixed: use element, not element_data
                table_crop = original_clean_image.crop((x1, y1, x2, y2))
                table_base64 = PILimage_to_base64(table_crop)
                
                html_element = f'''
                <div class="table-element" style="position: absolute; 
                            left: {scaled_x1}px; 
                            top: {scaled_y1}px; 
                            width: {width}px;
                            min-height: {height}px;
                            height: auto;
                            padding: 1px;
                            border: 1px solid #e2e8f0;
                            border-radius: 4px;
                            background-color: var(--page-bg-color);
                            z-index: 20;">
                    <img src="{table_base64}" 
                        class="clickable-image"
                        style="width: 100%; 
                                height: 100%; 
                                object-fit: contain;
                                border-radius: 4px;"
                        alt="Table {i+1}" />
                </div>
                '''
            except Exception as e:
                print(f"Warning: Could not crop table image: {e}")
                # Fallback if image cropping fails
                html_element = f'''
                <div class="table-element" style="position: absolute; 
                            left: {scaled_x1}px; 
                            top: {scaled_y1}px; 
                            width: {width}px;
                            min-height: {height}px;
                            height: auto;
                            padding: 1px;
                            background-color: var(--page-bg-color);
                            border: 2px dashed #ccc;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            border-radius: 4px;
                            z-index: 20;">
                    <span style="color: #666; font-size: 12px;">Table {i+1}</span>
                </div>
                '''
            
        else:  # Default for 'Text' and other non-mergeable categories
            clean_content = clean_text(content)
            html_element = f'''
            <div class="text-element" style="position: absolute; 
                        left: {scaled_x1}px; 
                        top: {scaled_y1}px; 
                        width: {width}px;
                        min-height: {height}px;
                        padding: 3px 5px;
                        overflow: visible;
                        text-align: justify;
                        background: var(--page-bg-color);
                        border-radius: 4px;
                        z-index: 5;">
                {clean_content}
            </div>
            '''
        
        html_elements.append(html_element)

    html_elements.append('</div>')
    return ''.join(html_elements)

# ============================================================================
# FASTAPI SERVER IMPLEMENTATION
# ============================================================================

# Global variables
TEMP_DIR = "temp_pipeline"
DOTS_OUTPUT_DIR = "dots_output"
PROCESSING_LOCK = asyncio.Lock()
CLEANUP_INTERVAL = 60  # 1 minute - check more frequently since we clean faster

# Storage for processing jobs
processing_jobs = {}

class ProcessingStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

def init_temp_directory():
    """Initialize temporary directories - clean if they exist"""
    # Clean temp pipeline directory
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Error cleaning temp directory: {e}")
    
    # Clean dots output directory
    if os.path.exists(DOTS_OUTPUT_DIR):
        try:
            shutil.rmtree(DOTS_OUTPUT_DIR)
        except Exception as e:
            print(f"Error cleaning dots output directory: {e}")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(DOTS_OUTPUT_DIR, exist_ok=True)

def cleanup_old_files():
    """Remove files 2 minutes after processing completes and server is idle"""
    # Check if server is idle (no active processing)
    active_jobs = [job for job in processing_jobs.values() 
                  if job["status"] in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]]
    
    if active_jobs:
        return  # Don't clean while server is busy
    
    cutoff_time = datetime.now() - timedelta(minutes=2)
    removed_count = 0
    
    # Get job directories that are safe to clean (completed > 2 minutes ago)
    safe_to_clean_dirs = set()
    jobs_to_remove = []
    
    for job_id, job in processing_jobs.items():
        if job["status"] in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            completion_time_str = job.get("completed_at") or job.get("created_at")
            try:
                completion_time = datetime.fromisoformat(completion_time_str)
                if completion_time < cutoff_time:
                    safe_to_clean_dirs.add(job["job_dir"])
                    jobs_to_remove.append(job_id)
            except Exception:
                pass
    
    # Clean job directories that are safe to clean
    for job_dir in safe_to_clean_dirs:
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
                removed_count += 1
            except Exception:
                pass
    
    # Remove completed jobs from memory
    for job_id in jobs_to_remove:
        try:
            del processing_jobs[job_id]
        except Exception:
            pass
    
    # Clean dots_output directory when server is idle
    if os.path.exists(DOTS_OUTPUT_DIR):
        dir_mod_time = datetime.fromtimestamp(os.path.getmtime(DOTS_OUTPUT_DIR))
        if dir_mod_time < cutoff_time:
            for item in os.listdir(DOTS_OUTPUT_DIR):
                item_path = os.path.join(DOTS_OUTPUT_DIR, item)
                try:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                    if mod_time < cutoff_time:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        removed_count += 1
                except Exception:
                    pass

async def periodic_cleanup():
    """Background task for periodic cleanup"""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        cleanup_old_files()

def process_pdf_to_html_server(pdf_path: str, job_id: str) -> Optional[str]:
    """
    Server version of PDF processing - NO INDEX.HTML
    Returns ZIP file path if successful, None if failed
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        # Update job status
        processing_jobs[job_id]["status"] = ProcessingStatus.PROCESSING
        processing_jobs[job_id]["message"] = "Initializing DotsOCR..."
        
        # Initialize DotsOCR parser
        dots_parser = DotsOCRParser(
            ip='localhost',
            port=8000,
            model_name='./weights/DotsOCR',# set the location of weights
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=1,
            dpi=200,
            output_dir=DOTS_OUTPUT_DIR,
            use_hf=True
        )
        
        # Update job status
        processing_jobs[job_id]["message"] = "Processing PDF with DotsOCR..."
        # Process PDF with DotsOCR
        results = dots_parser.parse_file(
            input_path=pdf_path,
            prompt_mode="prompt_layout_all_en"
        )
        
        # Update job status
        processing_jobs[job_id]["message"] = "Loading clean images..."
        
        # Load original clean images from PDF
        original_images = load_images_cached(pdf_path, dpi=200)

        # Create HTML output directory
        pdf_name = Path(pdf_path).stem
        job_dir = os.path.join(TEMP_DIR, job_id)
        html_output_dir = os.path.join(job_dir, "html_output")
        os.makedirs(html_output_dir, exist_ok=True)
        
        # Update job status
        processing_jobs[job_id]["message"] = "Creating HTML pages..."
        processing_jobs[job_id]["total_pages"] = len(results)
        
        # Process each page
        # Collect all page data for combined 
        pages_data = []
        
        successful_pages = 0

        for page_result in results:
            page_num = page_result['page_no'] + 1
            page_index = page_result['page_no']
            
            # Update progress
            processing_jobs[job_id]["current_page"] = page_num
            processing_jobs[job_id]["message"] = f"Processing page {page_num}..."
            
            try:
                # Load the layout JSON
                if 'layout_info_path' not in page_result:
                    continue
                    
                with open(page_result['layout_info_path'], 'r', encoding='utf-8') as f:
                    layout_data = json.load(f)
                
                # Load the markdown content
                markdown_path = None
                if 'md_content_nohf_path' in page_result:
                    markdown_path = page_result['md_content_nohf_path']
                elif 'md_content_path' in page_result:
                    markdown_path = page_result['md_content_path']
                
                if not markdown_path or not os.path.exists(markdown_path):
                    continue
                    
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Get the clean original image
                if page_index >= len(original_images):
                    continue
                    
                original_clean_image = original_images[page_index]
                
                # Collect page data
                page_data = {
                    'image': original_clean_image,
                    'layout': layout_data,
                    'markdown': markdown_content,
                    'page_num': page_num
                }
                pages_data.append(page_data)
                successful_pages += 1
                    
            except Exception:
                continue

        if successful_pages > 0:
            # Update job status
            processing_jobs[job_id]["message"] = "Creating combined HTML file..."
            
            # Create single combined HTML file
            combined_html = create_combined_positioned_html(pages_data, pdf_name, page_width=1000)
            
            # Save combined HTML file
            html_file_path = os.path.join(html_output_dir, f"{pdf_name}_combined.html")
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(combined_html)
            
            # Create ZIP file with the combined HTML
            zip_path = os.path.join(job_dir, f"{pdf_name}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(html_file_path, f"{pdf_name}_combined.html")
            
            # Update job status
            processing_jobs[job_id]["status"] = ProcessingStatus.COMPLETED
            processing_jobs[job_id]["message"] = f"Successfully created combined HTML with {successful_pages} pages"
            processing_jobs[job_id]["zip_path"] = zip_path
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            return zip_path
        else:
            processing_jobs[job_id]["status"] = ProcessingStatus.FAILED
            processing_jobs[job_id]["message"] = "No pages were successfully processed"
            processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            return None
            
    except Exception as e:
        processing_jobs[job_id]["status"] = ProcessingStatus.FAILED
        processing_jobs[job_id]["message"] = f"Processing failed: {str(e)}"
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        return None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup logic
    print("Starting PDF to HTML Converter server...")
    init_temp_directory()
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield  # Application runs here
    
    # Shutdown logic
    print("Shutting down server...")
    if cleanup_task and not cleanup_task.done():
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            print("Cleanup task cancelled")
    
    cleanup_old_files()  # Final cleanup
    print("Shutdown completed")

app = FastAPI(
    title="PDF to HTML Converter",
    description="Convert PDF files to positioned HTML pages",
    version="1.0.0",
    lifespan=lifespan 
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PDF to HTML Converter API",
        "version": "1.0.0",
        "status": "running",
        "temp_dir": TEMP_DIR,
        "features": [
            "Single PDF processing",
            "Combined HTML with all pages",
            "Interactive page navigation with hover arrows", 
            "Theme selection (White/Gray/Dark Green)",
            "Dynamic font scaling with text wrapping",
            "Smart bbox adjustment", 
            "Formula MathJax rendering",
            "Auto cleanup 2 minutes after completion when idle"
            ]
    }

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload PDF file and start processing
    Returns job_id for tracking progress
    """
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File size too large (max 50MB)")
    
    # Check if server is busy (only one PDF at a time)
    async with PROCESSING_LOCK:
        active_jobs = [job for job in processing_jobs.values() 
                      if job["status"] in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]]
        
        if active_jobs:
            raise HTTPException(status_code=429, detail="Server is busy processing another PDF. Please try again later.")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job directory
        job_dir = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Save uploaded file
        pdf_path = os.path.join(job_dir, file.filename)
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize job status
        processing_jobs[job_id] = {
            "status": ProcessingStatus.PENDING,
            "filename": file.filename,
            "pdf_path": pdf_path,
            "job_dir": job_dir,
            "created_at": datetime.now().isoformat(),
            "message": "PDF uploaded successfully",
            "total_pages": 0,
            "current_page": 0
        }
        
        # Start processing in background
        def process_in_background():
            try:
                zip_path = process_pdf_to_html_server(pdf_path, job_id)
                if not zip_path:
                    processing_jobs[job_id]["status"] = ProcessingStatus.FAILED
            except Exception as e:
                processing_jobs[job_id]["status"] = ProcessingStatus.FAILED
                processing_jobs[job_id]["message"] = f"Processing error: {str(e)}"
        
        background_tasks.add_task(process_in_background)
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": ProcessingStatus.PENDING,
            "message": "PDF uploaded successfully. Processing started."
        }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status for a job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "message": job["message"],
        "created_at": job["created_at"]
    }
    
    # Add progress info if available
    if "total_pages" in job and job["total_pages"] > 0:
        response["progress"] = {
            "current_page": job.get("current_page", 0),
            "total_pages": job["total_pages"],
            "percentage": round((job.get("current_page", 0) / job["total_pages"]) * 100, 1)
        }
    
    if job["status"] == ProcessingStatus.COMPLETED:
        response["completed_at"] = job.get("completed_at")
        response["download_ready"] = True
    
    return response

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the ZIP file for a completed job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    
    zip_path = job.get("zip_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    # Get original filename without extension and add .zip
    original_name = Path(job["filename"]).stem
    download_filename = f"{original_name}.zip"
    
    return FileResponse(
        path=zip_path,
        filename=download_filename,
        media_type="application/zip"
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {
        "jobs": {
            job_id: {
                "status": job["status"],
                "filename": job["filename"],
                "created_at": job["created_at"],
                "message": job["message"]
            }
            for job_id, job in processing_jobs.items()
        },
        "total_jobs": len(processing_jobs)
    }

@app.delete("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup (for admin)"""
    cleanup_old_files()
    return {"message": "Cleanup completed - removed files from completed jobs older than 2 minutes"}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a specific job and its files"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    # Remove job directory
    if os.path.exists(job["job_dir"]):
        shutil.rmtree(job["job_dir"])
    
    # Remove from memory
    del processing_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "server:app", 
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )