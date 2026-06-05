#!/usr/bin/env python3
"""Import ECBIT paper references into Zotero with PDF downloads."""

import sqlite3
import json
import os
import re
import hashlib
import time
import shutil
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

ZOTERO_DB = os.path.expanduser("~/Zotero/zotero.sqlite")
ZOTERO_STORAGE = os.path.expanduser("~/Zotero/storage")
BIB_FILE = os.path.expanduser("~/Zotero/tmp_ecbit_import.json")  # We'll create a JSON intermediate
COLLECTION_NAME = "ECBIT — Antarctic AWS Block Imputation"

# ---- helpers ----
def make_key():
    """Generate a Zotero-style 8-char random key."""
    import random, string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def make_item_key():
    return make_key()

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ---- parse bibtex manually (simpler than bibtexparser dependency issues) ----
def parse_bibtex(path):
    """Simple BibTeX parser returning list of dicts."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    entries = []
    # Split on @article{ or @inproceedings{
    pattern = re.compile(r'@(\w+)\s*\{\s*(\w+)\s*,\s*(.*?)\}\s*$', re.DOTALL | re.MULTILINE)
    # Simpler approach: find all @ blocks
    blocks = re.findall(r'@(\w+)\{(\w+),\s*(.*?)\n\}', text, re.DOTALL)

    for entry_type, cite_key, fields_str in blocks:
        entry = {'type': entry_type, 'key': cite_key}
        # Parse fields
        # Match field = {value} or field = "value" or field = value
        field_pattern = re.compile(r'(\w+)\s*=\s*[\{"]((?:[^\{\}]|\{[^\}]*\})*?)[\}"]\s*,?\s*$', re.MULTILINE)
        # Simpler: split on \n and parse
        lines = fields_str.strip().split('\n')
        current_field = None
        current_value = []
        for line in lines:
            line = line.strip().rstrip(',')
            if not line:
                continue
            fmatch = re.match(r'(\w+)\s*=\s*\{(.*)', line)
            if fmatch:
                if current_field:
                    entry[current_field.lower()] = ' '.join(current_value).strip()
                current_field = fmatch.group(1).lower()
                val = fmatch.group(2)
                # Check if it's a closing brace on same line
                if val.rstrip().endswith('}') and not val.rstrip().endswith('\\}'):
                    val = val.rstrip()[:-1]
                    entry[current_field] = val.strip()
                    current_field = None
                    current_value = []
                else:
                    current_value = [val]
            elif current_field:
                if line.rstrip().endswith('}') and not line.rstrip().endswith('\\}'):
                    current_value.append(line.rstrip()[:-1])
                    entry[current_field] = ' '.join(current_value).strip()
                    current_field = None
                    current_value = []
                else:
                    current_value.append(line)

        if current_field and current_value:
            entry[current_field] = ' '.join(current_value).strip()

        # Clean up field values
        for k in list(entry.keys()):
            v = entry[k]
            # Remove wrapping braces
            v = v.strip()
            while v.startswith('{') and v.endswith('}'):
                v = v[1:-1].strip()
            # Unescape
            v = v.replace('\\&', '&').replace('\\%', '%').replace('\\$', '$')
            v = v.replace('\\#', '#').replace('\\_', '_').replace('\\~', '~')
            v = re.sub(r'\\(.)', r'\1', v)  # strip remaining backslash escapes
            v = ' '.join(v.split())  # normalize whitespace
            entry[k] = v

        entries.append(entry)

    return entries


# ---- Zotero SQLite operations ----
def get_or_create_library(cursor):
    cursor.execute("SELECT libraryID FROM libraries WHERE type='user' LIMIT 1")
    row = cursor.fetchone()
    if row:
        return row[0]
    lid = 1
    cursor.execute(
        "INSERT INTO libraries (libraryID, type, editable, filesEditable, name, storageVersion, lastSync, version) "
        "VALUES (?, 'user', 1, 1, 'My Library', 0, 0, 0)",
        (lid,)
    )
    return lid

def get_or_create_collection(cursor, library_id, name):
    cursor.execute(
        "SELECT collectionID FROM collections WHERE collectionName=? AND libraryID=?",
        (name, library_id)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    cid = cursor.execute("SELECT COALESCE(MAX(collectionID),0)+1 FROM collections").fetchone()[0]
    key = make_key()
    ts = now_str()
    cursor.execute(
        "INSERT INTO collections (collectionID, collectionName, parentCollectionID, clientDateModified, libraryID, key, version, synced) "
        "VALUES (?, ?, NULL, ?, ?, ?, 0, 0)",
        (cid, name, ts, library_id, key)
    )
    return cid

def get_item_type_id(cursor, type_name):
    cursor.execute("SELECT itemTypeID FROM itemTypes WHERE typeName=?", (type_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    # fallback
    cursor.execute("SELECT itemTypeID FROM itemTypes WHERE typeName='journalArticle'")
    return cursor.fetchone()[0]

def get_field_id(cursor, field_name):
    cursor.execute("SELECT fieldID FROM fields WHERE fieldName=?", (field_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    return None

def get_or_create_value(cursor, value):
    if not value:
        return None
    cursor.execute("SELECT valueID FROM itemDataValues WHERE value=?", (value,))
    row = cursor.fetchone()
    if row:
        return row[0]
    vid = cursor.execute("SELECT COALESCE(MAX(valueID),0)+1 FROM itemDataValues").fetchone()[0]
    cursor.execute("INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)", (vid, value))
    return vid

def get_creator_type_id(cursor, ctype_name):
    cursor.execute("SELECT creatorTypeID FROM creatorTypes WHERE creatorType=?", (ctype_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    return None

def add_creator(cursor, first_name, last_name):
    """Add a creator if not exists, return creatorID."""
    cursor.execute(
        "SELECT creatorID FROM creators WHERE firstName=? AND lastName=?",
        (first_name, last_name)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    cid = cursor.execute("SELECT COALESCE(MAX(creatorID),0)+1 FROM creators").fetchone()[0]
    cursor.execute(
        "INSERT INTO creators (creatorID, firstName, lastName) VALUES (?, ?, ?)",
        (cid, first_name, last_name)
    )
    return cid

def parse_authors(author_str):
    """Parse BibTeX author string into list of (first, last) tuples."""
    authors = []
    if not author_str:
        return authors
    # Split on ' and '
    parts = re.split(r'\s+and\s+', author_str)
    for part in parts:
        part = part.strip().rstrip(',').strip()
        if not part:
            continue
        # Handle "Last, First" or "First Last" or "Last, First1 and First2"
        if ',' in part:
            bits = part.split(',', 1)
            last = bits[0].strip()
            first = bits[1].strip() if len(bits) > 1 else ''
        else:
            bits = part.rsplit(None, 1)
            if len(bits) == 2:
                first, last = bits[0].strip(), bits[1].strip()
            else:
                last = part.strip()
                first = ''
        # Clean up
        last = re.sub(r'[\{\}]', '', last)
        first = re.sub(r'[\{\}]', '', first)
        authors.append((first, last))
    return authors

def insert_item(cursor, library_id, entry, item_type_name):
    """Insert a single bibliographic item and return itemID."""
    item_type_id = get_item_type_id(cursor, item_type_name)
    item_id = cursor.execute("SELECT COALESCE(MAX(itemID),0)+1 FROM items").fetchone()[0]
    key = make_item_key()
    ts = now_str()

    cursor.execute(
        """INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)""",
        (item_id, item_type_id, ts, ts, ts, library_id, key)
    )

    # Map BibTeX fields to Zotero fieldIDs
    field_map = {
        'title': 'title',
        'journal': 'publicationTitle',
        'booktitle': 'publicationTitle',
        'volume': 'volume',
        'number': 'issue',
        'pages': 'pages',
        'year': 'date',
        'doi': 'DOI',
        'url': 'url',
        'publisher': 'publisher',
        'abstract': 'abstractNote',
    }

    for bib_field, zotero_field in field_map.items():
        if bib_field in entry and entry[bib_field]:
            fid = get_field_id(cursor, zotero_field)
            if fid:
                val = entry[bib_field]
                if zotero_field == 'date':
                    val = str(val)
                vid = get_or_create_value(cursor, val)
                if vid:
                    cursor.execute(
                        "INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
                        (item_id, fid, vid)
                    )

    # Authors
    if 'author' in entry and entry['author']:
        authors = parse_authors(entry['author'])
        author_type_id = get_creator_type_id(cursor, 'author')
        if author_type_id:
            for idx, (first, last) in enumerate(authors):
                cid = add_creator(cursor, first, last)
                cursor.execute(
                    "INSERT INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex) VALUES (?, ?, ?, ?)",
                    (item_id, cid, author_type_id, idx)
                )

    return item_id, key


def add_to_collection(cursor, collection_id, item_id):
    cursor.execute(
        "INSERT OR IGNORE INTO collectionItems (collectionID, itemID) VALUES (?, ?)",
        (collection_id, item_id)
    )


def download_pdf(entry, storage_dir, zotero_item_key):
    """Try to download PDF for a reference. Returns path or None."""
    doi = entry.get('doi', '')
    title = entry.get('title', '')

    pdf_path = None

    # Strategy 1: arXiv DOI → direct PDF
    if 'arxiv' in doi.lower() or 'arXiv' in doi:
        arxiv_id = doi.split('/')[-1]
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = _try_download(url, storage_dir, zotero_item_key, title)
        if pdf_path:
            return pdf_path

    # Strategy 2: DOI → sci-hub / unpaywall / direct
    if doi and not pdf_path:
        # Try direct DOI resolution first
        url = f"https://doi.org/{doi}"
        try:
            req = urllib.request.Request(url, method='HEAD')
            req.add_header('User-Agent', 'Zotero/7.0')
            resp = urllib.request.urlopen(req, timeout=10)
            # Many publishers redirect to the article page, not PDF
            # We'll use unpaywall as a fallback
        except:
            pass

        # Try unpaywall
        if not pdf_path:
            try:
                uw_url = f"https://api.unpaywall.org/v2/{doi}?email=yongzheli@seu.edu.cn"
                req = urllib.request.Request(uw_url)
                req.add_header('User-Agent', 'Zotero/7.0')
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read().decode())
                best_loc = data.get('best_oa_location', {})
                pdf_url = best_loc.get('url_for_pdf') or best_loc.get('url')
                if pdf_url:
                    ret = _try_download(pdf_url, storage_dir, zotero_item_key, title)
                    if ret:
                        pdf_path = ret
            except:
                pass

    return pdf_path


def _try_download(url, storage_dir, item_key, title):
    """Try to download a file; return local path or None."""
    if not url:
        return None
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Zotero/7.0 (mailto:yongzheli@seu.edu.cn)')
        resp = urllib.request.urlopen(req, timeout=30)
        content = resp.read()
        if len(content) < 1000:
            return None  # too small to be a real PDF
        # Verify it's a PDF
        if not content[:5].startswith(b'%PDF'):
            return None

        safe_name = re.sub(r'[^a-zA-Z0-9_\-. ]', '', title or 'paper')[:80]
        fname = f"{item_key}_{safe_name}.pdf"
        fpath = os.path.join(storage_dir, fname)
        with open(fpath, 'wb') as f:
            f.write(content)
        return fpath
    except Exception:
        return None


def main():
    print("=" * 60)
    print("ECBIT → Zotero Import & PDF Download")
    print("=" * 60)

    # 1. Parse BibTeX
    bib_path = os.path.expanduser("~/Zotero/ecbit_references.bib")
    # Copy bib to Zotero dir first
    src_bib = "/home/horeb/_code/ECAFT/ECBIT/paper/references.bib"
    shutil.copy(src_bib, bib_path)
    print(f"\n[1] Parsing: {bib_path}")

    entries = parse_bibtex(bib_path)
    print(f"    Found {len(entries)} references")

    # 2. Open Zotero DB
    print(f"\n[2] Opening Zotero database: {ZOTERO_DB}")
    conn = sqlite3.connect(ZOTERO_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.cursor()

    # 3. Get library and create collection
    library_id = get_or_create_library(cursor)
    print(f"    Library ID: {library_id}")

    collection_id = get_or_create_collection(cursor, library_id, COLLECTION_NAME)
    print(f"    Collection '{COLLECTION_NAME}' ID: {collection_id}")

    # 4. Insert items
    print(f"\n[3] Importing {len(entries)} references...")
    item_type_map = {
        'article': 'journalArticle',
        'inproceedings': 'conferencePaper',
        'incollection': 'bookSection',
        'book': 'book',
        'misc': 'document',
    }

    item_keys = {}  # cite_key → (item_id, zotero_key)

    for i, entry in enumerate(entries):
        cite_key = entry.get('key', f'unknown_{i}')
        etype = entry.get('type', 'article')
        ztype = item_type_map.get(etype, 'journalArticle')

        title = entry.get('title', f'[Untitled: {cite_key}]')[:200]

        try:
            item_id, zkey = insert_item(cursor, library_id, entry, ztype)
            add_to_collection(cursor, collection_id, item_id)
            item_keys[cite_key] = (item_id, zkey)
            print(f"    [{i+1}/{len(entries)}] {cite_key}: {title[:70]}...")
        except Exception as e:
            print(f"    [{i+1}/{len(entries)}] ERROR {cite_key}: {e}")

    conn.commit()
    print(f"\n    Inserted {len(item_keys)} items into collection.")

    # 5. Download PDFs
    print(f"\n[4] Downloading PDFs...")
    pdf_downloaded = 0
    for entry in entries:
        cite_key = entry.get('key', '')
        if cite_key not in item_keys:
            continue
        item_id, zkey = item_keys[cite_key]

        storage_item_dir = os.path.join(ZOTERO_STORAGE, zkey)
        os.makedirs(storage_item_dir, exist_ok=True)

        pdf_path = download_pdf(entry, storage_item_dir, zkey)
        if pdf_path:
            # Register attachment in Zotero
            att_id = cursor.execute("SELECT COALESCE(MAX(itemID),0)+1 FROM items").fetchone()[0]
            att_key = make_item_key()
            ts = now_str()
            att_type_id = get_item_type_id(cursor, 'attachment')
            cursor.execute(
                """INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)""",
                (att_id, att_type_id, ts, ts, ts, library_id, att_key)
            )
            # Link as child of parent item
            cursor.execute(
                "INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path, syncState) "
                "VALUES (?, ?, 0, 'application/pdf', ?, 0)",
                (att_id, item_id, os.path.basename(pdf_path))
            )
            pdf_downloaded += 1
            print(f"    [PDF] {cite_key}: {os.path.basename(pdf_path)}")

    conn.commit()
    conn.close()

    # 6. Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  References imported:  {len(item_keys)} / {len(entries)}")
    print(f"  PDFs downloaded:      {pdf_downloaded}")
    print(f"  Collection:           {COLLECTION_NAME}")
    print(f"  Bib file saved:       {bib_path}")
    print(f"\n  Start Zotero to use 'Find Available PDFs' for remaining references.")
    print(f"  Database backup at: ~/Zotero/zotero.sqlite.bak.*")


if __name__ == '__main__':
    main()
