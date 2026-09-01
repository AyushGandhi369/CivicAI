import os
from functools import wraps
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
from groq import Groq
import pymupdf  # PyMuPDF for PDF text extraction + page rendering
from paddleocr import PaddleOCR
import uuid

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Secret key for Flask sessions (used to store document text in memory)
# In production, use a strong random key. For a hackathon, this is fine.
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Server-side document storage keyed by session ID.
# Flask cookie sessions have a ~4KB limit, which silently drops large
# document_text. This dict stores documents server-side instead.
# In production, use Redis or a database. For a hackathon, this works.
_document_store = {}

def get_session_id():
    """Get or create a unique session ID for this user."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

# Create Groq client using the API key from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ──────────────────────────────────────────────────────────────────────
# WHY CHUNKING IS NEEDED
#
# LLMs (Large Language Models) have a "context window" — a maximum amount
# of text they can read in a single request. This limit exists because:
#   1. The model holds the entire input in memory at once.
#   2. Processing cost grows rapidly with input length.
#   3. API providers enforce request size limits (Groq returns 413 if
#      the request is too large).
#
# WHY HIERARCHICAL SUMMARIZATION IS BETTER THAN TRUNCATION
#
# Truncation (cutting off text after N characters) throws away the end
# of the document — which often contains critical information like
# deadlines, conditions, or contact details.
#
# Hierarchical summarization keeps ALL the information:
#   Step 1: Split the document into manageable chunks.
#   Step 2: Summarize each chunk independently.
#   Step 3: Combine all summaries into one final, polished report.
#
# This way, no part of the document is ignored.
# ──────────────────────────────────────────────────────────────────────

# Maximum characters per chunk (~6000 chars keeps us well within API limits
# after the prompt template is added)
CHUNK_SIZE = 6000

PROMPT_TEMPLATE = """You are an elite document analyst. A user uploaded a document for professional analysis.
The text may be in any language — always respond in clear English.

══════════════════════════════════════════
STEP 1: INTERNAL REASONING (show this)
══════════════════════════════════════════

Before writing your analysis, output these three lines at the very top:

📄 [Generate a smart, descriptive title for this document — NOT "Untitled" or the filename]
📂 [Identify the document type: Resume, Research Paper, Invoice, Legal Contract, Meeting Notes, Business Proposal, Government Notice, Technical Documentation, Article, Financial Statement, Medical Report, Academic Notes, Policy, Manual, or Other]

Then generate the analysis below.

══════════════════════════════════════════
STEP 2: STRUCTURED ANALYSIS
══════════════════════════════════════════

Rules:
1. Use ONLY information present in the document. NEVER invent facts.
2. Skip any section that has no relevant content — never output empty headings.
3. Be specific: use exact names, numbers, dates, amounts, and percentages.
4. NEVER write "This document discusses..." or "The document provides..." or "It appears that..." — write directly about the content.
5. Optimize for scanning — the reader should understand 80% of the document within 30 seconds.
6. Use Markdown. Use **bold** for critical values. Use bullet points. Keep paragraphs short (2-3 sentences max).
7. Only use tables when comparing 2-6 structured items. Otherwise use bullet points.

Auto-adapt based on document type:
- Resume: experience, skills, education, projects, achievements
- Research Paper: objective, methodology, key results, limitations, conclusion
- Invoice: vendor, customer, invoice number, line items, amount, tax, due date
- Legal/Contract: obligations, rights, penalties, deadlines, risks
- Meeting Notes: attendees, decisions, action items, next steps
- Business Proposal: problem, solution, market, pricing, revenue model
- Technical Docs: architecture, technologies, APIs, workflow, implementation
- Government Notice: who it affects, what changed, deadlines, required actions
- Financial: revenue, expenses, profit, key ratios, trends

Output format — use ONLY relevant sections:

## 📝 Overview
4-6 sentences. Purpose, key conclusions, why it matters. Be direct.

## 🔑 Key Highlights
- Bullet points for important facts, findings, decisions, arguments
- **bold** critical values (amounts, dates, percentages)
- Skip trivial information

## 👤 Important People & Organizations
Names, roles, organizations mentioned. Only if they exist.

## 📅 Important Dates & Deadlines
Specific dates, deadlines, time periods. Only if they exist.

## 💰 Financial Information
Amounts, costs, revenue, taxes, percentages. Only if they exist.

## ⚠️ Risks & Warnings
Risks, concerns, warnings, penalties. Only if they exist.

## ✅ Action Items & Next Steps
Tasks, obligations, recommendations with owners and deadlines. Only if they exist.

## 💡 Final Takeaway
2-3 sentences: why this matters and what to remember.

══════════════════════════════════════════
STEP 3: SUGGESTED QUESTIONS
══════════════════════════════════════════

At the very end, output exactly this format:

---SUGGESTED_QUESTIONS---
1. [First intelligent follow-up question based on the document]
2. [Second question]
3. [Third question]
4. [Fourth question]
5. [Fifth question]

Make questions specific to THIS document, not generic.

══════════════════════════════════════════

Here is the document:

---
{text}
---"""

# Strong system-level instruction to guide the model's internal reasoning
SYSTEM_INSTRUCTION = """You are an expert senior document analyst. Before answering, perform these internal steps in order and use them to guide your final output: 1) Identify the document type (Resume, Research Paper, Invoice, Contract, Meeting Notes, Proposal, Policy, Technical Doc, Article, Financial Statement, Medical Report, Manual, Other). 2) Determine the primary purpose. 3) Extract entities (people, organizations, dates, deadlines, amounts, percentages, addresses, phones, emails, URLs, technical terms, legal refs, products). 4) Identify key findings, decisions, risks, recommendations, action items, and deadlines. Only after this internal analysis produce the final answer following the user-visible prompt. Always use only the provided text; never hallucinate or add external facts. Keep outputs scannable, professional, and concise."""
CHUNK_PROMPT = """You are an elite document analyst extracting information from Part {chunk_num} of {total_chunks} of a large document.
The text may be in any language — always respond in clear English.

Extract EVERY important detail. Be exhaustive — this will be combined with other parts later.

Extract:
1. All facts, findings, arguments, conclusions, and recommendations
2. All people, organizations, companies, and roles
3. All dates, deadlines, durations, and time references
4. All monetary values, percentages, statistics, and measurements
5. All obligations, requirements, conditions, penalties, and warnings
6. All technical terms, legal references, product names, and definitions
7. Any section headers or topic labels visible in this part

Rules:
- Use ONLY information in this text. Never invent anything.
- Include exact numbers, names, dates, and quotes.
- Use bullet points grouped by topic.
- Note which section or topic each detail comes from when visible.

---
{text}
---"""
# Chunk-level prompt revised above; keep as-is for compatibility

COMBINE_PROMPT = """You are an elite document analyst. A large document was analyzed in parts.
Below are all part summaries. Produce ONE polished final report.

First, output these two lines at the very top:

📄 [Generate a smart, descriptive title for this document]
📂 [Identify the document type]

Then produce the analysis using ONLY relevant sections:

## 📝 Overview
4-6 sentences. Purpose, key conclusions, why it matters.

## 🔑 Key Highlights
- Important facts with **bold** critical values

## 👤 Important People & Organizations
## 📅 Important Dates & Deadlines
## 💰 Financial Information
## ⚠️ Risks & Warnings
## ✅ Action Items & Next Steps
## 💡 Final Takeaway

Rules:
1. Merge related info from different parts. Eliminate ALL repetition.
2. Preserve EVERY important detail — never drop facts.
3. Never invent information.
4. NEVER write "This document discusses..." — write directly.
5. Optimize for scanning: bullets, bold values, short paragraphs.
6. Only use tables for comparing 2-6 structured items.
7. Skip sections with no content.

At the very end:

---SUGGESTED_QUESTIONS---
1. [First specific follow-up question]
2. [Second question]
3. [Third question]
4. [Fourth question]
5. [Fifth question]

Here are the part summaries:

---
{summaries}
---"""
# Combine prompt revised above; keep as-is for compatibility

# ──────────────────────────────────────────────────────────────────────
# WHY GROUNDING ANSWERS IN THE DOCUMENT REDUCES HALLUCINATIONS
#
# LLMs can "hallucinate" — confidently state things that aren't true.
# By including the actual document text in every question prompt, we
# force the AI to look at the source material before answering.
# The prompt explicitly says "only answer from the document" and
# "say you don't know if the answer isn't there." This is called
# "grounding" — tying the AI's response to real evidence.
#
# WHY CHUNKING IS REUSED FOR Q&A
#
# The same document that was too large for analysis is also too large
# to include in a Q&A prompt. We reuse split_text() to break the
# document into chunks, search each chunk for relevant answers, and
# then combine partial answers into one final response.
# ──────────────────────────────────────────────────────────────────────

QA_PROMPT = """You are an elite document analyst answering a question about a specific uploaded document.

Rules:
1. Answer ONLY from the document below. NEVER use outside knowledge.
2. Start with a direct answer — no preamble, no "Based on the document..."
3. Then explain with specific details, numbers, or evidence from the document.
4. If the question asks "why", explain the reasoning found in the document.
5. If multiple sections are relevant, combine them into one complete answer.
6. If the answer is NOT in the document: say "I couldn't find this information in the uploaded document." Then mention any related information that IS available.
7. Use Markdown: **bold** important values, bullet points for lists.
8. Keep paragraphs short (2-3 sentences max).

At the end of your answer, add:

📍 **Found in:** [name the section(s), paragraph(s), or area(s) of the document where you found this information]

🎯 **Confidence:** [High/Medium/Low] — [one-line reason]

Document:
---
{document}
---

Question: {question}
"""
# QA prompt revised above; keep as-is for compatibility

QA_CHUNK_PROMPT = """You are a document analyst. A user has a question about a large document.
Below is one part of that document.

If this part contains relevant information:
- Provide a direct, specific answer with exact details.
- Note which section or topic area the information comes from.
- Be concise.

If this part does NOT contain relevant information, respond with exactly: "NOT_FOUND_IN_THIS_CHUNK"

Never invent information.

Document part:
---
{document}
---

Question: {question}
"""
# QA chunk prompt revised above; keep as-is for compatibility

QA_COMBINE_PROMPT = """You are an elite document analyst. Combine partial answers about a document into one complete response.

Ignore any parts that say "NOT_FOUND_IN_THIS_CHUNK".

Rules:
1. Start with a direct answer — no preamble.
2. Combine all relevant information into one clear response.
3. Eliminate repetition.
4. Include specific details: exact names, numbers, dates.
5. If ALL parts say not found: "I couldn't find this information in the uploaded document."
6. Use Markdown: **bold** important values, bullet points for lists.
7. Keep paragraphs short.

At the end add:

📍 **Found in:** [sections/areas where this was found]

🎯 **Confidence:** [High/Medium/Low] — [reason]

Partial answers:
---
{answers}
---

Original question: {question}
"""
# QA combine prompt revised above; keep as-is for compatibility

# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def call_llm(prompt):
    """Make a single call to the Groq API and return the response text.

    This is the ONLY function that talks to the AI.
    To switch providers later, only modify this function.
    """
    # Send a system-level instruction to ensure the model performs the
    # required internal reasoning steps before producing any user-facing output.
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    return response.choices[0].message.content


def split_text(text):
    """Split a large text into chunks of approximately CHUNK_SIZE characters.

    Splitting strategy:
    1. First, try to split at paragraph boundaries (double newlines).
    2. If a paragraph itself is too long, split at single newlines.
    3. If a line is still too long, split at the last space before the limit.
    4. Never split inside a word.
    """
    # If the text already fits, return it as a single chunk
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    current_chunk = ""

    # Split into paragraphs (double newline)
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed the limit
        if len(current_chunk) + len(paragraph) + 2 <= CHUNK_SIZE:
            # It fits — add it to the current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # It doesn't fit — save the current chunk (if any) and start fresh
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Check if this single paragraph is itself too long
            if len(paragraph) <= CHUNK_SIZE:
                current_chunk = paragraph
            else:
                # Split the oversized paragraph at line breaks or spaces
                lines = paragraph.split("\n")
                for line in lines:
                    if len(current_chunk) + len(line) + 1 <= CHUNK_SIZE:
                        if current_chunk:
                            current_chunk += "\n" + line
                        else:
                            current_chunk = line
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""

                        # If a single line is still too long, split at spaces
                        while len(line) > CHUNK_SIZE:
                            # Find the last space before the limit
                            split_pos = line.rfind(" ", 0, CHUNK_SIZE)
                            if split_pos == -1:
                                split_pos = CHUNK_SIZE  # no space found, hard split
                            chunks.append(line[:split_pos].strip())
                            line = line[split_pos:].strip()

                        current_chunk = line

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def analyze_document(text):
    """Analyze a document, automatically chunking if it is too large.

    Small documents  → one AI call with the full prompt.
    Large documents  → chunk → summarize each → combine into one final report.
    """
    chunks = split_text(text)

    # ── Small document: single call ──
    if len(chunks) == 1:
        prompt = PROMPT_TEMPLATE.format(text=chunks[0])
        return call_llm(prompt)

    # ── Large document: hierarchical summarization ──
    print(f"[Chunking] Document split into {len(chunks)} chunks")

    summaries = []
    failed_chunks = []

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        print(f"[Chunking] Summarizing chunk {chunk_num}/{len(chunks)} "
              f"({len(chunk)} chars)")

        try:
            prompt = CHUNK_PROMPT.format(
                chunk_num=chunk_num,
                total_chunks=len(chunks),
                text=chunk
            )
            summary = call_llm(prompt)
            summaries.append(f"--- Part {chunk_num} ---\n{summary}")

        except Exception as e:
            # Log the failure but continue with other chunks
            print(f"[Chunking] Chunk {chunk_num} failed: {type(e).__name__}: {e}")
            failed_chunks.append(chunk_num)

    # Check that we got at least some summaries
    if not summaries:
        raise RuntimeError("All chunks failed during summarization.")

    if failed_chunks:
        print(f"[Chunking] Warning: chunks {failed_chunks} failed, "
              f"continuing with {len(summaries)} successful summaries")

    # ── Final combination step ──
    print(f"[Chunking] Combining {len(summaries)} summaries into final report")

    combined_summaries = "\n\n".join(summaries)
    final_prompt = COMBINE_PROMPT.format(summaries=combined_summaries)
    final_report = call_llm(final_prompt)

    return final_report


def parse_analysis_response(raw_response):
    """Parse the AI response to extract title, document type, suggested questions, and clean analysis."""
    doc_title = "Document Analysis"
    doc_type = "General"
    suggested_questions = []
    analysis = raw_response

    # Extract title (📄 line)
    for line in raw_response.split("\n"):
        stripped = line.strip()
        if stripped.startswith("📄"):
            doc_title = stripped.replace("📄", "").strip().strip("[]").strip()
            break

    # Extract document type (📂 line)
    for line in raw_response.split("\n"):
        stripped = line.strip()
        if stripped.startswith("📂"):
            doc_type = stripped.replace("📂", "").strip().strip("[]").strip()
            break

    # Extract suggested questions
    if "---SUGGESTED_QUESTIONS---" in raw_response:
        parts = raw_response.split("---SUGGESTED_QUESTIONS---")
        analysis = parts[0].strip()
        questions_block = parts[1].strip()
        for line in questions_block.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove the "1. " prefix
                q = line.split(".", 1)[-1].strip() if "." in line else line
                if q:
                    suggested_questions.append(q)

    # Remove the 📄 and 📂 lines from the analysis body
    clean_lines = []
    for line in analysis.split("\n"):
        stripped = line.strip()
        if stripped.startswith("📄") and doc_title in stripped:
            continue
        if stripped.startswith("📂") and doc_type in stripped:
            continue
        clean_lines.append(line)
    analysis = "\n".join(clean_lines).strip()

    return {
        "title": doc_title,
        "type": doc_type,
        "analysis": analysis,
        "suggested_questions": suggested_questions[:5]
    }


def answer_question(document_text, question):
    """Answer a question grounded in the document, with automatic chunking.

    Reuses split_text() and call_llm() — no duplicated logic.
    Small documents  → one Q&A call.
    Large documents  → ask each chunk → combine relevant answers.
    """
    chunks = split_text(document_text)

    # ── Small document: single call ──
    if len(chunks) == 1:
        prompt = QA_PROMPT.format(document=chunks[0], question=question)
        return call_llm(prompt)

    # ── Large document: check each chunk for relevant answers ──
    print(f"[Q&A] Searching {len(chunks)} chunks for answer")

    partial_answers = []

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        try:
            prompt = QA_CHUNK_PROMPT.format(document=chunk, question=question)
            answer = call_llm(prompt)
            partial_answers.append(f"--- Part {chunk_num} ---\n{answer}")
        except Exception as e:
            print(f"[Q&A] Chunk {chunk_num} failed: {type(e).__name__}: {e}")

    if not partial_answers:
        raise RuntimeError("All chunks failed during Q&A.")

    # ── Combine partial answers ──
    combined = "\n\n".join(partial_answers)
    final_prompt = QA_COMBINE_PROMPT.format(answers=combined, question=question)
    return call_llm(final_prompt)


# ──────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION — WITH OCR FALLBACK
#
# WHY OCR IS ONLY USED AS A FALLBACK:
#   PyMuPDF text extraction is instant and accurate for digital PDFs.
#   OCR is slow, requires a large model download, and can introduce
#   recognition errors. We only use it when PyMuPDF finds no text,
#   which indicates the PDF is likely a scanned image.
#
# WHY PyMuPDF RENDERS PAGES TO IMAGES (NO pdf2image NEEDED):
#   PyMuPDF can render any PDF page to a pixel buffer (Pixmap) using
#   its built-in MuPDF engine. This avoids installing Poppler, Ghostscript,
#   or any external system dependency that pdf2image would require.
#
# WHY THE EXISTING CHUNKING PIPELINE IS REUSED:
#   Whether text comes from PyMuPDF or OCR, the result is the same:
#   a plain string. split_text(), analyze_document(), and
#   answer_question() all work identically regardless of source.
# ──────────────────────────────────────────────────────────────────────

# Minimum characters for PyMuPDF output to be considered "real" text.
# Scanned PDFs often produce 0–50 junk chars from metadata or watermarks.
MIN_TEXT_THRESHOLD = 50

# Lazy-loaded OCR engine (only initialized when needed)
_ocr_engine = None


def get_ocr_engine():
    """Initialize PaddleOCR only when first needed (lazy loading).

    This avoids the ~2s startup cost and large model download
    for users who only work with digital PDFs.

    PaddleOCR 3.7 notes:
    - enable_mkldnn=False is required on CPU to prevent the
      ConvertPirAttribute2RuntimeAttribute crash. The PIR executor
      in PaddlePaddle 3.x is incompatible with oneDNN (MKLDNN)
      acceleration, so we disable it explicitly.
    - use_textline_orientation=True replaces the deprecated
      use_angle_cls parameter from PaddleOCR 2.x.
    """
    global _ocr_engine
    if _ocr_engine is None:
        print("[OCR] Initializing PaddleOCR engine (first time only)...")
        _ocr_engine = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            enable_mkldnn=False   # Prevents PIR/oneDNN crash on CPU
        )
        print("[OCR] PaddleOCR engine ready.")
    return _ocr_engine


def ocr_extract_from_pdf(pdf_bytes):
    """Extract text from a scanned PDF using PyMuPDF rendering + PaddleOCR.

    Steps:
    1. Open the PDF with PyMuPDF.
    2. Render each page to an image (numpy array) at 2x zoom for clarity.
    3. Run PaddleOCR predict() on each page image.
    4. Collect all recognized text into one string.

    PaddleOCR 3.7 uses predict() instead of the deprecated ocr() method.
    The predict() method returns a list of result objects. Each result
    has a 'rec_texts' attribute containing a list of recognized strings.
    """
    ocr = get_ocr_engine()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    all_text = ""

    total_pages = len(doc)
    print(f"[OCR] Processing {total_pages} page(s)...")

    for page_num, page in enumerate(doc):
        # Render the page at 2x resolution for better OCR accuracy
        zoom = 2.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to numpy array (RGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )

        # If the image has an alpha channel (RGBA), drop it
        if pix.n == 4:
            img_array = img_array[:, :, :3]

        # ── PaddleOCR 3.7 predict() API ──
        # predict() returns a list of result objects.
        # Each result object has:
        #   .rec_texts  → list of recognized text strings
        #   .rec_scores → list of confidence scores
        #   .dt_polys   → list of bounding box coordinates
        results = ocr.predict(img_array)

        # Extract text from every result object
        page_text = ""
        if results:
            for res in results:
                # rec_texts is a list of strings recognized on this page
                if hasattr(res, "rec_texts") and res.rec_texts:
                    for text_line in res.rec_texts:
                        page_text += text_line + "\n"

        all_text += page_text

        print(f"[OCR] Page {page_num + 1}/{total_pages}: "
              f"{len(page_text.strip())} chars extracted")

        # Free memory
        pix = None

    doc.close()
    return all_text.strip()


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF. Tries PyMuPDF first, falls back to OCR.

    Returns the extracted text string, or None if the PDF is unreadable.
    """
    try:
        # Read the file bytes once (needed for both PyMuPDF and OCR)
        pdf_bytes = pdf_file.read()

        # ── Step 1: Try PyMuPDF digital text extraction ──
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        all_text = ""
        for page in doc:
            all_text += page.get_text()

        doc.close()
        digital_text = all_text.strip()

        # ── Step 2: Decide if the text is sufficient ──
        if len(digital_text) >= MIN_TEXT_THRESHOLD:
            # Digital PDF — plenty of text found
            print(f"[PDF] Digital PDF detected. "
                  f"Using PyMuPDF extraction. ({len(digital_text)} chars)")
            return digital_text

        # ── Step 3: Text is empty or tiny — likely a scanned PDF ──
        print(f"[PDF] Scanned PDF detected. "
              f"PyMuPDF found only {len(digital_text)} chars.")
        print(f"[PDF] Running PaddleOCR...")

        try:
            ocr_text = ocr_extract_from_pdf(pdf_bytes)

            if ocr_text:
                print(f"[PDF] OCR finished. Characters extracted: {len(ocr_text)}")
                return ocr_text
            else:
                print(f"[PDF] OCR finished but no text was recognized.")
                return ""

        except Exception as e:
            print(f"[OCR Error] {type(e).__name__}: {e}")
            # If OCR fails but we had some digital text, return that
            if digital_text:
                print(f"[PDF] Falling back to partial PyMuPDF text.")
                return digital_text
            return ""

    except Exception as e:
        print(f"[PDF Extraction Error] {type(e).__name__}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────


def login_required(view_func):
    """Protect authenticated routes."""
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped_view


@app.route("/")
def root():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        valid_username = os.getenv("APP_USERNAME", "admin")
        valid_password = os.getenv("APP_PASSWORD", "admin123")

        if username == valid_username and password == valid_password:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))

        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/app")
@login_required
def dashboard():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    text = ""
    source = "text"

    # --- Check if a PDF file was uploaded ---
    if "pdf" in request.files:
        pdf_file = request.files["pdf"]

        # Check that a file was actually selected
        if pdf_file.filename != "":
            # Validate file extension
            if not pdf_file.filename.lower().endswith(".pdf"):
                return jsonify({
                    "success": False,
                    "error": "Only .pdf files are accepted."
                }), 400

            # Extract text from the PDF
            extracted = extract_text_from_pdf(pdf_file)

            if extracted is None:
                return jsonify({
                    "success": False,
                    "error": "Could not read this PDF. The file may be damaged."
                }), 400

            if not extracted:
                return jsonify({
                    "success": False,
                    "error": "This PDF has no extractable text. It may be a scanned image."
                }), 400

            text = extracted
            source = "pdf"

    # --- If no PDF, fall back to pasted text ---
    if not text:
        # Text could come as form field or JSON
        pasted = request.form.get("text", "").strip()

        if not pasted:
            return jsonify({
                "success": False,
                "error": "Please paste some text or upload a PDF."
            }), 400

        text = pasted
        source = "text"

    # --- Send to AI for analysis (with automatic chunking) ---
    try:
        raw_analysis = analyze_document(text)
        parsed = parse_analysis_response(raw_analysis)

        # Store document server-side (not in cookie — avoids 4KB limit)
        sid = get_session_id()
        session.permanent = True
        _document_store[sid] = text

        return jsonify({
            "success": True,
            "analysis": parsed["analysis"],
            "doc_title": parsed["title"],
            "doc_type": parsed["type"],
            "suggested_questions": parsed["suggested_questions"]
        })

    except Exception as e:
        # Log the error in the terminal for debugging (never expose to user)
        print(f"[Groq API Error] {type(e).__name__}: {e}")

        return jsonify({
            "success": False,
            "error": "AI analysis failed. Please try again in a moment."
        }), 500


@app.route("/api/ask", methods=["POST"])
@login_required
def ask():
    """Answer a follow-up question grounded in the previously analyzed document."""
    # Get the question from JSON body
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"success": False, "error": "No question provided."}), 400

    question = data["question"].strip()

    if not question:
        return jsonify({"success": False, "error": "Question cannot be empty."}), 400

    # Retrieve the stored document text from the session
    sid = get_session_id()
    document_text = _document_store.get(sid)

    if not document_text:
        return jsonify({
            "success": False,
            "error": "No document found. Please analyze a document first."
        }), 400

    try:
        answer = answer_question(document_text, question)

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        print(f"[Q&A Error] {type(e).__name__}: {e}")

        return jsonify({
            "success": False,
            "error": "Could not answer the question. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
