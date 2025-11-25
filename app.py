from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import arxiv
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
import requests
from bs4 import BeautifulSoup
import gdown
import json
import whisper
try:
    from dotenv import load_dotenv
    # Attempt to load .env; if parse errors occur we fallback to a manual parser.
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not installed. .env file won't be loaded. Install with: pip install python-dotenv")
# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Gemini API Configuration
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

gemini_client = None
gemini_model_name = None

def init_gemini(preferred=None, silent=False):
    """Initialize Gemini client using the new google.genai.Client API."""
    global gemini_client, gemini_model_name
    if not GOOGLE_API_KEY:
        if not silent:
            print("WARNING: GOOGLE_API_KEY not set. Get one at https://makersuite.google.com/app/apikey")
        gemini_client = None
        gemini_model_name = None
        return None
    
    attempt_list = []
    desired = (preferred or os.environ.get('GEMINI_MODEL', '').strip())
    if desired:
        attempt_list.append(desired)
    # Common public model names for the new API
    attempt_list.extend([
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-pro',
        'gemini-1.5-pro-latest'
    ])
    # Deduplicate preserving order
    seen = set(); ordered = []
    for name in attempt_list:
        if name and name not in seen:
            ordered.append(name); seen.add(name)
    
    last_error = None
    for name in ordered:
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            # Test the model with a simple request
            response = client.models.generate_content(
                model=name,
                contents="Test"
            )
            # If successful, set globals
            gemini_client = client
            gemini_model_name = name
            if not silent:
                print(f"✓ Gemini model initialized: {name}")
            return client
        except Exception as e:
            last_error = e
            if not silent:
                print(f"⚠️ Failed model '{name}': {e}")
            continue
    
    print(f"❌ All Gemini model attempts failed. Last error: {last_error}")
    gemini_client = None
    gemini_model_name = None
    return None

def fallback_parse_env(path='.env'):
    """Simple .env parser tolerant of malformed lines. Only KEY=VALUE pairs kept."""
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith('#'):
                    continue
                if ' ' in raw.split('=')[0]:  # invalid key segment with space
                    continue
                if raw.startswith('export '):
                    raw = raw[len('export '):]
                if '=' not in raw:
                    continue
                key, val = raw.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if not key:
                    continue
                # Skip lines that look like commands (e.g., setx KEY ...)
                if key.lower() == 'setx':
                    continue
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
                    os.environ.setdefault(key, val)
        print("✓ Fallback .env parsing completed")
    except Exception as e:
        print(f"⚠️ Fallback .env parser error: {e}")

# If API key still missing after primary load, attempt fallback parsing
if not os.environ.get('GOOGLE_API_KEY'):
    fallback_parse_env()
    # Attempt re-init if key appeared
    if os.environ.get('GOOGLE_API_KEY') and gemini_client is None:
        init_gemini(silent=True)

# Perform eager initialization
init_gemini()

@app.route('/api/models', methods=['GET'])
def list_models():
    """Return available Gemini models supporting generateContent (normalized names)."""
    if not GOOGLE_API_KEY:
        return jsonify({'error': 'API key not configured'}), 400
    try:
        models = []
        client = genai.Client(api_key=GOOGLE_API_KEY)
        # List available models
        available_models = [
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro',
            'gemini-1.5-pro-latest'
        ]
        for model_name in available_models:
            models.append({'name': model_name, 'raw': model_name, 'methods': ['generateContent']})
        return jsonify({'models': models, 'count': len(models), 'active': gemini_model_name})
    except Exception as e:
        return jsonify({'error': f'Failed to list models: {e}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Report Gemini configuration status for debugging."""
    return jsonify({
        'has_api_key': bool(GOOGLE_API_KEY),
        'api_key_prefix': GOOGLE_API_KEY[:6] + '...' if GOOGLE_API_KEY else None,
        'model_initialized': gemini_client is not None,
        'active_model': gemini_model_name,
        'env_working_dir': os.getcwd(),
        'notes': 'If has_api_key is false, ensure .env exists with GOOGLE_API_KEY.'
    })

# Store paper contexts for chat (in-memory, use database for production)
paper_contexts = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def fetch_generic_url_content(url):
    """Fetch and extract textual content from a generic URL (HTML or PDF)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code >= 400:
            raise Exception(f"Request failed with status {resp.status_code}")
        content_type = resp.headers.get('Content-Type', '').lower()
        # Handle PDF either by header or extension
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_url_fetch.pdf')
            with open(tmp_path, 'wb') as f:
                f.write(resp.content)
            try:
                text = extract_text_from_pdf(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            return text
        # Assume HTML/text
        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'aside', 'form', 'nav']):
            tag.decompose()
        # Prefer main/article if present
        main_candidate = soup.find(['main', 'article'])
        if main_candidate:
            text = main_candidate.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Truncate extremely long pages
        max_chars = 200000
        if len(text) > max_chars:
            text = text[:max_chars]
        if len(text) < 100:
            raise Exception("Extracted text too short; page might be mostly dynamic or blocked.")
        return text
    except Exception as e:
        raise Exception(f"Error fetching URL content: {e}")

def extract_drive_file_id(url):
    """Extract Google Drive file ID from various share link formats."""
    patterns = [
        r'/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'open\?id=([a-zA-Z0-9_-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_google_drive_document(drive_url):
    """Download and extract text from Google Drive shared document (PDF or Docs)."""
    try:
        file_id = extract_drive_file_id(drive_url)
        if not file_id:
            raise Exception("Could not extract file ID from Google Drive URL. Please use a valid Drive share link.")
        
        # Try direct PDF download first
        download_url = f'https://drive.google.com/uc?id={file_id}'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'drive_{file_id}.pdf')
        
        try:
            # Use gdown to handle authentication and download
            # Set fuzzy=False and use direct URL
            try:
                output = gdown.download(download_url, temp_path, quiet=False, fuzzy=False)
            except Exception as gdown_err:
                # Try alternate method with fuzzy matching
                output = gdown.download(f'https://drive.google.com/file/d/{file_id}/view', temp_path, quiet=False, fuzzy=True)
            
            if not output or not os.path.exists(temp_path):
                raise Exception("Download failed. Ensure the file is shared with 'Anyone with the link' and try again.")
            
            # Check if it's a PDF by reading file header
            with open(temp_path, 'rb') as f:
                header = f.read(5)
                if header.startswith(b'%PDF'):
                    # It's a PDF
                    text = extract_text_from_pdf(temp_path)
                else:
                    # Might be text or other format
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as tf:
                        text = tf.read()
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if not text or len(text) < 100:
                raise Exception("Extracted text too short or empty")
            
            return text
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception(f"Failed to download or process Drive file: {str(e)}")
            
    except Exception as e:
        raise Exception(f"Error fetching Google Drive document: {str(e)}")

def fetch_arxiv_paper(arxiv_id):
    """Fetch paper from arXiv using the arxiv API"""
    try:
        arxiv_id = arxiv_id.strip()
        if 'arxiv.org' in arxiv_id:
            match = re.search(r'(\d+\.\d+)', arxiv_id)
            if match:
                arxiv_id = match.group(1)
        
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        
        paper_info = {
            'title': paper.title,
            'authors': ', '.join([author.name for author in paper.authors]),
            'published': paper.published.strftime('%Y-%m-%d'),
            'categories': ', '.join(paper.categories),
            'abstract': paper.summary,
            'pdf_url': paper.pdf_url
        }
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{arxiv_id}.pdf")
        paper.download_pdf(filename=pdf_path)
        
        full_text = extract_text_from_pdf(pdf_path)
        
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        return paper_info, full_text
    
    except StopIteration:
        raise Exception("Paper not found. Please check the arXiv ID.")
    except Exception as e:
        raise Exception(f"Error fetching arXiv paper: {str(e)}")

def extract_youtube_id(url):
    """Extract YouTube video ID from URL - supports all formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard youtube.com/watch?v=
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'youtu\.be\/([0-9A-Za-z_-]{11})',   # Short youtu.be links
        r'^([0-9A-Za-z_-]{11})$'             # Direct video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise Exception("Invalid YouTube URL format")

import whisper

def  get_youtube_transcript(youtube_url):
    import tempfile
    import subprocess

    # Download audio using youtube-dl (or yt-dlp)
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': '%(id)s.%(ext)s',
        'noplaylist': True,
        'no_warnings': True,
        'postprocessors': [{  # extract audio
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    import yt_dlp

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        audio_file = ydl.prepare_filename(info).replace(info['ext'], 'mp3')

    # Transcribe using whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcription = result['text']

    # Cleanup downloaded audio file
    os.remove(audio_file)

    return transcription
# def get_youtube_transcript(video_id):
#     """Get YouTube video transcript"""      
#     try:
#         # First try to get transcript via YouTubeTranscriptApi
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
#         # Prefer English transcript if available
#         if transcript_list.find_transcript(['en']):
#             transcript = transcript_list.find_transcript(['en'])
#         else:
#             transcript = transcript_list.find_transcript(transcript_list._transcripts.keys())
        
#         fetched_transcript = transcript.fetch()
#         full_text = " ".join([entry['text'] for entry in fetched_transcript])
#         return full_text.strip()
        
# except Exception as e:
#         error_msg = str(e)
#         if "No transcripts" in error_msg or "Could not retrieve" in error_msg:
#             raise Exception(f"This video doesn't have captions/transcripts available. Error: {error_msg}")
#         else:
#             raise Exception(f"Error fetching YouTube transcript: {error_msg}")


def get_youtube_metadata(video_id):
    """Get YouTube video metadata"""
    return {
        'title': f"YouTube Video {video_id}",
        'authors': 'YouTube Creator',
        'published': 'N/A',
        'categories': 'Video Content'
    }

def summarize_with_gemini(text, summary_length='medium'):
    """Summarize text using Google Gemini"""
    try:
        if not GOOGLE_API_KEY or gemini_client is None:
            # Attempt lazy initialization if key exists but client missing
            if GOOGLE_API_KEY and gemini_client is None:
                init_gemini(silent=True)
        if not GOOGLE_API_KEY or gemini_client is None:
            return summarize_extractive(text, summary_length)
        
        length_instructions = {
            'short': 'Provide a brief summary in 3-4 sentences (around 100-130 words).',
            'medium': 'Provide a comprehensive summary in 5-7 sentences (around 200-250 words).',
            'long': 'Provide a detailed summary in 10-12 sentences (around 400-500 words).'
        }
        
        instruction = length_instructions.get(summary_length, length_instructions['medium'])
        
        prompt = f"""
        Summarize the following research paper or content. {instruction}
        Focus on the main contributions, methodology, key findings, and significance.
        
        Content:
        {text[:15000]}
        
        Provide only the summary, without any preamble or additional commentary.
        """
        
        response = gemini_client.models.generate_content(
            model=gemini_model_name,
            contents=prompt
        )
        summary = (response.text or "").strip()
        
        return summary
    
    except Exception as e:
        print(f"Gemini summarization error: {e}")
        return summarize_extractive(text, summary_length)

def summarize_extractive(text, summary_length='medium'):
    """Extractive summarization using LexRank (fallback)"""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        
        sentence_counts = {
            'short': 3,
            'medium': 5,
            'long': 8
        }
        sentence_count = sentence_counts.get(summary_length, 5)
        
        summary_sentences = summarizer(parser.document, sentence_count)
        summary = ' '.join([str(sentence) for sentence in summary_sentences])
        
        return summary
    except Exception as e:
        raise Exception(f"Extractive summarization error: {str(e)}")

def generate_insights_with_gemini(summary, title, full_text=None):
    """Generate insights using Google Gemini with improved error handling.
    Adds provider flag to distinguish gemini vs fallback responses."""

    # Build extended context excerpt
    context_excerpt = ""
    if isinstance(full_text, str) and full_text:
        max_chars = 18000
        context_excerpt = full_text[:max_chars]

    prompt = f"""
    You are an expert research assistant. Using the provided paper summary and extended context, extract structured insights.

    Title: {title}
    Summary: {summary}
    ExtendedContext: {context_excerpt}

    Tasks:
    1. keyInsights: 3-5 concise bullet points capturing core contributions, methods, or results.
    2. implications: 3-5 points on impact, applications, or significance.
    3. recommendations: 3-5 specific, actionable follow-up readings (generic placeholders only if insufficient info).
    4. questions: 3-5 probing, open research questions emerging from this work.

    Rules:
    - Output ONLY raw JSON (no backticks, no markdown, no commentary before/after).
    - Keep each list item under 200 characters.
    - If information is missing, reason conservatively and avoid fabrication; prefix uncertain items with "Potential:".
    - Do not include the word 'insight' inside list items.

    JSON Example:
    {{
      "keyInsights": ["..."],
      "implications": ["..."],
      "recommendations": ["..."],
      "questions": ["..."]
    }}
    """

    try:
        if not GOOGLE_API_KEY or gemini_client is None:
            if GOOGLE_API_KEY and gemini_client is None:
                init_gemini(silent=True)
            print("⚠️ No API key configured, using fallback insights")
            fb = generate_fallback_insights(summary, title)
            fb['provider'] = 'fallback'
            return fb

        response = gemini_client.models.generate_content(
            model=gemini_model_name,
            contents=prompt,
            config={
                'temperature': 0.7,
                'max_output_tokens': 2048
            }
        )

        if not response or not getattr(response, 'text', None):
            print("⚠️ Empty response from Gemini, using fallback")
            fb = generate_fallback_insights(summary, title)
            fb['provider'] = 'fallback'
            return fb

        insights_text = (response.text or "").strip()
        print(f"✓ Gemini raw insights length: {len(insights_text)} chars")

        insights_text = re.sub(r'```\s*', '', insights_text)
        insights_text = re.sub(r'```\s*$', '', insights_text).strip()

        if len(insights_text) < 50:
            print(f"⚠️ Response too short: {insights_text}")
            fb = generate_fallback_insights(summary, title)
            fb['provider'] = 'fallback'
            return fb

        try:
            insights = json.loads(insights_text)
        except json.JSONDecodeError as je:
            print(f"⚠️ JSON decode error: {je}")
            json_match = re.search(r'\{[\s\S]*\}', insights_text)
            if json_match:
                try:
                    insights = json.loads(json_match.group(0))
                except Exception:
                    fb = generate_fallback_insights(summary, title)
                    fb['provider'] = 'fallback'
                    return fb
            else:
                fb = generate_fallback_insights(summary, title)
                fb['provider'] = 'fallback'
                return fb

        required_keys = ['keyInsights', 'implications', 'recommendations', 'questions']
        for key in required_keys:
            if key not in insights:
                print(f"⚠️ Missing key in response: {key}")
                fb = generate_fallback_insights(summary, title)
                fb['provider'] = 'fallback'
                return fb
            if not insights[key]:
                insights[key] = [f"Analysis pending for {key}"]

        insights['provider'] = 'gemini'
        print("✓ Successfully generated insights with Gemini")
        return insights

    except Exception as e:
        print(f"❌ Gemini insight generation error: {e}")
        fb = generate_fallback_insights(summary, title)
        fb['provider'] = 'fallback'
        return fb

def generate_fallback_insights(summary, title):
    """Generate basic insights without API"""
    return {
        'keyInsights': [
            "This work presents important contributions to its research field",
            "The methodology demonstrates a rigorous and systematic approach",
            "Results show significant findings that warrant further investigation"
        ],
        'implications': [
            "This research could influence future work in related areas",
            "The findings have potential practical applications in the domain",
            "Further validation and extension of these results is recommended"
        ],
        'recommendations': [
            "Review foundational papers cited in the references section",
            "Explore related work in similar research domains and methodologies",
            "Consider recent surveys and review papers in this field for broader context"
        ],
        'questions': [
            "How do these findings generalize to different contexts or datasets?",
            "What are the limitations and boundary conditions of the proposed approach?",
            "What future research directions does this work suggest or enable?"
        ]
    }

def answer_question_with_gemini(question, context, history):
    """Answer user questions about the paper using Gemini"""
    try:
        if not GOOGLE_API_KEY or gemini_client is None:
            if GOOGLE_API_KEY and gemini_client is None:
                init_gemini(silent=True)
            return "API key not configured. Please set GOOGLE_API_KEY environment variable."
        
        # Build conversation history
        conversation = ""
        for msg in history[-6:]:  # Last 3 exchanges
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation += f"{role}: {msg['content']}\n"
        
        prompt = f"""
        You are an AI assistant helping users understand a research paper or content.
        
        Paper Title: {context.get('title', 'Unknown')}
        
        Summary:
        {context.get('summary', '')}
        
        Previous conversation:
        {conversation}
        
        Current question: {question}
        
        Provide a clear, concise answer based on the paper summary and context. If the question cannot be answered from the available information, say so honestly. Keep your answer focused and informative.
        """
        
        response = gemini_client.models.generate_content(
            model=gemini_model_name,
            contents=prompt
        )
        answer = (response.text or "").strip()
        
        return answer
    
    except Exception as e:
        print(f"Chat error: {e}")
        return "I'm sorry, I encountered an error processing your question. Please try again."

def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    return text.strip()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/summarize/arxiv', methods=['POST'])
def summarize_arxiv():
    try:
        data = request.json
        arxiv_id = data.get('arxiv_id')
        summary_length = data.get('summary_length', 'medium')
        model_type = data.get('model_type', 'abstractive')
        
        if not arxiv_id:
            return jsonify({'error': 'arXiv ID is required'}), 400
        
        paper_info, full_text = fetch_arxiv_paper(arxiv_id)
        clean_full_text = clean_text(full_text)
        
        if model_type == 'abstractive':
            summary = summarize_with_gemini(clean_full_text, summary_length)
        else:
            summary = summarize_extractive(clean_full_text, summary_length)
        
        insights = generate_insights_with_gemini(summary, paper_info['title'], clean_full_text)
        
        # Store context for chat
        session_id = arxiv_id
        paper_contexts[session_id] = {
            'title': paper_info['title'],
            'summary': summary,
            'full_text': clean_full_text[:10000]  # Store first 10k chars
        }
        
        return jsonify({
            'paper_info': paper_info,
            'summary': summary,
            'insights': insights,
            'session_id': session_id,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize/pdf', methods=['POST'])
def summarize_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400
        
        file = request.files['pdf']
        summary_length = request.form.get('summary_length', 'medium')
        model_type = request.form.get('model_type', 'abstractive')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        full_text = extract_text_from_pdf(filepath)
        os.remove(filepath)
        
        if not full_text or len(full_text) < 100:
            return jsonify({'error': 'Could not extract sufficient text from PDF'}), 400
        
        clean_full_text = clean_text(full_text)
        
        if model_type == 'abstractive':
            summary = summarize_with_gemini(clean_full_text, summary_length)
        else:
            summary = summarize_extractive(clean_full_text, summary_length)
        
        paper_info = {
            'title': filename.replace('.pdf', ''),
            'authors': 'Extracted from PDF',
            'published': 'N/A'
        }
        
        insights = generate_insights_with_gemini(summary, paper_info['title'], clean_full_text)
        
        # Store context for chat
        session_id = filename
        paper_contexts[session_id] = {
            'title': paper_info['title'],
            'summary': summary,
            'full_text': clean_full_text[:10000]
        }
        
        return jsonify({
            'paper_info': paper_info,
            'summary': summary,
            'insights': insights,
            'session_id': session_id,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize/text', methods=['POST'])
def summarize_text_endpoint():
    try:
        data = request.json
        text = data.get('text')
        summary_length = data.get('summary_length', 'medium')
        model_type = data.get('model_type', 'abstractive')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(text) < 100:
            return jsonify({'error': 'Text is too short (minimum 100 characters)'}), 400
        
        clean_input_text = clean_text(text)
        
        if model_type == 'abstractive':
            summary = summarize_with_gemini(clean_input_text, summary_length)
        else:
            summary = summarize_extractive(clean_input_text, summary_length)
        
        paper_info = {
            'title': 'Direct Text Input',
            'authors': 'User Provided'
        }
        
        insights = generate_insights_with_gemini(summary, paper_info['title'], clean_input_text)
        
        # Store context for chat
        session_id = 'text_' + str(hash(text[:100]))
        paper_contexts[session_id] = {
            'title': paper_info['title'],
            'summary': summary,
            'full_text': clean_input_text[:10000]
        }
        
        return jsonify({
            'paper_info': paper_info,
            'summary': summary,
            'insights': insights,
            'session_id': session_id,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize/youtube', methods=['POST'])
def summarize_youtube():
    try:
        data = request.json
        youtube_url = data.get('youtube_url')
        summary_length = data.get('summary_length', 'medium')
        model_type = data.get('model_type', 'abstractive')
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        video_id = extract_youtube_id(youtube_url)
        transcript = get_youtube_transcript(video_id)
        
        if not transcript or len(transcript) < 100:
            return jsonify({'error': 'Could not extract transcript. Video may not have captions.'}), 400
        
        metadata = get_youtube_metadata(video_id)
        clean_transcript = clean_text(transcript)
        
        if model_type == 'abstractive':
            summary = summarize_with_gemini(clean_transcript, summary_length)
        else:
            summary = summarize_extractive(clean_transcript, summary_length)
        
        insights = generate_insights_with_gemini(summary, metadata['title'], clean_transcript)
        
        # Store context for chat
        session_id = video_id
        paper_contexts[session_id] = {
            'title': metadata['title'],
            'summary': summary,
            'full_text': clean_transcript[:10000]
        }
        
        return jsonify({
            'paper_info': metadata,
            'summary': summary,
            'insights': insights,
            'session_id': session_id,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize/url', methods=['POST'])
def summarize_generic_url():
    """Summarize arbitrary URL (HTML or PDF)."""
    try:
        data = request.json
        url = data.get('url')
        summary_length = data.get('summary_length', 'medium')
        model_type = data.get('model_type', 'abstractive')
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        if not re.match(r'^https?://', url):
            return jsonify({'error': 'URL must start with http:// or https://'}), 400
        raw_text = fetch_generic_url_content(url)
        clean_full_text = clean_text(raw_text)
        if model_type == 'abstractive':
            summary = summarize_with_gemini(clean_full_text, summary_length)
        else:
            summary = summarize_extractive(clean_full_text, summary_length)
        title_guess = url.split('/')[-1] or 'Web Document'
        paper_info = {
            'title': title_guess[:120],
            'authors': 'Web Source',
            'published': 'N/A',
            'source_url': url
        }
        insights = generate_insights_with_gemini(summary, paper_info['title'], clean_full_text)
        session_id = 'url_' + str(hash(url))
        paper_contexts[session_id] = {
            'title': paper_info['title'],
            'summary': summary,
            'full_text': clean_full_text[:10000]
        }
        return jsonify({
            'paper_info': paper_info,
            'summary': summary,
            'insights': insights,
            'session_id': session_id,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize/drive', methods=['POST'])
def summarize_google_drive():
    """Summarize Google Drive shared document (PDF or Docs)."""
    try:
        data = request.json
        drive_url = data.get('drive_url')
        summary_length = data.get('summary_length', 'medium')
        model_type = data.get('model_type', 'abstractive')
        
        if not drive_url:
            return jsonify({'error': 'Google Drive URL is required'}), 400
        
        if 'drive.google.com' not in drive_url:
            return jsonify({'error': 'Invalid Google Drive URL'}), 400
        
        raw_text = fetch_google_drive_document(drive_url)
        clean_full_text = clean_text(raw_text)
        
        if model_type == 'abstractive':
            summary = summarize_with_gemini(clean_full_text, summary_length)
        else:
            summary = summarize_extractive(clean_full_text, summary_length)
        
        file_id = extract_drive_file_id(drive_url)
        paper_info = {
            'title': f'Google Drive Document ({file_id[:8]}...)',
            'authors': 'Google Drive',
            'published': 'N/A',
            'source_url': drive_url
        }
        
        insights = generate_insights_with_gemini(summary, paper_info['title'], clean_full_text)
        
        session_id = 'drive_' + file_id
        paper_contexts[session_id] = {
            'title': paper_info['title'],
            'summary': summary,
            'full_text': clean_full_text[:10000]
        }
        
        return jsonify({
            'paper_info': paper_info,
            'summary': summary,
            'insights': insights,
            'session_id': session_id,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-insights', methods=['POST'])
def generate_insights_endpoint():
    try:
        data = request.json
        summary = data.get('summary')
        title = data.get('title', 'Content')
        
        if not summary:
            return jsonify({'error': 'Summary is required'}), 400
        
        insights = generate_insights_with_gemini(summary, title)
        
        return jsonify({
            'insights': insights,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat questions about the paper"""
    try:
        data = request.json
        question = data.get('question')
        context = data.get('context', {})
        history = data.get('history', [])
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if not context:
            return jsonify({'error': 'No paper context available'}), 400
        
        answer = answer_question_with_gemini(question, context, history)
        
        return jsonify({
            'answer': answer,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AI Research Paper Summarizer - Backend Server")
    print("="*60)
    if GOOGLE_API_KEY:
        print("✓ Google Gemini API: Connected")
    else:
        print("⚠ Google Gemini API: NOT CONFIGURED")
        print("  Get your free API key: https://makersuite.google.com/app/apikey")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
