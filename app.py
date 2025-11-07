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
import google.generativeai as genai
import json
import whisper
import dotenv
loaded = dotenv.load_dotenv()
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

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set. Please set it as environment variable.")
    print("Get your free API key from: https://makersuite.google.com/app/apikey")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("✓ Google Gemini initialized successfully")

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
        if not GOOGLE_API_KEY:
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
        
        response = gemini_model.generate_content(prompt)
        summary = response.text.strip()
        
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
    """Generate insights using Google Gemini with improved error handling"""
    
    prompt = f"""
    Analyze the following research summary and provide comprehensive insights in JSON format.
    
    Title: {title}
    Summary: {summary}
    
    Please provide:
    1. Key Insights (3-4 main points about the core contributions and findings)
    2. Research Implications (3-4 points about impact, significance, and real-world applications)
    3. Recommended Reading (3-4 related papers, books, or resources for further study)
    4. Critical Questions (3-4 important questions this research raises or leaves open)
    
    Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
    {{
        "keyInsights": ["insight 1", "insight 2", "insight 3"],
        "implications": ["implication 1", "implication 2", "implication 3"],
        "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
        "questions": ["question 1", "question 2", "question 3"]
    }}
    """
    
    try:
        if not GOOGLE_API_KEY:
            print("⚠️ No API key configured, using fallback insights")
            return generate_fallback_insights(summary, title)
        
        # Generate content with Gemini
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        # Check if response is valid
        if not response or not response.text:
            print("⚠️ Empty response from Gemini, using fallback")
            return generate_fallback_insights(summary, title)
        
        insights_text = response.text.strip()
        
        # Log the raw response for debugging
        print(f"✓ Gemini response length: {len(insights_text)} chars")
        
        # Clean up the response - remove markdown code blocks if present
        insights_text = re.sub(r'```\s*', '', insights_text)
        insights_text = re.sub(r'```\s*$', '', insights_text)
        insights_text = insights_text.strip()
        
        # If response is too short, likely an error
        if len(insights_text) < 50:
            print(f"⚠️ Response too short: {insights_text}")
            return generate_fallback_insights(summary, title)
        
        # Try to parse JSON
        try:
            insights = json.loads(insights_text)
        except json.JSONDecodeError as je:
            print(f"⚠️ JSON decode error: {je}")
            print(f"Raw response: {insights_text[:200]}...")
            
            # Try to extract JSON from the response if it's wrapped in text
            json_match = re.search(r'\{[\s\S]*\}', insights_text)
            if json_match:
                try:
                    insights = json.loads(json_match.group(0))
                except:
                    return generate_fallback_insights(summary, title)
            else:
                return generate_fallback_insights(summary, title)
        
        # Validate structure
        required_keys = ['keyInsights', 'implications', 'recommendations', 'questions']
        for key in required_keys:
            if key not in insights:
                print(f"⚠️ Missing key in response: {key}")
                return generate_fallback_insights(summary, title)
            
            # Ensure each key has at least one item
            if not insights[key] or len(insights[key]) == 0:
                insights[key] = [f"Analysis pending for {key}"]
        
        print("✓ Successfully generated insights with Gemini")
        return insights
    
    except Exception as e:
        print(f"❌ Gemini insight generation error: {e}")
        print(f"Error type: {type(e).__name__}")
        return generate_fallback_insights(summary, title)

    
    except Exception as e:
        print(f"Gemini insight generation error: {e}")
        return generate_fallback_insights(summary, title)

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
        if not GOOGLE_API_KEY:
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
        
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        
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
