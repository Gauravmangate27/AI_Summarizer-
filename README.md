🤖 AI Research Paper Summarizer

A powerful web-based application that extracts, summarizes, and provides AI-powered insights for research papers from multiple sources. Built with Python Flask backend and vanilla JavaScript frontend, powered by Google Gemini AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 📚 Multiple Input Sources
- **arXiv Papers** - Automatically fetch and process papers from arXiv using paper ID or URL
- **PDF Upload** - Extract and summarize text from uploaded PDF documents
- **Direct Text** - Paste and analyze paper content directly
- **YouTube Videos** - Extract and summarize video transcripts (requires captions)

### 🧠 AI-Powered Analysis
- **Smart Summarization** - Choose between extractive (fast) or abstractive (AI-powered) summarization
- **Customizable Length** - Short, medium, or long summaries based on your needs
- **AI Insights** - Automatically generated key insights, research implications, recommendations, and critical questions
- **Interactive Chat** - Ask questions about the loaded paper and get AI-powered answers

### 🎨 Modern UI
- Clean, responsive design with dark mode support
- Tab-based interface for different input methods
- Real-time processing indicators
- Copy-to-clipboard functionality
- Beautiful card-based layout for insights

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (free tier available)

### Installation

1. **Clone the repository**
git clone https://github.com/yourusername/ai-paper-summarizer.git
cd ai-paper-summarizer

2. **Install dependencies**
pip install -r requirements.txt

3. **Set up environment variables**
# Linux/Mac
export GOOGLE_API_KEY='your-gemini-api-key-here'

# Windows CMD
set GOOGLE_API_KEY=your-gemini-api-key-here

# Windows PowerShell
$env:GOOGLE_API_KEY='your-gemini-api-key-here'

**Get your FREE API key:** https://makersuite.google.com/app/apikey

4. **Run the application**
python app.py

5. **Open in browser**
http://localhost:5000

## 📦 Project Structure

ai-paper-summarizer/
│
├── app.py                  # Flask backend server
├── index.html             # Frontend UI
├── requirements.txt       # Python dependencies
├── uploads/              # Temporary file storage (auto-created)
└── README.md             # This file

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **Google Gemini AI** - LLM for insights and chat
- **PyMuPDF** - PDF text extraction
- **arxiv** - arXiv API integration
- **youtube-transcript-api** - YouTube transcript extraction
- **sumy** - Extractive summarization
- **NLTK** - Natural language processing

### Frontend
- **Vanilla JavaScript** - No frameworks needed
- **Modern CSS** - Custom design system with dark mode
- **Responsive Design** - Works on all devices

## 📖 Usage Guide

### Summarizing an arXiv Paper

1. Go to the "arXiv URL" tab
2. Enter paper URL (`https://arxiv.org/abs/2103.00020`) or just the ID (`2103.00020`)
3. Select summary length and model type
4. Click "Fetch & Summarize"

### Processing a PDF

1. Go to the "PDF Upload" tab
2. Click to select a PDF file or drag and drop
3. Select summary settings
4. Click "Extract & Summarize"

### Analyzing YouTube Videos

1. Go to the "YouTube Video" tab
2. Enter video URL (must have captions/transcripts)
3. Choose summary preferences
4. Click "Extract & Summarize"

**Note:** Only works with videos that have captions enabled (educational videos, lectures, talks). Most music videos won't work.

### Chat with Your Paper

After generating a summary:
1. Scroll to "Chat with Your Paper" section
2. Ask questions about the content
3. Get AI-powered answers based on the paper context

## ⚙️ Configuration

### Summary Settings

- **Length Options:**
  - Short (3-4 sentences, ~100-130 words)
  - Medium (5-7 sentences, ~200-250 words)
  - Long (10-12 sentences, ~400-500 words)

- **Model Types:**
  - Extractive - Fast, rule-based summarization using LexRank
  - Abstractive - AI-powered summarization using Google Gemini

## 🔑 API Keys

### Google Gemini API (Required)

1. Visit https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy and set as environment variable

**Free Tier Limits:**
- 60 requests per minute
- 1,500 requests per day
- No credit card required

## 🐛 Troubleshooting

### Backend Connection Issues

**Error:** "Failed to connect to backend"

**Solution:** Make sure Flask server is running:
python app.py

### Gemini API Errors

**Error:** "Expecting value: line 1 column 1"

**Solutions:**
- Check API key is set correctly
- Verify you haven't exceeded rate limits (60/min)
- Wait a few moments and try again

### YouTube Transcript Errors

**Error:** "Video doesn't have captions"

**Solution:** Only use videos with captions enabled. Try:
- Educational videos (Khan Academy, Coursera)
- Tech talks and presentations
- Tutorial videos
- TED Talks

## 📝 Dependencies

flask==3.0.0
flask-cors==4.0.0
arxiv==2.1.0
PyMuPDF==1.23.8
sumy==0.11.0
nltk==3.8.1
werkzeug==3.0.1
youtube-transcript-api==0.6.2
google-generativeai==0.3.2

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for powerful LLM capabilities
- arXiv for open access to research papers
- All open-source libraries used in this project

## 📧 Contact



Project Link: [[https://github.com//ai-paper-summarizer](https://github.com/Gauravmangate27/AI_Summarizer)]

---

**Made with ❤️ for the research community**
