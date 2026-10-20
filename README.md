# AI Interview Platform

A mock interview web app that reads a candidate's resume (PDF) and generates interview questions using a locally run AI model. The interview runs live between the browser and the backend over WebSockets.

## Features
- Upload a resume PDF and get interview questions based on its content
- Live interview session using WebSockets
- Questions generated locally with Gemma through Ollama (no external API needed)
- User registration and login with JWT authentication
- User data stored in SQLite

**EDIT:** if your speech, face-tracking or analytics features work, add them here. If they are unfinished, move them to "Future work" below.

## Tech Stack
| Part | Technology |
|---|---|
| Backend | Python, FastAPI |
| Real-time | WebSockets |
| AI model | Gemma, run locally with Ollama |
| Database | SQLite |
| Authentication | JWT |
| Frontend | HTML, CSS, JavaScript |

## Project Structure
```
AI PROJECT/
├── main.py              # FastAPI app, routes and WebSocket endpoint
├── auth_service.py      # Registration, login, JWT
├── database.py          # SQLite setup
├── services/
│   ├── llm_service.py       # Calls Gemma through Ollama
│   ├── resume_service.py    # Reads text from the resume PDF
│   ├── interview_service.py # Interview flow
│   ├── interview_state.py   # Tracks the current session
│   ├── analytics_service.py
│   ├── speech_service.py
│   ├── vad_service.py
│   └── vision_service.py
├── frontend/
│   └── index.html       # Browser interface
├── assets/
│   └── face_landmarker.task
├── requirements.txt
└── README.md
```

## How It Works
1. The user registers or logs in and receives a JWT.
2. The user uploads a resume PDF.
3. The backend extracts the resume text.
4. The text is sent to Gemma (through Ollama), which generates questions.
5. The browser and backend exchange messages over a WebSocket during the live interview.

## Setup

### Requirements
- Python 3.10 or later
- [Ollama](https://ollama.com) installed

### Steps
1. Clone the repository
```
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
```
2. Create and activate a virtual environment
```
   python -m venv .venv
   .venv\Scripts\activate
```
3. Install dependencies
```
   pip install -r requirements.txt
```
4. Download the model (**EDIT:** use the exact model name you use)
```
   ollama pull gemma
```
5. Set a secret key for JWT (**EDIT:** match the variable name in your code, or remove this step)
```
   set SECRET_KEY=your_secret_here
```
6. Start the server
```
   uvicorn main:app --reload
```
7. Open `http://localhost:8000` in your browser.

## Usage
1. Register and log in.
2. Upload your resume as a PDF.
3. Start the interview and answer the questions.

## Future Work
- Speech-to-text for spoken answers
- Facial landmark analysis
- Answer scoring and feedback reports
- Support for more models

## Author
Thandeeswaran M
ECE student, Rajalakshmi Engineering College
