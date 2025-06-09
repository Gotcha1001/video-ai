import os
import uuid
import base64
import logging
import time
import pickle
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import psutil

try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError as e:
    print(f"SocketIO import error: {e}")
    SOCKETIO_AVAILABLE = False

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Session storage file
SESSION_FILE = '/tmp/sessions.pkl'

# Load or initialize sessions
def load_sessions():
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load sessions: {e}")
    return {}

def save_sessions(sessions):
    try:
        with open(SESSION_FILE, 'wb') as f:
            pickle.dump(sessions, f)
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")

sessions = load_sessions()

# Initialize SocketIO
if SOCKETIO_AVAILABLE:
    try:
        socketio = SocketIO(app, cors_allowed_origins="*", cors_credentials=True, async_mode='threading')
        logger.info("SocketIO initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SocketIO: {e}")
        socketio = None
else:
    socketio = None

# Initialize Gemini model
if LANGCHAIN_AVAILABLE:
    try:
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY not found")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)
        logger.info("Gemini model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        llm = None
else:
    llm = None

def process_frame(frame_data, scale_factor=0.3):
    try:
        if ',' in frame_data:
            img_data = base64.b64decode(frame_data.split(",")[1])
        else:
            img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        if scale_factor != 1.0:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=50)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.debug(f"Frame processed, size: {len(img_str)} bytes")
        return img_str
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return None
    finally:
        if 'img' in locals():
            img.close()

@app.route('/')
def index():
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    if session_id not in sessions:
        sessions[session_id] = {'mode': None, 'responses': [], 'frame': None}
        save_sessions(sessions)
    logger.info(f"Session {session_id} initialized")
    return render_template('index.html')

@app.route('/chat/<mode>')
def chat(mode):
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        logger.error(f"Session not found for session_id: {session_id}")
        return redirect(url_for('index'))
    return render_template('chat.html', mode=mode, responses=sessions[session_id]['responses'])

@app.route('/start_stream', methods=['POST'])
def start_stream():
    session_id = session.get('session_id')
    logger.info(f"Starting stream for session_id: {session_id}")
    if not session_id or session_id not in sessions:
        logger.error(f"Session not found for session_id: {session_id}")
        return jsonify({'error': 'Session not found'}), 400

    data = request.get_json()
    mode = data.get('mode')
    if mode not in ['desktop', 'camera']:
        logger.error(f"Invalid mode: {mode}")
        return jsonify({'error': 'Invalid mode'}), 400

    sessions[session_id]['responses'] = []
    sessions[session_id]['mode'] = mode
    sessions[session_id]['frame'] = None
    save_sessions(sessions)
    logger.info(f"Stream started for mode: {mode}, session_id: {session_id}")
    return jsonify({'status': 'Stream started', 'redirect': url_for('chat', mode=mode)})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        logger.error(f"Session not found for session_id: {session_id}")
        return redirect(url_for('index')), 200

    data = request.get_json()
    mode = data.get('mode')
    logger.info(f"Stop stream requested for mode: {mode}, session_id: {session_id}")

    if sessions[session_id].get('mode') == mode:
        sessions[session_id]['mode'] = None
        sessions[session_id]['responses'] = []
        sessions[session_id]['frame'] = None
        save_sessions(sessions)
        logger.info(f"Session {session_id} reset")
    return redirect(url_for('index')), 200

if socketio:
    @socketio.on('screen_frame')
    def handle_screen_frame(frame_data):
        session_id = session.get('session_id')
        logger.info(f"Received screen frame for session_id: {session_id}")
        if session_id and session_id in sessions and sessions[session_id]['mode'] == 'desktop':
            processed_frame = process_frame(frame_data)
            if processed_frame:
                sessions[session_id]['frame'] = processed_frame
                save_sessions(sessions)
                logger.info(f"Desktop frame updated, size: {len(processed_frame)} bytes")
                log_memory_usage()

    @socketio.on('camera_frame')
    def handle_camera_frame(frame_data):
        session_id = session.get('session_id')
        logger.info(f"Received camera frame for session_id: {session_id}")
        if session_id and session_id in sessions and sessions[session_id]['mode'] == 'camera':
            processed_frame = process_frame(frame_data)
            if processed_frame:
                sessions[session_id]['frame'] = processed_frame
                save_sessions(sessions)
                logger.info(f"Camera frame updated, size: {len(processed_frame)} bytes")
                log_memory_usage()

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f}MB, VMS={mem_info.vms / 1024**2:.2f}MB")

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Received /process_audio request")
    session_id = session.get('session_id')
    logger.info(f"Session ID: {session_id}")
    if not session_id or session_id not in sessions:
        logger.error(f"Session not found for session_id: {session_id}")
        return jsonify({'error': 'Session not found'}), 400

    data = request.get_json()
    prompt = data.get('prompt')
    mode = data.get('mode')

    if not prompt:
        logger.warning("No prompt provided")
        return jsonify({'error': 'No prompt provided'}), 400
    if sessions[session_id]['mode'] != mode:
        logger.error(f"Stream not initialized for mode: {mode}")
        return jsonify({'error': 'Stream not initialized'}), 400

    try:
        # For testing: Skip frame requirement
        logger.debug(f"Processing prompt without frame: {prompt}")
        system_prompt = SystemMessage(content="You are a helpful AI assistant responding to user prompts.")
        user_prompt = HumanMessage(content=[{"type": "text", "text": prompt}])
        logger.debug(f"Invoking LLM with prompt: {prompt}")
        response = llm.invoke([system_prompt, user_prompt])
        response_text = response.content if hasattr(response, 'content') else str(response)
        sessions[session_id]['responses'].append({'prompt': prompt, 'response': response_text})
        save_sessions(sessions)
        logger.info(f"LLM response generated: {response_text}")
        log_memory_usage()
        return jsonify({'response': response_text})
    except Exception as e:
        logger.error(f"LLM error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Failed to process prompt: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'socketio_available': SOCKETIO_AVAILABLE,
        'langchain_available': LANGCHAIN_AVAILABLE,
        'google_api_key_set': bool(os.getenv("GOOGLE_API_KEY"))
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting application on port {port}")
    if not SOCKETIO_AVAILABLE:
        print("Warning: SocketIO not available")
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('RENDER'):
            socketio.run(app, host='0.0.0.0', port=port, debug=False)
        else:
            socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)