# import os
# import cv2
# import uuid
# import base64
# import logging
# from PIL import Image
# from io import BytesIO
# from dotenv import load_dotenv
# from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session
# from flask_socketio import SocketIO
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
#
# app = Flask(__name__)
# app.secret_key = os.urandom(24)
# socketio = SocketIO(app)
#
# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# # Load environment variables
# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#
# # Initialize the Gemini model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)
#
# # Global variables
# sessions = {}
#
# def capture_screen(scale_factor=0.5):
#     """Capture the screen, resize it, and return it as a base64-encoded string."""
#     try:
#         import pyautogui
#         screenshot = pyautogui.screenshot()
#         img = screenshot  # screenshot is already a PIL.Image
#         if scale_factor != 1.0:
#             new_width = int(img.width * scale_factor)
#             new_height = int(img.height * scale_factor)
#             img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
#         buffered = BytesIO()
#         img.save(buffered, format="JPEG", quality=70)
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         logger.debug(f"Screen captured successfully, size: {len(img_str)} bytes")
#         return img_str
#     except Exception as e:
#         logger.error(f"Screen capture error: {str(e)}")
#         return None
#
# def process_frame(frame_data, scale_factor=0.5):
#     """Process a base64-encoded frame, resize it, and return it as a base64-encoded string."""
#     try:
#         img_data = base64.b64decode(frame_data.split(",")[1])
#         img = Image.open(BytesIO(img_data))
#         if scale_factor != 1.0:
#             new_width = int(img.width * scale_factor)
#             new_height = int(img.height * scale_factor)
#             img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
#         buffered = BytesIO()
#         img.save(buffered, format="JPEG", quality=70)
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         logger.debug(f"Camera frame processed successfully, size: {len(img_str)} bytes")
#         return img_str
#     except Exception as e:
#         logger.error(f"Frame processing error: {str(e)}")
#         return None
#
# @app.route('/')
# def index():
#     """Render the index page."""
#     session_id = session.get('session_id', str(uuid.uuid4()))
#     session['session_id'] = session_id
#     if session_id not in sessions:
#         sessions[session_id] = {'mode': None, 'responses': [], 'frame': None}
#         logger.info(f"Session {session_id} reset")
#     return render_template('index.html')
#
# @app.route('/chat/<mode>')
# def chat(mode):
#     """Render the chat page for the specified mode."""
#     session_id = session.get('session_id')
#     if not session_id or session_id not in sessions:
#         return redirect(url_for('index'))
#     return render_template('chat.html', mode=mode, responses=sessions[session_id]['responses'])
#
# @app.route('/start_stream', methods=['POST'])
# def start_stream():
#     """Start the video stream for the specified mode."""
#     session_id = session.get('session_id')
#     if not session_id or session_id not in sessions:
#         return jsonify({'error': 'Session not found'}), 400
#
#     data = request.get_json()
#     mode = data.get('mode')
#     if mode not in ['desktop', 'camera']:
#         return jsonify({'error': 'Invalid mode'}), 400
#
#     # Reset responses to clear LLM history
#     sessions[session_id]['responses'] = []
#     sessions[session_id]['mode'] = mode
#     if mode == 'desktop':
#         sessions[session_id]['frame'] = capture_screen()
#         logger.info("Desktop stream started")
#     else:
#         logger.info("Camera stream started (client-side)")
#     logger.info(f"Session {session_id} responses reset for mode: {mode}")
#     return jsonify({'status': 'Stream started', 'redirect': url_for('chat', mode=mode)})
#
# @app.route('/stop_stream', methods=['POST'])
# def stop_stream():
#     """Stop the video stream for the specified mode."""
#     session_id = session.get('session_id')
#     if not session_id or session_id not in sessions:
#         return jsonify({'error': 'Session not found'}), 400
#
#     data = request.get_json()
#     mode = data.get('mode')
#     logger.info(f"Stop stream requested for mode: {mode}")
#
#     if sessions[session_id]['mode'] == mode:
#         if mode == 'desktop':
#             sessions[session_id]['frame'] = None
#             logger.info("Desktop stream stopped")
#         else:
#             sessions[session_id]['frame'] = None
#             logger.info("Camera stream stopped")
#         sessions[session_id]['mode'] = None
#         sessions[session_id]['responses'] = []
#         logger.info(f"Session {session_id} reset")
#         return jsonify({'redirect': url_for('index')})
#     else:
#         logger.warning(f"Invalid stop stream request: mode {mode} does not match session mode {sessions[session_id]['mode']}")
#         return jsonify({'error': 'Invalid mode'}), 400
#
# @app.route('/video_feed/<mode>')
# def video_feed(mode):
#     """Stream video feed for the specified mode."""
#     session_id = session.get('session_id')
#     if not session_id or session_id not in sessions or sessions[session_id]['mode'] != mode:
#         return Response("Stream not initialized", status=400)
#
#     def generate():
#         while sessions[session_id]['mode'] == mode:
#             if mode == 'desktop':
#                 frame = capture_screen()
#                 if frame:
#                     sessions[session_id]['frame'] = frame
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' +
#                            base64.b64decode(frame) + b'\r\n')
#             else:
#                 frame = sessions[session_id]['frame']
#                 if frame:
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' +
#                            base64.b64decode(frame) + b'\r\n')
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# @socketio.on('camera_frame')
# def handle_camera_frame(frame_data):
#     """Handle camera frame updates from the client."""
#     session_id = session.get('session_id')
#     if session_id and session_id in sessions and sessions[session_id]['mode'] == 'camera':
#         processed_frame = process_frame(frame_data)
#         if processed_frame:
#             sessions[session_id]['frame'] = processed_frame
#             logger.debug("Camera frame updated from client")
#
# @app.route('/process_audio', methods=['POST'])
# def process_audio():
#     """Process audio input and generate a response."""
#     session_id = session.get('session_id')
#     if not session_id or session_id not in sessions:
#         return jsonify({'error': 'Session not found'}), 400
#
#     data = request.get_json()
#     prompt = data.get('prompt')
#     mode = data.get('mode')
#
#     if not prompt:
#         return jsonify({'error': 'No prompt provided'}), 400
#     if sessions[session_id]['mode'] != mode:
#         return jsonify({'error': 'Stream not initialized'}), 400
#
#     try:
#         frame = sessions[session_id]['frame']
#         if not frame:
#             logger.error(f"No frame available for mode: {mode}")
#             return jsonify({'error': 'No frame available'}), 500
#
#         system_prompt = SystemMessage(content="You are a helpful AI assistant analyzing images and responding to user prompts.")
#         user_prompt = HumanMessage(content=[
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": f"data:image/jpeg;base64,{frame}"}
#         ])
#         logger.debug(f"Invoking LLM with prompt: {prompt}, frame size: {len(frame)} bytes")
#         response = llm.invoke([system_prompt, user_prompt])
#         response_text = response.content if hasattr(response, 'content') else str(response)
#         sessions[session_id]['responses'].append({'prompt': prompt, 'response': response_text})
#         logger.info(f"LLM response: {response_text}")
#         return jsonify({'response': response_text})
#     except Exception as e:
#         logger.error(f"Assistant error in process_audio: {str(e)}", exc_info=True)
#         return jsonify({'error': f"Failed to process audio: {str(e)}"}), 500
#
# if __name__ == '__main__':
#     import os
#     if os.environ.get('FLASK_ENV') == 'production':
#         pass
#     else:
#         socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)



import os
import cv2
import uuid
import base64
import logging
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session
from flask_socketio import SocketIO
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, max_http_buffer_size=256 * 1024, ping_timeout=20, ping_interval=10)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)

sessions = {}

def capture_screen(scale_factor=0.1):
    try:
        logger.info(f"DISPLAY={os.getenv('DISPLAY')}, XAUTHORITY={os.getenv('XAUTHORITY')}")
        import pyautogui
        pyautogui.FAILSAFE = False
        screenshot = pyautogui.screenshot()
        img = screenshot  # screenshot is already a PIL.Image.Image
        if scale_factor != 1.0:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=20)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.debug(f"Screen captured, size: {len(img_str)} bytes")
        return img_str
    except Exception as e:
        logger.error(f"Screen capture error: {str(e)}")
        return None

def process_frame(frame_data, scale_factor=0.1):
    try:
        img_data = base64.b64decode(frame_data.split(",")[1])
        img = Image.open(BytesIO(img_data))
        if scale_factor != 1.0:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=15)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.debug(f"Camera frame processed, size: {len(img_str)} bytes")
        return img_str
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return None

@app.route('/')
def index():
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    if session_id not in sessions:
        sessions[session_id] = {'mode': None, 'responses': [], 'frame': None}
        logger.info(f"Session {session_id} initialized")
    return render_template('index.html')

@app.route('/chat/<mode>')
def chat(mode):
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        logger.warning(f"Session {session_id} not found, redirecting")
        return redirect(url_for('index'))
    return render_template('chat.html', mode=mode, responses=sessions[session_id]['responses'])

@app.route('/start_stream', methods=['POST'])
def start_stream():
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        sessions[session_id] = {'mode': None, 'responses': [], 'frame': None}
        logger.info(f"New session {session_id} created")

    data = request.get_json()
    mode = data.get('mode')
    if mode not in ['desktop', 'camera']:
        return jsonify({'error': 'Invalid mode'}), 400

    sessions[session_id]['responses'] = []
    sessions[session_id]['mode'] = mode
    if mode == 'desktop':
        frame = capture_screen()
        if not frame:
            logger.error("Failed to capture desktop frame")
            return jsonify({'error': 'Failed to initialize desktop stream'}), 500
        sessions[session_id]['frame'] = frame
        logger.info("Desktop stream started")
    else:
        logger.info("Camera stream started (client-side)")
    logger.info(f"Session {session_id} reset for mode: {mode}")
    return jsonify({'redirect': url_for('chat', mode=mode)})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    session_id = session.get('session_id')
    if not session_id:
        logger.error("No session ID provided")
        return jsonify({'error': 'Session not found'}), 400
    if session_id not in sessions:
        logger.error(f"Session {session_id} not found")
        sessions[session_id] = {'mode': None, 'responses': [], 'frame': None}
        logger.info(f"Re-created session {session_id}")

    data = request.get_json()
    mode = data.get('mode')
    logger.info(f"Stop stream requested for mode: {mode}")

    sessions[session_id]['frame'] = None
    sessions[session_id]['mode'] = None
    sessions[session_id]['responses'] = []
    logger.info(f"Session {session_id} reset")
    return jsonify({'redirect': url_for('index')})

@app.route('/video_feed/<mode>')
def video_feed(mode):
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions or sessions[session_id]['mode'] != mode:
        logger.error(f"Stream not initialized for mode {mode}, session {session_id}")
        return Response("Stream not initialized", status=400)

    def generate():
        while session_id in sessions and sessions[session_id]['mode'] == mode:
            if mode == 'desktop':
                frame = capture_screen()
                if frame:
                    sessions[session_id]['frame'] = frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           base64.b64decode(frame) + b'\r\n')
            else:
                frame = sessions[session_id]['frame']
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           base64.b64decode(frame) + b'\r\n')
                else:
                    logger.debug("No camera frame available, skipping")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('camera_frame')
def handle_camera_frame(frame_data):
    session_id = session.get('session_id')
    if session_id and session_id in sessions and sessions[session_id]['mode'] == 'camera':
        processed_frame = process_frame(frame_data)
        if processed_frame:
            sessions[session_id]['frame'] = processed_frame
            logger.debug("Camera frame updated")

@app.route('/process_audio', methods=['POST'])
def process_audio():
    session_id = session.get('session_id')
    if not session_id:
        logger.error("No session ID provided")
        return jsonify({'error': 'Session not found'}), 400
    if session_id not in sessions:
        logger.error(f"Session {session_id} not found")
        sessions[session_id] = {'mode': None, 'responses': [], 'frame': None}
        logger.info(f"Re-created session {session_id}")

    data = request.get_json()
    prompt = data.get('prompt')
    mode = data.get('mode')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    if sessions[session_id]['mode'] != mode:
        logger.error(f"Stream not initialized for mode {mode}")
        return jsonify({'error': 'Stream not initialized'}), 400

    try:
        frame = sessions[session_id]['frame']
        logger.debug(f"Session {session_id} frame exists: {bool(frame)}")
        if not frame:
            logger.error(f"No frame available for mode: {mode}")
            if mode == 'camera':
                logger.info("Waiting for camera frame, using placeholder response")
                return jsonify({'response': 'Please wait, camera frame not ready yet'})
            return jsonify({'error': 'No frame available'}), 500

        system_prompt = SystemMessage(content="You are a helpful AI assistant analyzing images and responding to user prompts.")
        user_prompt = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{frame}"}
        ])
        logger.debug(f"Invoking LLM, frame size: {len(frame)} bytes")
        response = llm.invoke([system_prompt, user_prompt])
        response_text = response.content if hasattr(response, 'content') else str(response)
        sessions[session_id]['responses'].append({'prompt': prompt, 'response': response_text})
        logger.info(f"LLM response: {response_text[:50]}...")
        return jsonify({'response': response_text})
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return jsonify({'error': f"Failed to process audio: {str(e)}"}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=10000, allow_unsafe_werkzeug=True)