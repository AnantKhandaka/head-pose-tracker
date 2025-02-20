from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import mediapipe as mp
import winsound  # For beeping functionality
import time

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Global variables
last_beep_time = 0
BEEP_COOLDOWN = 1.5  # seconds

# Function to generate the video stream with selected options
def generate_frames(show_box, show_face_mesh, show_nose_line, show_text, enable_beep):
    global last_beep_time
    cap = cv2.VideoCapture(0)  # Start webcam capture
    
    # MediaPipe modules
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    
    # Drawing spec
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            # Flip the frame horizontally to mirror the view (fixing left/right orientation)
            frame = cv2.flip(frame, 1)
            
            # Convert frame to RGB for MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            detection_results = face_detection.process(rgb)
            results = face_mesh.process(rgb)
        
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Face detection with bounding box
            if show_box and detection_results.detections:
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
            
            # Face mesh and pose estimation
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # If mesh visualization is enabled
                    if show_face_mesh:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                    
                    # Head pose estimation
                    face_2d = []
                    face_3d = []
                    
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x_lm = int(lm.x * frame.shape[1])
                            y_lm = int(lm.y * frame.shape[0])
                            face_2d.append([x_lm, y_lm])
                            face_3d.append([x_lm, y_lm, lm.z])
                            
                            if idx == 1:  # Nose point
                                nose_2d = (x_lm, y_lm)
                                nose_3d = (x_lm, y_lm, lm.z * 3000)
                    
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    
                    # Camera matrix
                    focal_length = 1 * frame.shape[1]
                    cam_matrix = np.array([[focal_length, 0, frame.shape[0] / 2],
                                        [0, focal_length, frame.shape[1] / 2],
                                        [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    
                    # Calculate pose
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    
                    # Convert rotation angles
                    pitch = angles[0] * 360  # x
                    yaw = angles[1] * 360    # y
                    roll = angles[2] * 360   # z
                    
                    # Determine head direction
                    direction = "Forward"
                    beep_now = False
                    
                    if yaw < -10:
                        direction = "Looking Left"
                        beep_now = True
                    elif yaw > 10:
                        direction = "Looking Right"
                        beep_now = True
                    elif pitch < -10:
                        direction = "Looking Down"
                        beep_now = True
                    elif pitch > 10:
                        direction = "Looking Up"
                        beep_now = True
                    
                    # Trigger beep if needed and cooldown has passed
                    if beep_now and enable_beep:
                        current_time = time.time()
                        if current_time - last_beep_time > BEEP_COOLDOWN:
                            winsound.Beep(1000, 200)
                            last_beep_time = current_time
                    
                    # Draw nose direction line
                    if show_nose_line:
                        p1 = nose_2d
                        p2 = (int(nose_2d[0] + yaw * 5), int(nose_2d[1]))
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)
                    
                    # Display head pose information
                    if show_text:
                        cv2.putText(frame, direction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f'Pitch: {round(pitch, 2)}', (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(frame, f'Yaw: {round(yaw, 2)}', (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Yield the frame as bytes (MJPEG stream)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Completely replace any existing template with our custom one
    with open("templates/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/video")
def video_stream(
    show_box: bool = False,
    show_face_mesh: bool = False,
    show_nose_line: bool = False,
    show_text: bool = False,
    enable_beep: bool = False
):
    return StreamingResponse(
        generate_frames(
            show_box=show_box,
            show_face_mesh=show_face_mesh,
            show_nose_line=show_nose_line,
            show_text=show_text,
            enable_beep=enable_beep
        ),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )