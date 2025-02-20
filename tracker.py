import cv2
import mediapipe as mp
import numpy as np
import winsound
import time

# MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Drawing spec
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Beep control
last_beep_time = 0
BEEP_COOLDOWN = 1.5  # seconds

def generate_frames(
    show_box=True,
    show_face_mesh=True,
    show_nose_line=True,
    enable_beep=True
):
    global last_beep_time

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            detection_results = face_detection.process(rgb)
            results = face_mesh.process(rgb)

            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if detection_results.detections and show_box:
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_2d = []
                    face_3d = []

                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x_lm = int(lm.x * frame.shape[1])
                            y_lm = int(lm.y * frame.shape[0])
                            face_2d.append([x_lm, y_lm])
                            face_3d.append([x_lm, y_lm, lm.z])

                            if idx == 1:
                                nose_2d = (x_lm, y_lm)
                                nose_3d = (x_lm, y_lm, lm.z * 3000)

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * frame.shape[1]
                    cam_matrix = np.array([[focal_length, 0, frame.shape[0] / 2],
                                           [0, focal_length, frame.shape[1] / 2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    pitch = angles[0] * 360  # x
                    yaw = angles[1] * 360    # y
                    roll = angles[2] * 360   # z

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

                    if beep_now and enable_beep:
                        current_time = time.time()
                        if current_time - last_beep_time > BEEP_COOLDOWN:
                            winsound.Beep(1000, 200)
                            last_beep_time = current_time

                    # Draw mesh
                    if show_face_mesh:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)

                    # Nose direction line (shorter)
                    if show_nose_line:
                        nose_3d_proj, _ = cv2.projectPoints(
                            np.array([nose_3d]), rot_vec, trans_vec, cam_matrix, dist_matrix)
                        p1 = nose_2d
                        p2 = (int(nose_2d[0] + yaw * 5), int(nose_2d[1]))  # Half length

                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

                    # Direction label
                    cv2.putText(frame, direction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Pitch: {round(pitch, 2)}', (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame, f'Yaw: {round(yaw, 2)}', (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
