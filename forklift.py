import argparse
import sys
import logging
import re
import os
import csv
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import easyocr
from skimage import exposure
import supervision as sv
from ultralytics import YOLO
import smtplib
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

# ----------------------------------------------------------------------------
# Global Defaults (these can be overridden by command line args)
# ----------------------------------------------------------------------------
VIOLATION_SPEED = 5.0
VIOLATION_FRAMES = 5
MAX_OCR_THREADS = 4
DISPLAY_HEIGHT = 540


def generate_content_id(name):
    import uuid
    unique_id = uuid.uuid4()
    return f"<{name}-{unique_id}>"


def send_email(subject, files, args):
    """
    Sends an email with embedded images or files attached.
    Requires: args.send_email, args.password, args.recipient
    """
    if not args.send_email or not args.password or not args.recipient:
        print("Email not sent (missing credentials).")
        return

    outlook_user = args.send_email
    outlook_password = args.password
    to_email = args.recipient
    smtp_server = 'smtp.office365.com'
    smtp_port = 587
    local_time = str(time.ctime(time.time()))

    msg = MIMEMultipart()
    msg['From'] = outlook_user
    msg['To'] = to_email
    msg['Subject'] = subject

    html = f"""
    <html>
      <body>
        <p>There has been a forklift speed violation at {local_time}</p>
    """

    for file_path in files:
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' not found. Skipping.")
            continue

        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        # If image, embed inline
        if file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
            image_cid = generate_content_id('image')
            html += f'<img src="cid:{image_cid}"><br>'

            with open(file_path, "rb") as img:
                img_data = img.read()
                image = MIMEImage(img_data, _subtype=file_extension.replace('.', ''))
                image.add_header('Content-ID', f'<{image_cid}>')
                image.add_header('Content-Disposition', 'inline', filename=os.path.basename(file_path))
                msg.attach(image)
        else:
            # Other files: attach
            with open(file_path, "rb") as fil:
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type is None:
                    mime_type = 'application/octet-stream'
                main_type, sub_type = mime_type.split('/', 1)

                if main_type == 'text':
                    mime = MIMEText(fil.read().decode('utf-8'), _subtype=sub_type)
                elif main_type == 'image':
                    mime = MIMEImage(fil.read(), _subtype=sub_type)
                else:
                    mime = MIMEBase(main_type, sub_type)
                    mime.set_payload(fil.read())
                    encoders.encode_base64(mime)

                mime.add_header('Content-Disposition', 'attachment',
                                filename=os.path.basename(file_path))
                msg.attach(mime)

    html += "</body></html>"
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(outlook_user, outlook_password)
            server.sendmail(outlook_user, to_email, msg.as_string())
            print('Email sent successfully!')
    except Exception as e:
        print(f'Failed to send email. Error: {str(e)}')
        return


def save_violation_log(timestamp, forklift_id, plate_id, speed):
    log_path = 'forklift_logs.csv'
    write_header = (not os.path.exists(log_path)) or (os.stat(log_path).st_size == 0)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["TimeStamp", "Forklift_ID", "Plate_ID", "Speed(mph)"])
        writer.writerow([timestamp, forklift_id, plate_id, f"{speed:.1f}"])


def delete_files():
    """
    Removes files older than 1 day from the 'saved' folder.
    """
    folder = "saved"
    path = os.path.join(os.getcwd(), folder)

    if not os.path.isdir(path):
        return

    list_of_files = os.listdir(path)
    current_time = time.time()

    for i in list_of_files:
        file_location = os.path.join(path, i)
        file_time = os.stat(file_location).st_mtime
        # If older than 1 day => remove
        if file_time < current_time - (1 * 24 * 60 * 60):
            os.remove(file_location)


def save_violation_frame(frame, forklift_id, plate_id, x1, y1, x2, y2, speed):
    """
    Draws a red bounding box + "Violation!" and speed on the forklift,
    saves the frame with a timestamp + forklift/plate info in the filename,
    and logs the violation in CSV format.
    """
    if not os.path.exists("saved"):
        os.makedirs("saved")

    import uuid
    unique_id = uuid.uuid4().hex
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    local_time = time.ctime(time.time())

    # Draw red bounding box on forklift
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, "Violation", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Additional info on frame
    cv2.putText(frame, 'Violation!', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, local_time, (50, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.putText(frame, f"Speed: {speed:.1f} mph", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    pid_for_filename = plate_id if plate_id else forklift_id
    filename = f"saved/violation_{timestamp}_{pid_for_filename}_{unique_id}.jpg"
    cv2.imwrite(filename, frame)

    # Log it
    save_violation_log(timestamp, forklift_id, plate_id if plate_id else "NotFound", speed)
    return filename


# ==================== KALMAN + SPEED UTILS ====================
class ViewTransformer:
    """
    Optional perspective transform for real-world distances.
    """
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class SimpleKalmanFilter:
    def __init__(self, initial_estimate: float, process_variance: float = 0.1, measurement_variance: float = 1.0):
        self.estimate = initial_estimate
        self.error_covariance = 1.0
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def update(self, measurement: float) -> float:
        self.error_covariance += self.process_variance
        K = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + K * (measurement - self.estimate)
        self.error_covariance = (1 - K) * self.error_covariance
        return self.estimate


def convert_speed(speed_mps, unit='mph'):
    conversions = {'mph': 2.23694, 'kmh': 3.6, 'mps': 1.0}
    return speed_mps * conversions.get(unit, 1.0)


class SpeedEstimator:
    """
    Maintains a rolling window of positions per forklift ID
    and calculates a speed (in mph) using linear regression
    + an instantaneous displacement. Then applies a 1D Kalman
    filter for smoothing.
    """
    def __init__(self, fps, window_size=5):
        self.fps = fps
        self.window_size = window_size
        self.positions = defaultdict(lambda: deque(maxlen=int(fps * 3)))
        self.speed_filters = {}
        self.last_speeds = defaultdict(float)
        self.real_world_positions = defaultdict(lambda: deque(maxlen=int(fps * 3)))

    def add_position(self, tracker_id, point, real_world_point=None):
        self.positions[tracker_id].append(point)
        if real_world_point is not None:
            self.real_world_positions[tracker_id].append(real_world_point)
        if len(self.positions[tracker_id]) >= 2:
            return self.calculate_speed(tracker_id)
        return 0.0

    def calculate_speed(self, tracker_id):
        if (tracker_id in self.real_world_positions
                and len(self.real_world_positions[tracker_id]) >= self.window_size):
            pts = np.array(self.real_world_positions[tracker_id])
        else:
            pts = np.array(self.positions[tracker_id])

        if pts.shape[0] < self.window_size:
            return self.last_speeds.get(tracker_id, 0.0)

        # 1. Regression-based speed
        t = np.arange(self.window_size)
        window_pts = pts[-self.window_size:]
        slope_x, _ = np.polyfit(t, window_pts[:, 0], 1)
        slope_y, _ = np.polyfit(t, window_pts[:, 1], 1)
        reg_speed_pixels = np.sqrt(slope_x**2 + slope_y**2) * self.fps

        # 2. Instantaneous displacement
        displacement_pixels = np.linalg.norm(pts[-1] - pts[-self.window_size])
        dt = (self.window_size - 1) / self.fps
        inst_speed_pixels = displacement_pixels / dt

        measured_speed_mps = np.mean([reg_speed_pixels, inst_speed_pixels])

        # Kalman filter
        if tracker_id not in self.speed_filters:
            self.speed_filters[tracker_id] = SimpleKalmanFilter(
                initial_estimate=measured_speed_mps,
                process_variance=0.03,
                measurement_variance=0.15
            )
        filtered_speed_mps = self.speed_filters[tracker_id].update(measured_speed_mps)

        mph = convert_speed(filtered_speed_mps, 'mph')
        self.last_speeds[tracker_id] = mph
        return mph


# ==================== OCR / PLATE UTILS ====================
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def adjust_gamma(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    gamma = 0.8 if brightness < 100 else 1.2 if brightness > 150 else 1.0
    return exposure.adjust_gamma(image, gamma)


def enhance_image_colors(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    a = cv2.equalizeHist(a)
    b = cv2.equalizeHist(b)
    lab_image = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def remove_white_pixels(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    non_white_mask = cv2.bitwise_not(white_mask)
    filtered_image = cv2.bitwise_and(image, image, mask=non_white_mask)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)


def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9
    )
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    height, width = cleaned.shape
    center_y, center_x = height // 2, width // 2
    min_size = 100
    max_distance = width // 4
    result = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        size = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        if size > min_size and distance < max_distance:
            result[labels == i] = 255
    return result


def sharpen_image(image):
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def fill_black_holes(binary_image):
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=0)
    return dilated


def full_preprocess_pipeline(plate_image):
    plate_image = denoise_image(plate_image)
    plate_image = adjust_gamma(plate_image)
    plate_image = enhance_image_colors(plate_image)
    plate_image = remove_white_pixels(plate_image)
    plate_image = adaptive_thresholding(plate_image)
    plate_image = sharpen_image(plate_image)
    plate_image = fill_black_holes(plate_image)
    return plate_image


def quick_precheck(plate_crop, precheck_white_ratio=0.01):
    """
    Checks if there's enough 'foreground' in plate_crop
    to be worth doing full OCR.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    ratio = white_pixels / total_pixels
    return ratio > precheck_white_ratio


def find_frequent_element(li):
    count = {}
    mode = None
    max_count = 0
    for item in li:
        count[item] = count.get(item, 0) + 1
        if count[item] > max_count:
            max_count = count[item]
            mode = item
    return mode if max_count >= 3 else 0


def find_base_angle(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    else:
        angle = -angle
    return angle


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def detect_two_digit_numbers(image, reader, pattern):
    results = reader.readtext(image, detail=1, paragraph=False, decoder='greedy')
    valid_texts = []
    for (bbox, text, prob) in results:
        text_clean = text.strip().replace(" ", "")
        if pattern.fullmatch(text_clean):
            valid_texts.append(text_clean)
    return valid_texts


def extract_plate(image, box, margin=0.2):
    """
    Extracts the bounding box region from an image with margin.
    box should be [x1, y1, x2, y2].
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        aspect_ratio = box_w / box_h if box_h != 0 else 0
        if 2.0 < aspect_ratio < 6.0:
            margin *= 1.2
        dx = int(margin * box_w)
        dy = int(margin * box_h)
        x1e, y1e = max(0, x1 - dx), max(0, y1 - dy)
        x2e, y2e = min(w, x2 + dx), min(h, y2 + dy)
        crop = image[y1e:y2e, x1e:x2e]
        return crop
    except Exception as e:
        logging.error(f"Plate extraction failed: {e}")
        return None


# ==================== FORKLIFT TRACKING ====================
class ForkliftTrackerData:
    """
    Holds forklift-specific data: plate id, positions, texts, etc.
    """
    def __init__(self):
        self.plate_id = None
        self.position_history = deque(maxlen=10)
        self.raw_texts = []
        self.confirmed_plate = False


class ForkliftTracker:
    """
    Tracks forklift data, assigned plates, etc.
    """
    def __init__(self):
        # forklift_id -> ForkliftTrackerData
        self.forklift_data = {}
        self.plate_to_forklift = {}
        self.assigned_plates = set()
        self.plate_detection_history = defaultdict(lambda: deque(maxlen=10))
        self.forklift_positions = defaultdict(lambda: deque(maxlen=10))
        self.speed_estimator = None

    def set_speed_estimator(self, speed_estimator):
        self.speed_estimator = speed_estimator

    def get_bbox_hash(self, bbox):
        return hash(tuple(map(int, bbox)))

    def register_forklift(self, forklift_id, position):
        if forklift_id not in self.forklift_data:
            self.forklift_data[forklift_id] = ForkliftTrackerData()
        self.forklift_data[forklift_id].position_history.append(position)
        self.forklift_positions[forklift_id].append(position)

    def is_plate_processed(self, plate_bbox):
        bbox_hash = self.get_bbox_hash(plate_bbox)
        return bbox_hash in self.assigned_plates

    def mark_plate_processed(self, plate_bbox):
        bbox_hash = self.get_bbox_hash(plate_bbox)
        self.assigned_plates.add(bbox_hash)

    def assign_plate_to_forklift(self, forklift_id, plate_number):
        if forklift_id not in self.forklift_data:
            return False
        forklift_obj = self.forklift_data[forklift_id]
        if forklift_obj.plate_id is None and plate_number not in self.plate_to_forklift:
            forklift_obj.plate_id = plate_number
            forklift_obj.confirmed_plate = True
            forklift_obj.raw_texts = []
            self.plate_to_forklift[plate_number] = forklift_id
            return True
        return False

    def get_forklift_plate(self, forklift_id):
        if forklift_id in self.forklift_data:
            return self.forklift_data[forklift_id].plate_id
        return None

    def add_raw_text_to_forklift(self, forklift_id, text):
        if forklift_id in self.forklift_data:
            self.forklift_data[forklift_id].raw_texts.append(text)

    def get_raw_texts_for_forklift(self, forklift_id):
        if forklift_id in self.forklift_data:
            return self.forklift_data[forklift_id].raw_texts
        return []

    def get_forklift_speed(self, forklift_id):
        if self.speed_estimator is not None:
            return self.speed_estimator.calculate_speed(forklift_id)
        return 0.0

    def is_forklift_confirmed(self, forklift_id):
        if forklift_id in self.forklift_data:
            return self.forklift_data[forklift_id].confirmed_plate
        return False

    def update_plate_detection_history(self, forklift_id, plate_center):
        self.plate_detection_history[forklift_id].append(plate_center)

    def is_plate_near_previous_detection(self, plate_bbox, threshold=50):
        plate_center = np.array([
            (plate_bbox[0] + plate_bbox[2]) / 2,
            (plate_bbox[1] + plate_bbox[3]) / 2
        ])
        for _, history in self.plate_detection_history.items():
            if history:
                last_center = history[-1]
                if np.linalg.norm(plate_center - last_center) < threshold:
                    return True
        return False


# ========================= MAIN CODE =========================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Forklift speed violation + plate OCR + stop sign + multi-threaded OCR."
    )

    parser.add_argument("source_video_path", nargs="?", default=r"test3plate.mp4",
                        help="Path to the source video file", type=str)
    parser.add_argument("target_video_path", nargs="?", default=r"test_output1.mp4",
                        help="Path to the target video file (output)", type=str)
    parser.add_argument("--confidence_threshold", default=0.3, type=float,
                        help="Confidence threshold for YOLO")
    parser.add_argument("--iou_threshold", default=0.7, type=float,
                        help="IOU threshold for NMS")
    parser.add_argument("--target_fps", default=6.0, type=float,
                        help="Target processing FPS (skip frames)")
    parser.add_argument("--debug", action="store_true",
                        help="Bypass forklift polygon zone filter if true.")

    # Email fields
    parser.add_argument("--send_email", type=str, default="", help="Your Outlook email (blank to skip).")
    parser.add_argument("--password", type=str, default="", help="Your Outlook email password.")
    parser.add_argument("--recipient", type=str, default="", help="Recipient email address.")

    # NEW arguments for RTSP, violation speed/frames
    parser.add_argument("--rtsp_url", type=str, default="", 
                        help="RTSP stream URL (optional)")
    parser.add_argument("--violation_speed", type=float, default=5.0, 
                        help="Speed threshold (mph) for violation.")
    parser.add_argument("--violation_frames", type=int, default=5, 
                        help="Consecutive frames above threshold to declare violation.")

    return parser.parse_args()


# Class IDs
STOP_SIGN_CLASS = 2
FORKLIFT_CLASS = 0
PLATE_CLASS = 1

# For restricting plate region
PLATE_POLYGON = np.array([[726, 466], [1255, 474], [1403, 1046], [600, 1057]], dtype=np.int32)


def is_point_in_polygon(px, py, polygon):
    return cv2.pointPolygonTest(polygon, (px, py), False) >= 0


def _do_ocr_for_plate(frame, plate_box, forklift_tracker, nearest_forklift_id, reader, ocr_pattern):
    plate_crop = extract_plate(frame, plate_box)
    if plate_crop is None or not quick_precheck(plate_crop):
        return None

    #plate_crop = cv2.cuda_GpuMat(plate_crop)

    processed_plate = full_preprocess_pipeline(plate_crop)
    base_angle = find_base_angle(processed_plate)
    angle_offsets = [-10, -5, 0, 5, 10]
    best_texts = []

    for offset in angle_offsets:
        test_angle = base_angle + offset
        rotated_img = rotate_image(processed_plate, test_angle)
        valid_texts = detect_two_digit_numbers(rotated_img, reader, ocr_pattern)
        if len(valid_texts) > len(best_texts):
            best_texts = valid_texts
        if len(best_texts) >= 2:
            break

    return (nearest_forklift_id, best_texts) if best_texts else None


def main():
    

    global VIOLATION_SPEED, VIOLATION_FRAMES

    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Override global violation thresholds from args
    VIOLATION_SPEED = args.violation_speed
    VIOLATION_FRAMES = args.violation_frames
    print(f"VIOLATION_SPEED={VIOLATION_SPEED}, VIOLATION_FRAMES={VIOLATION_FRAMES}")

    # ------------------- NEW GSTREAMER + FALLBACK LOGIC -------------------
    # If user gave an RTSP URL, build GStreamer pipeline
    if args.rtsp_url.strip() != "":
        gst_pipeline = (
        f"rtspsrc location={args.rtsp_url} ! "
        "rtph264depay ! decodebin ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1"
        )

        print(f"RTSP URL: {args.rtsp_url}")

        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # Fallback if RTSP can't be opened
            print("Warning: Unable to open RTSP stream. Falling back to local 'test3plate.mp4'")
            cap = cv2.VideoCapture("test3plate.mp4")
        # ...removed: video_info = sv.VideoInfo.from_video_path("test3plate.mp4")
    else:
        print("No RTSP URL provided. Using local file: test3plate.mp4")
        cap = cv2.VideoCapture("test3plate.mp4")
        # ...removed: video_info = sv.VideoInfo.from_video_path("test3plate.mp4")

    if not cap.isOpened():
        print("Error: Unable to open any stream or file.")
        sys.exit(1)

    # Create video_info from VideoCapture properties instead of using sv.VideoInfo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    from collections import namedtuple
    VideoInfo = namedtuple("VideoInfo", ["fps", "resolution_wh"])
    video_info = VideoInfo(fps, (frame_width, frame_height))

    frame_gpu = cv2.cuda_GpuMat()

    # For storing the stop sign bounding box once we fix it
    fixed_stop_sign = None  # [x1, y1, x2, y2]

    model = YOLO("best2.engine")  # or "best.engine"

    # ByteTrack for forklift
    byte_track = sv.ByteTrack(frame_rate=video_info.fps,
                              track_activation_threshold=args.confidence_threshold)

    # Visualization setup
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale,
                                        text_thickness=thickness,
                                        text_position=sv.Position.BOTTOM_CENTER)
    trace_annotator = sv.TraceAnnotator(thickness=thickness,
                                        trace_length=int(video_info.fps) * 3,
                                        position=sv.Position.BOTTOM_CENTER)

    # Perspective transform (if needed)
    SOURCE = np.array([[568, 310], [1321, 330], [1600, 1002], [264, 976]])
    TARGET = np.array([[0, 0], [12.192, 0], [12.192, 12.192], [0, 12.192]])
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # OCR
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    ocr_pattern = re.compile(r'^(0[1-9]|[1-6]\d|70)$')

    # Forklift + speed
    forklift_tracker = ForkliftTracker()
    speed_estimator = SpeedEstimator(args.target_fps, window_size=5)
    forklift_tracker.set_speed_estimator(speed_estimator)

    # Consecutive frames for violation
    consecutive_speed_count = defaultdict(int)
    forklift_in_violation = set()

    # Expand forklift polygon zone if not debug
    min_x = np.min(SOURCE[:, 0])
    max_x = np.max(SOURCE[:, 0])
    mid_x = (min_x + max_x) / 2
    expanded_SOURCE = SOURCE.copy()
    for i in range(len(expanded_SOURCE)):
        if expanded_SOURCE[i, 0] < mid_x:
            expanded_SOURCE[i, 0] -= 50
        else:
            expanded_SOURCE[i, 0] += 50
    polygon_zone = sv.PolygonZone(polygon=expanded_SOURCE)

    skip_frames = max(int(video_info.fps / args.target_fps), 1) if video_info.fps > 0 else 1
    logging.info(f"Original FPS={video_info.fps}, target_fps={args.target_fps}, skip_frames={skip_frames}")

    # We already have 'cap' from the GStreamer or fallback file
    frame_index = 0

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % skip_frames != 0:
                sink.write_frame(frame)
                continue

            height, width = frame.shape[:2]

            # Inference
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Separate classes
            forklift_dets = detections[
                (detections.class_id == FORKLIFT_CLASS) &
                (detections.confidence > args.confidence_threshold)
            ]
            plate_dets = detections[
                (detections.class_id == PLATE_CLASS) &
                (detections.confidence > args.confidence_threshold)
            ]
            stop_dets = detections[
                (detections.class_id == STOP_SIGN_CLASS) &
                (detections.confidence > args.confidence_threshold)
            ]

            # Fix the stop sign if we haven't done so yet
            if fixed_stop_sign is None and len(stop_dets) > 0:
                fixed_stop_sign = stop_dets.xyxy[0].tolist()  # [x1, y1, x2, y2]

            # Filter forklifts with polygon unless debug
            if not args.debug:
                forklift_dets = forklift_dets[polygon_zone.trigger(forklift_dets)]
            forklift_dets = forklift_dets.with_nms(threshold=args.iou_threshold)

            # Track forklifts
            forklift_dets = byte_track.update_with_detections(detections=forklift_dets)
            forklift_centers = forklift_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            forklift_ids = forklift_dets.tracker_id

            # Real-world centers
            real_world_centers = view_transformer.transform_points(forklift_centers)

            # Register forklift positions, speeds
            for i, f_id in enumerate(forklift_ids):
                forklift_tracker.register_forklift(f_id, forklift_centers[i])
                speed_estimator.add_position(
                    f_id,
                    forklift_centers[i],
                    real_world_centers[i] if len(real_world_centers) > 0 else None
                )

            # --- MULTI-THREADED OCR FOR PLATES ---
            plate_tasks = []
            with ThreadPoolExecutor(max_workers=MAX_OCR_THREADS) as executor:
                for plate_box in plate_dets.xyxy:
                    px = (plate_box[0] + plate_box[2]) / 2
                    py = (plate_box[1] + plate_box[3]) / 2

                    # If outside polygon or near previous or processed => skip
                    if (not is_point_in_polygon(px, py, PLATE_POLYGON) or
                        forklift_tracker.is_plate_processed(plate_box) or
                        forklift_tracker.is_plate_near_previous_detection(plate_box)):
                        continue

                    forklift_tracker.mark_plate_processed(plate_box)

                    # find nearest forklift to assign
                    if len(forklift_centers) > 0:
                        plate_center = np.array([px, py])
                        distances = [np.linalg.norm(plate_center - fc) for fc in forklift_centers]
                        nearest_idx = np.argmin(distances)
                        nearest_forklift_id = forklift_ids[nearest_idx]

                        if forklift_tracker.is_forklift_confirmed(nearest_forklift_id):
                            continue

                        # Submit this plate detection to a separate thread
                        plate_tasks.append(
                            executor.submit(
                                _do_ocr_for_plate,
                                frame,
                                plate_box,
                                forklift_tracker,
                                nearest_forklift_id,
                                reader,
                                ocr_pattern
                            )
                        )

                # Wait for all OCR tasks
                for future in as_completed(plate_tasks):
                    result = future.result()
                    if result is None:
                        continue
                    assigned_forklift_id, best_texts = result
                    # Add raw texts
                    for text in best_texts:
                        forklift_tracker.add_raw_text_to_forklift(assigned_forklift_id, text)

                    # Check if we can finalize plate
                    raw_texts = forklift_tracker.get_raw_texts_for_forklift(assigned_forklift_id)
                    if raw_texts:
                        freq_result = find_frequent_element(raw_texts)
                        if freq_result != 0:
                            ok = forklift_tracker.assign_plate_to_forklift(assigned_forklift_id, freq_result)
                            if ok:
                                logging.info(f"Forklift #{assigned_forklift_id} => Plate assigned: {freq_result}")

            # Check forklift speeds & violations
            labels = []
            for i, f_id in enumerate(forklift_ids):
                speed_mph = forklift_tracker.get_forklift_speed(f_id)
                plate_str = forklift_tracker.get_forklift_plate(f_id)
                if plate_str is None:
                    lbl = f"Forklift #{f_id} Plate not found Speed:{speed_mph:.1f} mph"
                else:
                    lbl = f"Forklift #{f_id} Plate:{plate_str} Speed:{speed_mph:.1f} mph"

                if speed_mph >= VIOLATION_SPEED:
                    consecutive_speed_count[f_id] += 1
                else:
                    consecutive_speed_count[f_id] = 0

                if (consecutive_speed_count[f_id] >= VIOLATION_FRAMES and
                        f_id not in forklift_in_violation):
                    forklift_in_violation.add(f_id)
                    # find forklift's bounding box
                    for det_i, box in enumerate(forklift_dets.xyxy):
                        if forklift_ids[det_i] == f_id:
                            x1, y1, x2, y2 = map(int, box)
                            violation_img = frame.copy()
                            plate_for_log = forklift_tracker.get_forklift_plate(f_id)
                            violation_file = save_violation_frame(
                                violation_img,
                                f_id,
                                plate_for_log,
                                x1, y1, x2, y2,
                                speed_mph
                            )
                            # Optionally send email
                            send_email("Forklift Speed Violation", [violation_file], args)
                            lbl += " (Violation)"
                            break

                labels.append(lbl)

            # Draw bounding boxes for forklifts
            annotated_frame = frame.copy()

            # If we have a stop sign
            if fixed_stop_sign is not None:
                sx1, sy1, sx2, sy2 = map(int, fixed_stop_sign)
                cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "Stop Sign", (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # The trace_annotator draws forklift trails
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=forklift_dets)

            # Box & label
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=forklift_dets)
            annotated_frame = label_annotator.annotate(scene=annotated_frame,
                                                       detections=forklift_dets,
                                                       labels=labels)

            # Resize for display
            display_width = int((DISPLAY_HEIGHT / height) * width)
            display_frame = cv2.resize(annotated_frame, (display_width, DISPLAY_HEIGHT))

            forklift_count = len(forklift_ids)
            plate_count = len(plate_dets.xyxy)
            stop_count = (1 if fixed_stop_sign else 0)
            overlay_text_line = (f"Frame "
                                 f"Forklifts:{forklift_count} Plates:{plate_count} StopSign:{stop_count}")
            print(overlay_text_line)

            # Write to output video
            sink.write_frame(annotated_frame)

            cv2.imshow("Forklift Speed + Violation + StopSign (Threaded OCR)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    delete_files()
    sys.exit(0)


if __name__ == "__main__":
    main()
