import os
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ultralytics import YOLO
from threading import Thread, Lock, Event
import numpy as np
from datetime import datetime
import geocoder
from PIL import Image, ImageTk, ImageOps
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import requests
from twilio.rest import Client
# Global variables
stop_event = Event()
output_dir = "detected_frames"
confidence_threshold = 0.5
nms_threshold = 0.4
video_thread = None
email_thread = None
manual_location = None
last_location_time = 0  # To track last location request time
location_cache = None




# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# results = model.train(
#     data='config.yaml',       
#     epochs=50,                
#     imgsz=640,                
#     batch=16,                 
#     name='rescue_detector'    
# )
# metrics = model.val()
# print("mAP@0.5:", metrics.box.map50)
# print("mAP@0.5:0.95:", metrics.box.map)


EMAIL_ADDRESS = "vaishu.vaitla@gmail.com"  
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  
RECIPIENT_EMAIL = "vaishu.vaitla1@gmail.com"  

relevant_classes = ['person',  'dog', 'cat', 'horse', 'sheep', 'cow', 'bird', 'elephant', 'car', 'bus', 'truck', 'bike', 'motorcycle']

def send_email_alert(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email alert sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def email_alert_thread(detection_counter):
    while not stop_event.is_set():
        counts = detection_counter.get_counts()
        if any(counts.values()):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            city, state, country, (lat, lng) = get_location()
            subject = f"Detection Alert - {timestamp}"
            body = (f"Detections at {timestamp}\n"
                   f"Location: {city}, {state}, {country}\n"
                   #f"Coordinates: Lat {lat}, Lng {lng}\n\n"
                   f"Coordinates: Lat {lat:.6f}, Lng {lng:.6f}\n"
                   f"Persons: {counts['person']}\n"
                   
                   f"Animals: {counts['animal']}\n"
                   f"Vehicles: {counts['vehicle']}")
            send_email_alert(subject, body)
            send_sms_alert(body)
            detection_counter.reset()

            

        time.sleep(10)


def send_sms_alert(body):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = '+19163132526'
    to_number = '+919110312728'        

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body,
            from_=twilio_number,
            to=to_number
        )
        print(f"✅ SMS sent: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

def detect_weather_condition(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    saturation = np.mean(hsv[:, :, 1])
    
    if brightness < 50:
        return "night"
    elif saturation < 30 and brightness > 150:
        return "foggy"
    elif np.var(image) < 1000:
        return "rainy"
    else:
        return "clear"

def enhance_low_light(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)

def enhance_thermal_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(enhanced, None, 15, 15, 7, 21)

def adaptive_deblur(image, weather):
    if weather == "rainy":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def adaptive_dehaze(image, weather):
    image_float = image.astype(np.float32) / 255.0
    dark_channel = np.min(image_float, axis=2)
    atmospheric_light = np.percentile(dark_channel, 95) if weather == "foggy" else np.percentile(dark_channel, 99.9)
    transmission_factor = 0.85 if weather == "foggy" else 0.95
    transmission = 1 - transmission_factor * (dark_channel / atmospheric_light)
    dehazed = np.zeros_like(image_float)
    for i in range(3):
        dehazed[:, :, i] = (image_float[:, :, i] - atmospheric_light) / np.maximum(transmission, 0.1) + atmospheric_light
    return np.clip(dehazed * 255, 0, 255).astype(np.uint8)

def get_location():
    global manual_location, last_location_time, location_cache
    if manual_location:
        return manual_location
    
    # Use cached location if recent (within 60 seconds)
    current_time = time.time()
    if location_cache and (current_time - last_location_time < 60):
        return location_cache

    # Try geocoder first
    try:
        g = geocoder.ip('me')
        if g.ok and g.latlng:
            print(f"Location from geocoder: {g.city}, {g.state}, {g.country}")
            location_cache = (g.city or "Unknown", g.state or "Unknown", g.country or "Unknown", g.latlng)
            last_location_time = current_time
            return location_cache
    except Exception as e:
        print(f"Geocoder failed: {e}")

    # Fallback to ip-api.com with rate limiting
    try:
        time.sleep(1)  # Add delay to avoid rate limiting
        response = requests.get('http://ip-api.com/json', timeout=5)
        if response.status_code == 429:
            print("IP-API rate limit exceeded")
        else:
            data = response.json()
            if data['status'] == 'success':
                print(f"Location from ip-api: {data['city']}, {data['regionName']}, {data['country']}")
                location_cache = (data['city'], data['regionName'], data['country'], (data['lat'], data['lon']))
                last_location_time = current_time
                return location_cache
    except requests.RequestException as e:
        print(f"IP-API fallback failed: {e}")

    # If all else fails
    print("All location services failed")
    return "Unknown", "Unknown", "Unknown", (0.0, 0.0)

def set_manual_location(city_entry, state_entry, country_entry, lat_entry, lng_entry):
    global manual_location
    try:
        city = city_entry.get() or "Unknown"
        state = state_entry.get() or "Unknown"
        country = country_entry.get() or "Unknown"
        lat = float(lat_entry.get() or 0.0)
        lng = float(lng_entry.get() or 0.0)
        manual_location = (city, state, country, (lat, lng))
        messagebox.showinfo("Success", "Manual location set successfully")
    except ValueError:
        messagebox.showerror("Error", "Invalid latitude or longitude format")

def open_location_window():
    location_window = tk.Toplevel()
    location_window.title("Set Manual Location")
    location_window.geometry("300x250")
    location_window.configure(bg='white')

    tk.Label(location_window, text="Manual Location", font=("Arial", 14, "bold"),
            bg='white', fg='black').pack(pady=10)

    frame = tk.Frame(location_window, bg='white')
    frame.pack(pady=10)

    tk.Label(frame, text="City:", bg='white', fg='black').grid(row=0, column=0, padx=5)
    city_entry = tk.Entry(frame)
    city_entry.grid(row=0, column=1)

    tk.Label(frame, text="State:", bg='white', fg='black').grid(row=1, column=0, padx=5)
    state_entry = tk.Entry(frame)
    state_entry.grid(row=1, column=1)

    tk.Label(frame, text="Country:", bg='white', fg='black').grid(row=2, column=0, padx=5)
    country_entry = tk.Entry(frame)
    country_entry.grid(row=2, column=1)

    tk.Label(frame, text="Latitude:", bg='white', fg='black').grid(row=3, column=0, padx=5)
    lat_entry = tk.Entry(frame)
    lat_entry.grid(row=3, column=1)

    tk.Label(frame, text="Longitude:", bg='white', fg='black').grid(row=4, column=0, padx=5)
    lng_entry = tk.Entry(frame)
    lng_entry.grid(row=4, column=1)

    tk.Button(location_window, text="Set Location",
             command=lambda: set_manual_location(city_entry, state_entry, country_entry, lat_entry, lng_entry),
             bg='#add8e6', fg='black', font=("Arial", 10, "bold")).pack(pady=10)

class DetectionCounter:
    def __init__(self):
        self.person_count = 0
        self.animal_count = 0
        self.vehicle_count = 0
        
        self.lock = Lock()

    def update_count(self, label):
        with self.lock:
            if label == 'person':
                self.person_count += 1
            elif label in ['dog', 'cat', 'horse', 'sheep', 'cow', 'bird', 'elephant']:
                self.animal_count += 1
            elif label in ['car', 'bus', 'truck', 'bike', 'motorcycle']:
                self.vehicle_count += 1
            

    def get_counts(self):
        with self.lock:
            return {
                'person': self.person_count,
                'animal': self.animal_count,
                'vehicle': self.vehicle_count,
                
            }

    def reset(self):
        with self.lock:
            self.person_count = 0
            self.animal_count = 0
            self.vehicle_count = 0
            

def is_thermal_image(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return True
    color_variance = np.var(image, axis=(0, 1))
    return np.mean(color_variance) < 100

def filter_relevant_detections(results):
    filtered_results = []
    for result in results[0].boxes.data:
        label_index = int(result[5])
        confidence = result[4]
        label = results[0].names[label_index]
        if confidence >= confidence_threshold and label in relevant_classes:
            filtered_results.append(result)
    return filtered_results

def process_frame(frame, model, detection_counter):
    weather = detect_weather_condition(frame)
    if weather == "night":
        frame = enhance_low_light(frame)
    frame = adaptive_deblur(frame, weather)
    frame = adaptive_dehaze(frame, weather)
    if is_thermal_image(frame):
        frame = enhance_thermal_image(frame)

    results = model(frame)
    filtered_results = filter_relevant_detections(results)

    annotated_frame = frame.copy()
    class_colors = {
        'person': (0, 255, 0),
        
        'vehicle': (0, 0, 255),
        'animal': (255, 0, 255)
    }
    
    for result in filtered_results:
        x1, y1, x2, y2, confidence, label_index = result
        label = results[0].names[int(label_index)]
        color = class_colors.get(label, (0, 255, 0)) if label in ['person'] else \
                class_colors['vehicle'] if label in ['car', 'bus', 'truck', 'bike', 'motorcycle'] else \
                class_colors['animal']
        
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        detection_counter.update_count(label)

    cv2.putText(annotated_frame, f"Weather: {weather}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return annotated_frame, filtered_results

def run(weights='yolov8m.pt', source='0', view_img=True):
    global video_thread, email_thread
    try:
        model = YOLO(weights)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return

    detection_counter = DetectionCounter()
    detection_counter.reset()

    cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source.")
        return

    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0

    email_thread = Thread(target=email_alert_thread, args=(detection_counter,))
    email_thread.daemon = True
    email_thread.start()

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, filtered_results = process_frame(frame, model, detection_counter)

    # Add this line right after process_frame
        

        if len(filtered_results) > 0:
            frame_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            city, state, country, (lat, lng) = get_location()
            city = city.replace('/', '_').replace('\\', '_')
            state = state.replace('/', '_').replace('\\', '_')
            country = country.replace('/', '_').replace('\\', '_')
            filename = f"frame_{timestamp}_{city}_{state}_{country}_{lat}_{lng}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), annotated_frame)

        counts = detection_counter.get_counts()
        print(f"Persons: {counts['person']}, Animals: {counts['animal']}, Vehicles: {counts['vehicle']}")

        if view_img:
            cv2.imshow('Live Monitoring', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()

    cap.release()
    cv2.destroyAllWindows()
    video_thread = None

def test_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            model = YOLO('yolov8m.pt')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        annotated_image, _ = process_frame(image, model, DetectionCounter())
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        city, state, country, (lat, lng) = get_location()
        city = city.replace('/', '_').replace('\\', '_')
        state = state.replace('/', '_').replace('\\', '_')
        country = country.replace('/', '_').replace('\\', '_')
        filename = f"test_output_{timestamp}_{city}_{state}_{country}_{lat}_{lng}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), annotated_image)

        
        # Send alert for test image
        detection_counter = DetectionCounter()
        annotated_image, _ = process_frame(image, model, detection_counter)
        counts = detection_counter.get_counts()
        if any(counts.values()):
            subject = f"Test Image Detection Alert - {timestamp}"
            body = (f"Detections at {timestamp}\n"
                    f"Location: {city}, {state}, {country}\n"
                    f"Coordinates: Lat {lat:.6f}, Lng {lng:.6f}\n\n"
                    f"Persons: {counts['person']}\n"
                    f"Animals: {counts['animal']}\n"
                    f"Vehicles: {counts['vehicle']}")
            send_email_alert(subject, body)
            send_sms_alert(body)
            print("[DEBUG] Alert sent for test image.")
        else:
            print("[DEBUG] No relevant detections for test image.")

        cv2.imshow('Test Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def view_gallery():
    gallery_window = tk.Toplevel()
    gallery_window.title("Detected Images Gallery")
    gallery_window.configure(bg='white')
    gallery_window.geometry("1000x800")

    canvas = tk.Canvas(gallery_window, bg='white')
    scrollbar = ttk.Scrollbar(gallery_window, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    detected_images = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')]
    detected_images.sort(key=os.path.getmtime, reverse=True)

    row, col = 0, 0
    for image_path in detected_images:
        image = Image.open(image_path)
        image.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(image)

        frame = tk.Frame(scrollable_frame, bg='white', relief="raised", borderwidth=2)
        frame.grid(row=row, column=col, padx=10, pady=10)

        label = tk.Label(frame, image=photo, bg='white')
        label.image = photo
        label.pack(pady=5)

        metadata = os.path.basename(image_path).split('_')
        timestamp = metadata[1] + " " + metadata[2]
        location = f"{metadata[3]}, {metadata[4]}, {metadata[5]}"
        coordinates = f"Lat: {metadata[6]}, Lng: {metadata[7].replace('.jpg', '')}"
        tk.Label(frame, text=f"Time: {timestamp}\nLoc: {location}\nCoords: {coordinates}",
                bg='white', fg='black', font=("Arial", 10)).pack()

        def create_zoom_window(img_path):
            zoom_window = tk.Toplevel(bg='white')
            zoom_window.title("Zoomed Image")
            original_img = Image.open(img_path)
            current_img = original_img.copy()
            photo_zoom = ImageTk.PhotoImage(current_img)
            
            label_zoom = tk.Label(zoom_window, image=photo_zoom, bg='white')
            label_zoom.image = photo_zoom
            label_zoom.pack(pady=10)

            button_frame = tk.Frame(zoom_window, bg='white')
            button_frame.pack(pady=5)

            def zoom_in():
                nonlocal current_img
                width, height = current_img.size
                current_img = current_img.resize((int(width * 1.2), int(height * 1.2)), Image.LANCZOS)
                new_photo = ImageTk.PhotoImage(current_img)
                label_zoom.configure(image=new_photo)
                label_zoom.image = new_photo

            def zoom_out():
                nonlocal current_img
                width, height = current_img.size
                current_img = current_img.resize((int(width / 1.2), int(height / 1.2)), Image.LANCZOS)
                new_photo = ImageTk.PhotoImage(current_img)
                label_zoom.configure(image=new_photo)
                label_zoom.image = new_photo

            tk.Button(button_frame, text="Zoom In", command=zoom_in, bg='#add8e6', fg='black',
                     font=("Arial", 10, "bold")).pack(side="left", padx=5)
            tk.Button(button_frame, text="Zoom Out", command=zoom_out, bg='#add8e6', fg='black',
                     font=("Arial", 10, "bold")).pack(side="left", padx=5)

        tk.Button(frame, text="Zoom", command=lambda p=image_path: create_zoom_window(p),
                 bg='#add8e6', fg='black', font=("Arial", 10, "bold")).pack(pady=5)

        col += 1
        if col > 3:
            col = 0
            row += 3

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

def select_video():
    global video_thread
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        stop_event.clear()
        video_thread = Thread(target=run, args=('yolov8m.pt', file_path, True))
        video_thread.start()

def start_webcam():
    global video_thread
    stop_event.clear()
    video_thread = Thread(target=run, args=('yolov8m.pt', '0', True))
    video_thread.start()

def stop_processing():
    stop_event.set()
    if video_thread and video_thread.is_alive():
        video_thread.join()
    if email_thread and email_thread.is_alive():
        email_thread.join()

def create_gui():
    root = tk.Tk()
    root.title("Disaster Rescue Tracking")
    root.geometry("400x550")  # Increased height for new button
    root.configure(bg='white')

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12, "bold"))

    main_frame = tk.Frame(root, bg='white')
    main_frame.pack(expand=True)

    tk.Label(main_frame, text="Disaster Rescue System", font=("Arial", 18, "bold"),
            bg='white', fg='black').pack(pady=20)

    tk.Button(main_frame, text="Live Webcam", command=start_webcam,
             bg='#add8e6', fg='black', font=("Arial", 12, "bold"),
             width=20, height=2).pack(pady=10)
    
    tk.Button(main_frame, text="Upload Video", command=select_video,
             bg='#add8e6', fg='black', font=("Arial", 12, "bold"),
             width=20, height=2).pack(pady=10)
    
    tk.Button(main_frame, text="Test Image", command=test_image,
             bg='#add8e6', fg='black', font=("Arial", 12, "bold"),
             width=20, height=2).pack(pady=10)
    
    tk.Button(main_frame, text="View Gallery", command=view_gallery,
             bg='#add8e6', fg='black', font=("Arial", 12, "bold"),
             width=20, height=2).pack(pady=10)
    
    tk.Button(main_frame, text="Set Location", command=open_location_window,
             bg='#add8e6', fg='black', font=("Arial", 12, "bold"),
             width=20, height=2).pack(pady=10)
    
    tk.Button(main_frame, text="Stop", command=stop_processing,
             bg='#87ceeb', fg='black', font=("Arial", 12, "bold"),
             width=20, height=2).pack(pady=10)
    
    tk.Button(main_frame, text="Exit", command=lambda: [stop_processing(), root.destroy()],
             bg='#87ceeb', fg='black', font=("Arial", 12, "bold"),
             width=20, height=2).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()