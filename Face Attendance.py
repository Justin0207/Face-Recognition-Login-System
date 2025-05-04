import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import pickle
import face_recognition
import util
import datetime

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1300x600+250+100")
        self.main_window.title("Face Recognition App")

        self.tab_control = ttk.Notebook(self.main_window)
        self.login_tab = ttk.Frame(self.tab_control)
        self.logout_tab = ttk.Frame(self.tab_control)
        self.register_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.login_tab, text='Login')
        self.tab_control.add(self.register_tab, text='Register')
        self.logout_tab_added = False
        self.tab_control.pack(expand=1, fill="both")

        self.db_dir = './db'
        os.makedirs(self.db_dir, exist_ok=True)
        self.log_path = './log.txt'

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.init_login_tab()
        self.init_logout_tab()
        self.init_register_tab()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.process_webcam()

    def init_login_tab(self):
        self.login_image_label = util.get_img_label(self.login_tab)
        self.login_image_label.place(x=10, y=0, width=700, height=500)
        util.get_button(self.login_tab, "Login", "green", self.login).place(x=800, y=200)
        util.get_text_label(self.login_tab, "Ensure good lighting.\nOnly one face should be visible.").place(x=750, y=30)

    def init_logout_tab(self):
        self.logout_image_label = util.get_img_label(self.logout_tab)
        self.logout_image_label.place(x=10, y=0, width=700, height=500)
        util.get_button(self.logout_tab, "Logout", "red", self.logout).place(x=800, y=200)
        util.get_text_label(self.logout_tab, "Ensure good lighting.\nOnly one face should be visible.").place(x=750, y=30)

    def init_register_tab(self):
        self.register_image_label = util.get_img_label(self.register_tab)
        self.register_image_label.place(x=10, y=0, width=700, height=500)

        self.user_info_frame = tk.LabelFrame(self.register_tab, text="User Info", font=("Arial", 12))
        self.user_info_frame.place(x=750, y=70, width=400, height=200)

        tk.Label(self.user_info_frame, text="Username:", font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.entry_text_register_new_user = tk.Entry(self.user_info_frame, font=("Arial", 10), width=30)
        self.entry_text_register_new_user.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self.user_info_frame, text="Password:", font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.password_entry = tk.Entry(self.user_info_frame, font=("Arial", 10), width=30, show='*')
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self.user_info_frame, text="Confirm Password:", font=("Arial", 10)).grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.confirm_password_entry = tk.Entry(self.user_info_frame, font=("Arial", 10), width=30, show='*')
        self.confirm_password_entry.grid(row=2, column=1, padx=10, pady=10)

        util.get_button(self.register_tab, "Register", "green", self.accept_register_new_user).place(x=750, y=300)

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        # Update all image labels
        for label in [self.login_image_label, self.logout_image_label, self.register_image_label]:
            label.imgtk = imgtk
            label.configure(image=imgtk)

        self.main_window.after(20, self.process_webcam)

    def draw_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        return frame

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def detect_liveness(self, frame, ear_threshold=0.25):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            leftEAR = self.eye_aspect_ratio(shape_np[42:48])
            rightEAR = self.eye_aspect_ratio(shape_np[36:42])
            if (leftEAR + rightEAR) / 2.0 < ear_threshold:
                return True
        return False

    def login(self):
        frame = self.current_frame.copy()
        frame = self.draw_faces(frame)
        if not self.detect_liveness(frame):
            util.msg_box("Liveness Check Failed", "Please blink to verify you're alive.")
            return

        encodings = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not encodings:
            util.msg_box("Error", "No face detected.")
            return

        for file in os.listdir(self.db_dir):
            with open(os.path.join(self.db_dir, file), 'rb') as f:
                data = pickle.load(f)
                if face_recognition.compare_faces([data['embedding']], encodings[0])[0]:
                    self.prompt_for_password(data['name'])
                    return
        util.msg_box("Login Failed", "Face not recognized.")

    def prompt_for_password(self, username):
        self.tab_control.select(self.login_tab)
        top = tk.Toplevel(self.main_window)
        top.title("Enter Password")
        tk.Label(top, text="Password:").pack(pady=10)
        entry = tk.Entry(top, show='*')
        entry.pack(pady=10)
        tk.Button(top, text="Submit", command=lambda: self.verify_password(username, entry.get(), top)).pack(pady=10)

    def verify_password(self, username, entered_password, window):
        with open(os.path.join(self.db_dir, f'{username}.pickle'), 'rb') as f:
            stored_password = pickle.load(f)['password']
        if entered_password == stored_password:
            util.msg_box("Success", f"Welcome, {username}")
            self.log_event(f"{username} logged in")
            if not self.logout_tab_added:
                self.tab_control.add(self.logout_tab, text='Logout')
                self.logout_tab_added = True

            window.destroy()
        else:
            util.msg_box("Login Failed", "Incorrect password.")
        

    def logout(self):
        frame = self.current_frame.copy()
        frame = self.draw_faces(frame)
        if not self.detect_liveness(frame):
            util.msg_box("Liveness Check Failed", "Please blink to verify you're alive.")
            return

        encodings = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not encodings:
            util.msg_box("Error", "No face detected.")
            return

        for file in os.listdir(self.db_dir):
            with open(os.path.join(self.db_dir, file), 'rb') as f:
                data = pickle.load(f)
                if face_recognition.compare_faces([data['embedding']], encodings[0])[0]:
                    util.msg_box("Logout Success", f"Goodbye, {data['name']}!")
                    self.log_event(f"{data['name']} logged out")
                    return
        util.msg_box("Logout Failed", "Face not recognized.")

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get().strip()
        password = self.password_entry.get().strip()
        confirm_password = self.confirm_password_entry.get().strip()

        if not all([name, password, confirm_password]):
            util.msg_box("Error", "All fields are required.")
            return

        if password != confirm_password:
            util.msg_box("Error", "Passwords do not match.")
            return

        frame = self.current_frame.copy()
        frame = self.draw_faces(frame)
        encodings = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not encodings:
            util.msg_box("Error", "No face detected.")
            return

        data = {"name": name, "password": password, "embedding": encodings[0]}
        with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as f:
            pickle.dump(data, f)

        util.msg_box("Success", "User registered successfully.")
        self.entry_text_register_new_user.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.confirm_password_entry.delete(0, tk.END)

    def log_event(self, message):
        with open(self.log_path, 'a') as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - {message}\n")

    def start(self):
        self.main_window.mainloop()

if __name__ == '__main__':
    app = App()
    app.start()
