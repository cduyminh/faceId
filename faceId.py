import threading
import cv2
import time
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.config.fd_config import define_img_size
from annoy import AnnoyIndex
import os
import numpy as np
import torch
from image_batch_processor import process_image

torch.set_num_threads(1)

# Initialize face detection and recognition models
define_img_size(640)
device = 'cpu'
net = create_mb_tiny_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1500, device=device)
model_path = "models/pretrained/version-slim-320.pth"
net.load(model_path)

mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

class FaceLocker:
    def __init__(self):
        self.lockers = {i: {'state': 'free', 'id': None} for i in range(1, 7)}

    def assign_locker(self, id):
        for locker, details in self.lockers.items():
            if details['state'] == 'free':
                details['state'] = 'assigned'
                details['id'] = id
                return locker
        return None

    def free_locker(self, id):
        for locker, details in self.lockers.items():
            if details['id'] == id:
                details['state'] = 'free'
                details['id'] = None
                return locker
        return None

    def __str__(self):
        return f"Lockers: {self.lockers}"

class CameraCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

class FrameProcessor:
    def __init__(self, face_locker):
        self.display_window = 'Real-Time Face Recognition'
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.filename = ""
        self.found = False
        self.addingNew = False
        self.t = 0
        self.file_map = []
        self.face_locker = face_locker
        self.locker_size = 100
        self.loadMap()

    def loadMap(self):
        self.t = AnnoyIndex(512, 'euclidean')
        index_path = "db.ann"
        file_map_path = "file_map.npy"
        if os.path.exists(index_path) and os.path.exists(file_map_path):
            self.t.load(index_path)
            self.file_map = np.load(file_map_path, allow_pickle=True).item()
        else:
            self.rebuild_index()

    def rebuild_index(self):
        index_path = "db.ann"
        file_map_path = "file_map.npy"
        file_map = {}
        i = 0
        for filename in os.listdir('database'):
            img_path = os.path.join('database', filename)
            img = Image.open(img_path)
            face = mtcnn(img)
            if face is not None:
                embedding = resnet(face.unsqueeze(0)).detach().numpy().flatten()
                self.t.add_item(i, embedding)
                file_map[i] = filename
                i += 1
        self.t.build(10)
        self.t.save(index_path)
        np.save(file_map_path, file_map)
        print("Index built and saved.")

    def draw_lockers(self, frame):
        start_x = frame.shape[1] - (3 * self.locker_size) - 50  # Offset from right
        start_y = 50  # Offset from top
        locker_idx = 1
        for row in range(2):  # Two rows
            for col in range(3):  # Three columns
                top_left = (start_x + col * self.locker_size, start_y + row * self.locker_size)
                bottom_right = (top_left[0] + self.locker_size, top_left[1] + self.locker_size)
                locker_details = self.face_locker.lockers[locker_idx]
                color = (0, 255, 0) if locker_details['state'] == 'assigned' else (128, 128, 128)
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
                if locker_details['state'] == 'assigned':
                    img_path = os.path.join('database', locker_details['id'])
                    print("Assign: ",img_path)
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        img = img.resize((self.locker_size, self.locker_size), Image.ANTIALIAS)
                        img_array = np.array(img)
                        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = img
                locker_idx += 1

    def process_frame(self, frame):
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = self.curr_frame_time

        if not self.found and not len(self.filename) > 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, labels, probs = predictor.predict(img_rgb, 10, 0.6)

            for i in range(boxes.size(0)):
                box = boxes[i, :]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                if (probs[i]) < 0.999:
                    continue
                x1, y1, x2, y2 = map(int, box)
                face = Image.fromarray(img_rgb[y1:y2, x1:x2])
                face_tensor = mtcnn(face)
                
                if face_tensor is not None and not self.found:
                    face_embedding = resnet(face_tensor.unsqueeze(0)).detach().numpy().flatten()
                    neighbor_id, distance = self.t.get_nns_by_vector(face_embedding, 1, include_distances=True)
                    self.filename = self.file_map[neighbor_id[0]]
                    if (distance[0] > 0.9):
                        self.filename = "Not Found 'R' try again"
                    else:
                        locker_id = self.filename
                        self.filename = os.path.join('database', self.filename)
                        self.face_locker.free_locker(locker_id)
                    self.found = True

        if self.found and len(self.filename) > 0:
            if not self.filename == "Not Found 'R' try again":
                neighbor_img = cv2.imread(self.filename)
                neighbor_img = cv2.resize(neighbor_img, (200, 200))
                frame[10:210, 420:620] = neighbor_img
            cv2.putText(frame, self.filename, (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (18, 107, 107), 3)
        
        # Draw lockers on the frame
        self.draw_lockers(frame)

        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (18, 107, 107), 5)
        cv2.imshow(self.display_window, frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return False
        elif key & 0xFF == ord('r'):
            self.reset()
        elif key & 0xFF == ord('n'):
            threading.Thread(target=self.handle_new_image, args=(frame,)).start()
        return True
    
    def handle_new_image(self, frame):
        locker_id = process_image(frame)
        if locker_id != "Not Found 'R' try again":
            self.face_locker.assign_locker(locker_id)
        self.loadMap()
        self.addingNew = True

    def reset(self):
        self.filename = ""
        self.found = False
    
    def close(self):
        cv2.destroyAllWindows()

def main():
    face_locker = FaceLocker()
    camera = CameraCapture().start()
    frame_processor = FrameProcessor(face_locker)
    try:
        while True:
            frame = camera.read()
            if frame is not None and not frame_processor.process_frame(frame):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        frame_processor.close()

if __name__ == "__main__":
    main()
