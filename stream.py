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
device = 'cpu'  # or 'cuda' for GPU
net = create_mb_tiny_fd(2, is_test=True, device=device)
predictor = create_mb_tiny_fd_predictor(net, candidate_size=1500, device=device)
model_path = "models/pretrained/version-slim-320.pth"
net.load(model_path)


mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



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
    def __init__(self):
        self.display_window = 'Real-Time Face Recognition'
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.filename = ""
        self.found = False
        self.addingNew = False
        self.t = 0
        self.file_map = []
        self.loadMap()

    def loadMap(self):
        # Initialize Annoy index for nearest neighbor search
        self.t = AnnoyIndex(512, 'euclidean')
        index_path = "db.ann"
        file_map_path = "file_map.npy"
        # Load or create Annoy index
        if os.path.exists(index_path) and os.path.exists(file_map_path):
            self.t.load(index_path)
            self.file_map = np.load(file_map_path, allow_pickle=True).item()
        else:
            print("Building new index...")
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

    def process_frame(self, frame):
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = self.curr_frame_time

        if not self.found and not len(self.filename) > 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Face detection
            boxes, labels, probs = predictor.predict(img_rgb, 10, 0.6)

            for i in range(boxes.size(0)):
                box = boxes[i, :]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                if (probs[i]) < 0.999:
                    continue
                x1, y1, x2, y2 = map(int, box)
                # Crop and prepare face for recognition
                face = Image.fromarray(img_rgb[y1:y2, x1:x2])
                face_tensor = mtcnn(face)
                
                if face_tensor is not None and not self.found:
                    face_embedding = resnet(face_tensor.unsqueeze(0)).detach().numpy().flatten()
                    neighbor_id, distance = self.t.get_nns_by_vector(face_embedding, 2, include_distances=True)
                    self.filename = self.file_map[neighbor_id[0]]
                    print("distance: ", str(distance[0]) + " - " + str(distance[1]))
                    if (distance[0] > 0.9):
                        self.filename = "Not Found 'R' try again"
                    else:
                        self.filename = os.path.join('database', self.filename)
                    
                    self.found = True

        if self.found and len(self.filename) > 0:
            if not self.filename == "Not Found 'R' try again":
                neighbor_img = cv2.imread(self.filename)
                neighbor_img = cv2.resize(neighbor_img, (200, 200))
                # Display neighbor image next to the webcam feed
                frame[10:210, 420:620] = neighbor_img
            cv2.putText(frame, self.filename, (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (18, 107, 107), 3)
        
        # Display FPS
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (18, 107, 107), 5)
        cv2.imshow(self.display_window, frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return False
        elif key & 0xFF == ord('r'):  # Reset on pressing 'r'
            self.reset()
        elif key & 0xFF == ord('n'):
            threading.Thread(target=self.handle_new_image, args=(frame,)).start()
        return True
    
    def handle_new_image(self, frame):
        lockerid = process_image(frame)
        self.loadMap()
        self.addingNew = True

    def reset(self):
        self.filename = ""
        self.found = False
    def close(self):
        cv2.destroyAllWindows()

def main():
    camera = CameraCapture().start()
    frame_processor = FrameProcessor()
    try:
        while True:
            frame = camera.read()
            if frame is not None and not frame_processor.process_frame(frame):
                break
            time.sleep(0.01)  # Sleep to allow for other processes
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        frame_processor.close()

if __name__ == "__main__":
    main()
