import os
import sys
import uuid
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from facenet_pytorch import MTCNN, InceptionResnetV1
from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from annoy import AnnoyIndex

def process_image(orig_image):
    # Default setup
    net_type = 'RFB'  # 'RFB' for higher precision or 'slim' for faster processing
    input_size = 640  # Options: 128, 160, 320, 480, 640, 1280
    threshold = 0.6   # Detection threshold
    candidate_size = 1500  # Non-max suppression candidate size
    test_device = 'cpu'  # 'cuda:0' for GPU or 'cpu' for CPU

    define_img_size(input_size)

    result_path = "./database"
    label_path = "./models/voc-model-labels.txt"

    class_names = [name.strip() for name in open(label_path).readlines()]
    if net_type == 'slim':
        model_path = "models/pretrained/version-slim-320.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
    elif net_type == 'RFB':
        model_path = "models/pretrained/version-RFB-320.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)

    device = 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, min_face_size=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    t = AnnoyIndex(512, 'euclidean')
    index_path = "db.ann"
    file_map_path = "file_map.npy"
    file_map = {}
    i = 0

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    _id = uuid.uuid4().hex
    for i in range(boxes.size(0)):
        if (probs[i]) < 0.999:
            continue
        box = boxes[i, :]
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        label = f"{probs[i]:.2f}"
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped_img = orig_image[y1:y2, x1:x2]
        
        cropped_img_path = os.path.join(result_path, f"{_id}.jpg")
        if (abs(x2-x1) > 60):
            cv2.imwrite(cropped_img_path, cropped_img)

    for filename in os.listdir('database'):
      img_path = os.path.join('database', filename)
      print(img_path)
      img = PILImage.open(img_path)
      face = mtcnn(img)
      if face is not None:
          embedding = resnet(face.unsqueeze(0)).detach().numpy().flatten()
          t.add_item(i, embedding)
          file_map[i] = filename
          i += 1
    t.build(10)
    t.save(index_path)
    np.save(file_map_path, file_map)
    print("Index built and saved.")

    print(f"Found {len(probs)} objects.")
    return _id
