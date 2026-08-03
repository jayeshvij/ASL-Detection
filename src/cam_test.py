'''
USAGE:
python cam_test.py 
'''

import torch
import joblib
import numpy as np
import cv2
import time
import cnn_models

# load label binarizer
lb = joblib.load('../outputs/lb.pkl')

# load model (CPU)
model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('../outputs/model.pth', map_location=torch.device('cpu')))
model.eval()

print('Model loaded')

def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (224, 224))
    return hand

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error while trying to open camera. Please check again...')

# get frame width & height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# video writer
out = cv2.VideoWriter('../outputs/asl.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      30,
                      (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # draw box
    cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)

    # extract hand
    hand = hand_area(frame)

    # preprocess
    image = np.transpose(hand, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0)

    # prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        pred_label = lb.classes_[preds.item()]

    # display
    cv2.putText(frame, pred_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)

    cv2.imshow('ASL Detection', frame)
    out.write(frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
out.release()
cv2.destroyAllWindows() 