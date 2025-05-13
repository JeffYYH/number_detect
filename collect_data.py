import cv2
import HandTrackingModule as htm
import os
from utils import sequence_to_image

# Create dataset directories
dataset_dir = 'dataset'
for i in range(10):
    os.makedirs(os.path.join(dataset_dir, str(i)), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'non'), exist_ok=True)

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = htm.FindHands(detection_con=0.7)

# Initialize variables
current_label = None
drawing = False
current_sequence = []
# counters: f"{0-9}" for 0-9
counters = {str(i): 0 for i in range(10)}
counters['non'] = 0

for label in range(10):
    label_dir = os.path.join(dataset_dir, str(label))
    if not os.path.exists(label_dir):
        continue
    file_dir_s = os.listdir(label_dir)
    if len(file_dir_s) != 0:
        print(f"Label {label}: {len(file_dir_s)} samples")
        counters[str(label)] = len(file_dir_s) + 1

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hand landmarks
    lmlist = detector.getPosition(img, list(range(21)), draw=True)

    if len(lmlist) != 0:
        thumb_tip = lmlist[4]
        index_tip = lmlist[8]
        dist = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

        if dist < 20 and not drawing:
            drawing = True
            current_sequence = []
        elif dist >= 55 and drawing:
            drawing = False
            if current_label is not None and len(current_sequence) > 10:
                
                img_small = sequence_to_image(current_sequence)
                filename = os.path.join(dataset_dir, current_label, f"{current_label}_{counters[current_label]}.png")
                cv2.imwrite(filename, img_small)
                counters[current_label] += 1
        if drawing:
            current_sequence.append(index_tip)

        for i in range(1, len(current_sequence)):
            cv2.line(img, current_sequence[i - 1], current_sequence[i], (0, 255, 0), 2)

    # Display current label and sample count
    if current_label is not None:
        text = f"Label: {current_label}, Samples: {counters[current_label]}"
    else:
        text = "Press 0-9 for digits, 'n' for non-digit"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Handle key presses
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if 48 <= key <= 57:  # Digits 0-9
        current_label = chr(key)
    elif key == ord('n'):  # Non-digit class
        current_label = 'non'
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()