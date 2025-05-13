import cv2
import HandTrackingModule as htm
import torch
import torch.nn.functional as F
from utils import sequence_to_image
from model import CNN
from dataset import val_transform


# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = htm.FindHands(detection_con=0.75)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('digit_model.pth'))
model.eval()


# Variables for drawing and prediction
drawing = False
current_sequence = []
last_digit = None
confidence_threshold = 0.7  # Only output predictions with probability >= 0.7

while True:
    success, img = cap.read()
    if not success:
        break

    lmlist = detector.getPosition(img, list(range(21)), draw=True)

    if len(lmlist) != 0:
        thumb_tip = lmlist[4]
        index_tip = lmlist[8]
        dist = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

        if dist < 20 and not drawing:
            drawing = True
            current_sequence = []
        elif dist >= 65 and drawing:
            drawing = False
            if len(current_sequence) > 10:
                img_small = sequence_to_image(current_sequence)
                img_small = torch.from_numpy(img_small).float().unsqueeze(0).unsqueeze(0) / 255.0
                img_small = val_transform(img_small).to(device)
                with torch.no_grad():
                    outputs = model(img_small)
                    probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
                    max_prob, predicted = torch.max(probabilities, 1)
                    if max_prob.item() >= confidence_threshold:
                        last_digit = predicted.item()
                    else:
                        last_digit = None  # Suppress output if confidence is too low
        if drawing:
            current_sequence.append(index_tip)

        for i in range(1, len(current_sequence)):
            cv2.line(img, current_sequence[i - 1], current_sequence[i], (0, 255, 0), 2)

    if last_digit is not None:
        cv2.putText(img, f"Digit: {last_digit}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()