import cv2
import HandTrackingModule as htm
import torch
import torch.nn.functional as F
from utils import sequence_to_image, output, backspace
from model import CNN
from dataset import val_transform
import time
import threading
from mouse_control import mouse_control, click_control
import numpy as np

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = htm.FindHands(detection_con=0.75)

# Load the trained model
weight_path = 'digit_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load(weight_path))
# model.eval()

# Variables for drawing, prediction, mode control, and mouse stall
drawing = False
current_sequence = []
last_digit = None
confidence_threshold = 0.4  # Only output predictions with probability >= 0.4
tolerance_seconds = 0.2  # Tolerance for undetected hand
undetected_start_time = None
smoothed_index_tip = None  # For smoothing mouse movement
alpha = 0.7  # Smoothing factor for mouse movement
current_mode = "Mouse"  # Start in Mouse mode ("Mouse" or "Writing")
mouse_control_state = "Mouse"  # Mouse state ("Mouse" or "Stall")
mouse_control_timestamp = None  # Timestamp for mouse state transitions
previous_index_tip = None  # Previous index finger position for stall detection

# Function to calculate distance between two points
check_dist = lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) if p1 and p2 else 999

# Function to process the sequence and recognize the digit
def process_sequence(sequence):
    """Process the sequence in a separate thread to recognize a digit and simulate keyboard input."""
    global last_digit
    img_small = sequence_to_image(sequence)
    if img_small.max() == 0:  # Skip if the image is empty
        return
    img_small = torch.from_numpy(img_small).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_small = val_transform(img_small).to(device)
    with torch.no_grad():
        outputs = model(img_small)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        if max_prob.item() >= confidence_threshold:
            last_digit = predicted.item()
            output(last_digit)
        else:
            last_digit = None

# Function to check if all five fingers are up (open palm)
def all_fingers_up(lmlist):
    """Check if all five fingers are up by comparing fingertip and MCP joint positions."""
    if len(lmlist) != 21:
        return False
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Little
    finger_mcp = [2, 5, 9, 13, 17]    # MCP joints for each finger
    for tip, mcp in zip(finger_tips, finger_mcp):
        if lmlist[tip][1] >= lmlist[mcp][1]:  # If fingertip is not above MCP joint
            return False
    return True

def all_fingers_up(lmlist):
    """Check if all five fingers are up by comparing fingertip and MCP joint positions."""
    if len(lmlist) != 21:
        return False
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Little
    finger_mcp = [2, 5, 9, 13, 17]    # MCP joints for each finger
    for tip, mcp in zip(finger_tips, finger_mcp):
        if lmlist[tip][1] >= lmlist[mcp][1]:  # If fingertip is not above MCP joint
            return False
    return True

def backspace_action(lmlist, previous_index_tip):
    #print(f"{lmlist[8][0]} - {previous_index_tip[0]}")
    return (lmlist[8][0] - previous_index_tip[0]) > 100

while True:
    success, img = cap.read()
    if not success:
        break

    lmlist = detector.getPosition(img, list(range(21)), draw=True)

    if len(lmlist) != 0:
        thumb_tip = lmlist[4]  # Thumb tip
        index_tip = lmlist[8]  # Index finger tip
        dist = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

        # Smooth the index finger tip position for mouse movement
        if smoothed_index_tip is None:
            smoothed_index_tip = index_tip
        else:
            smoothed_index_tip = (
                alpha * index_tip[0] + (1 - alpha) * smoothed_index_tip[0],
                alpha * index_tip[1] + (1 - alpha) * smoothed_index_tip[1]
            )

        if current_mode == "Mouse":
            # Mouse control mode: handle mouse movement, stalling, and clicking
            if mouse_control_state == "Mouse":
                mouse_control(smoothed_index_tip, img.shape)  # Move mouse with smoothed position
                if check_dist(previous_index_tip, index_tip) < 10:  # Detect if finger is stationary
                    if mouse_control_timestamp and time.time() - mouse_control_timestamp > 0.33:
                        mouse_control_state = "Stall"  # Switch to Stall mode
                        print("Mouse stalled")
                else:
                    mouse_control_timestamp = time.time()  # Update timestamp if moving
                previous_index_tip = index_tip

            elif mouse_control_state == "Stall":
                if check_dist(previous_index_tip, index_tip) > 10:  # Detect movement to resume
                    if time.time() - mouse_control_timestamp > 0.33:
                        mouse_control_state = "Mouse"
                        mouse_control_timestamp = time.time()
                        print("Mouse resumed")
                else:
                    mouse_control_timestamp = time.time()

            # Detect click only in Stall mode to prevent movement during click
            if mouse_control_state == "Stall" and dist < 20:
                click_control(index_tip, img.shape)  # Perform click at current position
                current_mode = "Writing"  # Switch to Writing mode
                drawing = False  # Reset drawing state
                time.sleep(0.1)
                print("Click detected, switching to Writing mode")

        elif current_mode == "Writing":
            # Writing mode: handle digit drawing and recognition
            if dist < 20 and not drawing:  # Start drawing when thumb and index are close
                drawing = True
                current_sequence = []
                print("Started drawing")
            elif dist >= 60 and drawing:  # Stop drawing when thumb and index are far apart
                drawing = False
                if len(current_sequence) > 10:  # Process sequence if long enou
                    print("Stopped drawing, processing sequence")
                    threading.Thread(target=process_sequence, args=(current_sequence.copy(),)).start()
            if drawing:
                current_sequence.append(index_tip)  # Add point to drawing sequence
            else:
                # Check if you are doing the backs
                if backspace_action(lmlist, previous_index_tip):
                    # Perform backspace action
                    print("Backspace detected")
                    # Add your backspace action here
                    backspace()
                previous_index_tip = index_tip  # Update previous index tip position

            # Switch back to Mouse mode if all fingers are up
            if all_fingers_up(lmlist) and dist >= 60:
                current_mode = "Mouse"
                mouse_control_state = "Mouse"
                mouse_control_timestamp = time.time()
                drawing = False
                current_sequence = []
                print("Open palm detected, switching to Mouse mode")

        # Draw the writing sequence on the screen
        for i in range(1, len(current_sequence)):
            cv2.line(img, current_sequence[i - 1], current_sequence[i], (0, 255, 0), 2)
    else:
        # Handle case when hand is not detected
        if drawing:
            if undetected_start_time is None:
                undetected_start_time = time.time()
            else:
                elapsed_time = time.time() - undetected_start_time
                if elapsed_time > tolerance_seconds:
                    drawing = False
                    if len(current_sequence) > 10:
                        threading.Thread(target=process_sequence, args=(current_sequence.copy(),)).start()
                        print("Stopped drawing due to hand loss, processing sequence")
                    undetected_start_time = None

    # Display the last recognized digit and current mode
    if last_digit is not None:
        cv2.putText(img, f"Digit: {last_digit}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Mode: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()