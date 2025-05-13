import pyautogui
import time

_mouse_internal_timestamp = None

pyautogui.FAILSAFE = False

def mouse_control(pos, screen_size):
    """
    Control the mouse cursor based on the position of the index finger tip.
    """
    # Get the screen size
    screen_width, screen_height = pyautogui.size()

    # Normalize the index finger tip position to screen coordinates
    x = int(((1 - pos[0]  / screen_size[1]) * 1.5 - 0.22) * screen_width)
    y = int((pos[1]  / screen_size[0] * 1.5 - 0.22) * screen_height)

    pyautogui.moveTo(x, y)

def click_control(pos, screen_size):
    """
    Control the mouse click based on the position of the index finger tip.
    """
    # Get the screen size
    screen_width, screen_height = pyautogui.size()

    # Perform a mouse click
    pyautogui.click()
    print("Mouse clicked at position:", pos)