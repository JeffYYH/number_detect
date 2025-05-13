import pyautogui
import time

_mouse_internal_timestamp = None

def mouse_control(pos, screen_size):
    """
    Control the mouse cursor based on the position of the index finger tip.
    """
    # Get the screen size
    screen_width, screen_height = pyautogui.size()

    # Normalize the index finger tip position to screen coordinates
    x = screen_width - int(pos[0] * screen_width / screen_size[1])
    y = int(pos[1] * screen_height / screen_size[0])

    # Move the mouse cursor to the new position

    


    #if _mouse_internal_timestamp is None: _mouse_internal_timestamp = time.time()

    #if (time.time() - _mouse_internal_timestamp) > 0.1:
        # Move the mouse cursor to the new position
    pyautogui.moveTo(x, y)
    #    _mouse_internal_timestamp = time.time()
    #else:
        # If the time since the last move is less than 0.1 seconds, do not move the mouse
    #    pass

def click_control(pos, screen_size):
    """
    Control the mouse click based on the position of the index finger tip.
    """
    # Get the screen size
    screen_width, screen_height = pyautogui.size()

    # Perform a mouse click
    pyautogui.click()
    print("Mouse clicked at position:", pos)