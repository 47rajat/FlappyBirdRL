import cv2
import numpy as np
import io

def show_image(window_title = ''):
    """
    Displays the provided stack of images provided to the generator (screen) in a cv2 window with the
    provided window title.

    Args
    ----
    `window_title` (str): The title that should be displayed in the cv2 window.
    """
    while True:
            screen = (yield)
            window_title = window_title
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            
            # stack each of the provided greyscale images horizontally.
            img = None
            for s in range(screen.shape[2]):
                if img is None:
                    img = screen[:, :, s]
                else:
                    img = np.hstack((img, screen[:, :, s]))
            
            cv2.imshow(window_title, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break