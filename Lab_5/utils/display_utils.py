import cv2

def draw_fps(image, fps):
    """Draw FPS counter on image"""
    cv2.putText(image, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_text(image, text, position, color=(255, 255, 255), font_scale=1, thickness=2):
    """Draw text on image at specified position"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)

def draw_counter(image, count, label="Count"):
    """Draw a counter display on image"""
    text = f'{label}: {count}'
    cv2.putText(image, text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def draw_status(image, status, position=(10, 110)):
    """Draw status text on image"""
    color = (0, 255, 0) if status else (0, 0, 255)
    text = "DETECTED" if status else "NOT DETECTED"
    cv2.putText(image, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)