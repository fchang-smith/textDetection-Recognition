import numpy as np
import cv2

def decode_predictions(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    for y in range(geometry.shape[2]):
        for x in range(geometry.shape[3]):
            score = scores[0, 0, y, x]
            if score < scoreThresh:
                continue

            # Calculate geometry and offset
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            endX = offsetX + (cos * geometry[0, 1, y, x]) + (sin * geometry[0, 2, y, x])
            endY = offsetY - (sin * geometry[0, 1, y, x]) + (cos * geometry[0, 2, y, x])

            rect = (offsetX, offsetY, w, h, angle)
            detections.append(rect)
            confidences.append(score)

    return detections, confidences

def text_detector(image, net):
    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (320, 320)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    [rects, confidences] = decode_predictions(scores, geometry, 0.5)
    confidences = np.array(confidences)
    rects = np.array(rects, dtype=np.float32)

    indices = cv2.dnn.NMSBoxesRotated(rects, confidences, 0.5, 0.4)

    for i in indices:
        vertices = cv2.boxPoints(rects[i[0]])
        vertices = np.int0(vertices)
        cv2.polylines(orig, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)

    return orig

# Load the pre-trained EAST text detector
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Load an image
image = cv2.imread('image5.jpg')

# Detect text in the image
text_detected = text_detector(image, net)

# Display the image
cv2.imshow("Text Detection", text_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()
