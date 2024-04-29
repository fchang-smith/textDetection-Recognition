import numpy as np
import cv2

def decode_predictions(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    # loop over the number of rows
    for y in range(0, geometry.shape[2]):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, geometry.shape[3]):
            score = scoresData[x]

            if score < scoreThresh:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            p1 = offsetX + (cos * xData1[x]) + (sin * xData2[x])
            p2 = offsetY - (sin * xData1[x]) + (cos * xData2[x])

            centerX = int(p1)
            centerY = int(p2)
            width = int(w)
            height = int(h)

            # Create a RotatedRect object
            detections.append((centerX, centerY, width, height, angle))
            confidences.append(float(score))

    return detections, confidences

def text_detector(image, net):
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    [rects, confidences] = decode_predictions(scores, geometry, 0.5)

    # Prepare for NMSBoxesRotated
    confidences = np.array(confidences)
    rects = np.array(rects, dtype="float32")

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxesRotated(rects, confidences, 0.5, 0.4)

    # Draw bounding boxes
    for i in indices:
        # vertices of the rotated rect
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
