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

            # if score is lower than the threshold, ignore
            if score < scoreThresh:
                continue

            # calculate offset
            offsetX, offsetY = x * 4.0, y * 4.0

            # calculate angle and the cosine and sine of angle
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # calculate the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # calculate starting and ending x coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to the list
            detections.append((startX, startY, endX, endY))
            confidences.append(score)

    # return both the detection and confidence list
    return [detections, confidences]

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
    boxes = cv2.dnn.NMSBoxesRotated(rects, confidences, 0.5, 0.4)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return orig

# Load the pre-trained EAST text detector
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Load an image
image0 = cv2.imread('image0.jpg')
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')
image3 = cv2.imread('image3.jpg')
image4 = cv2.imread('image4.jpg')
image5 = cv2.imread('image5.jpg')
image6 = cv2.imread('image6.jpg')
image7 = cv2.imread('image7.jpg')
image8 = cv2.imread('image8.jpg')
image9 = cv2.imread('image9.jpg')
array = [image0,image1,image2,image3,image4,image5,image6,image7,image8,image9]

# Detect text in the image
for image in array:
    text_detected = text_detector(image, net)

# Display the image
    cv2.imshow("Text Detection", text_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
