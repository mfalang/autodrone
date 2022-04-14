# File to help with labeling the ground truth coordinates of the arrow and the corners of the H in
# the helipad

import sys
import glob
import cv2 as cv
import numpy as np

def click_event(event, x, y, flags, param):

    img = param[0]
    filename = param[1]

    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        idx = param[2]["current_idx"]
        if idx >= 13:
            print("13 points already added. Not adding more.")
            return

        center = (x,y)
        param[2]["coords"][idx,:] = center
        param[2]["current_idx"] += 1

        color = (0,0,255) # red

        cv.circle(img, center, 4, color, cv.FILLED)

        text_face = cv.FONT_HERSHEY_DUPLEX
        text_scale = 0.3
        text_thickness = 1
        text = f"{center}"
        text_offset = 10

        text_size, _ = cv.getTextSize(text, text_face, text_scale, text_thickness)
        text_origin = (
            int(center[0] - text_size[0] / 2),
            int(center[1] + text_size[1] / 2) - text_offset
        )

        cv.circle(img, center, 4, color, cv.FILLED)
        cv.putText(img, text, text_origin, text_face, text_scale, (127,255,127), text_thickness, cv.LINE_AA)

        cv.imshow("image", img)

# Load images
images = [(cv.imread(file), file) for file in sorted(glob.glob("test_images/real/*.jpg"))]

start_image = 0
print(f"Starting from image {start_image}: {images[start_image][1]}")

for (img, filename) in images[start_image:]:
    # img, filename = images[i]
    header = f"{'='*10} Labeling image: {filename} {'='*10}"
    print(header)

    ans = "r"

    img_with_corners = img.copy()

    labels = {"coords": np.zeros((13,2)), "current_idx": 0}

    while ans != "":
        cv.imshow("image", img_with_corners)
        cv.setMouseCallback("image", click_event, param=(img_with_corners, filename, labels))
        cv.waitKey(0)

        ans = input("Save labels and go to next image [Enter], retry [r] or quit program [q]? ")
        if ans == "r":
            print("Resetting labels on current image")
            img_with_corners = img.copy() # clear labels from image
            labels = {"coords": np.zeros((13,2)), "current_idx": 0} # clear coords
        elif ans == "q":
            cv.destroyAllWindows()
            sys.exit(0)
        else:
            print(f"Labelling complete for image {filename}")
            gt_labels_filename = f"{filename[:-4]}_gt_labels.txt" # save in same folder as images
            print(f"Saving labels to: {gt_labels_filename}")
            np.savetxt(gt_labels_filename, labels["coords"])
            print("="*len(header))

cv.destroyAllWindows()