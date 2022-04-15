import sys
import glob
import cv2 as cv
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# Create a function based on a CV2 Event (Left button click)
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing

    img = param[0]
    filename = param[1]

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        # we take note of where that mouse was located
        ix,iy = x,y
        cv.imshow("image", img)

    elif event == cv.EVENT_MOUSEMOVE:
        drawing == True

    elif event == cv.EVENT_LBUTTONUP:
        radius = int(np.math.sqrt( ((ix-x)**2)+((iy-y)**2)))
        param[2][0] = x
        param[2][1] = y
        param[2][2] = radius
        img_with_circle = img.copy()
        cv.circle(img_with_circle,(ix,iy),radius, (127,255,127), thickness=2)
        cv.imshow("image", img_with_circle)
        drawing = False

# Load images
images = [(cv.imread(file), file) for file in sorted(glob.glob("test_images/real/*.jpg"))]

start_image = 0
print(f"Starting from image {start_image}: {images[start_image][1]}")

for (img, filename) in images[start_image:]:
    header = f"{'='*10} Labeling image: {filename} {'='*10}"
    print(header)

    ans = "r"

    label = np.zeros(3) # format: [x,y,r]

    while ans != "":
        cv.imshow("image", img)
        cv.setMouseCallback("image", draw_circle, param=(img, filename, label))
        cv.waitKey(0)

        ans = input("Save labels and go to next image [Enter], retry [r] or quit program [q]? ")
        if ans == "r":
            print("Resetting labels on current image")
            label = np.zeros(3) # clear coords
        elif ans == "q":
            cv.destroyAllWindows()
            sys.exit(0)
        else:
            if np.count_nonzero(label) == 0:
                print("No circle drawn, add one.")
                ans = "r"
                continue
            print(f"Labelling complete for image {filename}")
            print(label)
            gt_labels_filename = f"{filename[:-4]}_gt_circle.txt" # save in same folder as images
            print(f"Saving labels to: {gt_labels_filename}")
            # TODO: Fix this saving
            # np.savetxt(gt_labels_filename, label["center"])
            print("="*len(header))

cv.destroyAllWindows()
