# 
## Homework 10

# In this homework, you are going to use and compare two different trackers (of your liking) and compare the results.

# ### Step 1
# Decide what video you are going to use for this homework, select an object and generate the template. You can use any video you want (your own, from Youtube, etc.)
# and track any object you want (e.g. a car, a pedestrian, etc.).

# %%

import os
import cv2
from matplotlib import pyplot as plt

# Load the video

video_name = 'video1t.mp4'

base_dir = 'C:/users/admin/Documents/math_for_DS/Machine Vision/Computer-Vision-v2/lesson_10/homework/'


video_dir = os.path.join(base_dir, 'video')
frame_dir = os.path.join(base_dir, 'frames')
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

video_path = os.path.join(video_dir, video_name)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, img = cap.read()
if not ok:
   print("End of video.")
   
#resize because it is too large   
#img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
 
#bbox = cv2.selectROI('Selector',img,False)
#print(bbox)
#cv2.destroyAllWindows()
bbox=(312, 118, 132, 151)
#print(bbox)
x1, y1 = bbox[0], bbox[1]
width, height = bbox[2], bbox[3]

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
plt.imshow(img)

#cv2.destroyAllWindows()

# %%

# ### Step 2
# Initialize a tracker (e.g. KCF).

# I borrowed the below code from lecture reference but limited to supported by my version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
tracker_types = [ 'MIL','KCF',  'CSRT']
tracker_type = tracker_types[1]
 
if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

tracker_frame_dir = os.path.join(frame_dir, tracker_type)
if not os.path.exists(tracker_frame_dir):
    os.makedirs(tracker_frame_dir)


# ### Step 3
# Run the tracker on the video and the selected object. Run the tracker for around 10-15 frames.
# %%
ok = tracker.init(img, bbox)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
frameNo=0

# ### Step 4
# For each frame, print the bounding box on the image and save it.

# ### Step 5
# Select a different tracker (e.g. CSRT) and repeat steps 2, 3 and 4.

while frameNo < 15:
    # Read a new frame
    ok, img = cap.read()
    if not ok:
        print(f"Opening of frame #{frameNo} failed")
        break
    
#    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    # Update tracker
    ok, bbox = tracker.update(img)
    failed=''
    if not ok:
        print(f"Tracking for frame #{frameNo} failed")
        failed='FAILED !'
        #break
    

    
    # Draw bounding box if tracking is successful
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (0, 255, 0), 5)

    # Display tracker type on frame
    cv2.putText(img, tracker_type + " Tracker " + failed, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 7)

    # Show live tracking in OpenCV window
    cv2.imshow("Tracking", img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    frame_filename = f"{video_name}_{tracker_type}_frame_{frameNo:03d}.jpg"
    cv2.imwrite(os.path.join(tracker_frame_dir, frame_filename), img)
    plt.axis('off')
    plt.show()
    plt.draw()

    # Exit if ESC is pressed
    if cv2.waitKey(10) & 0xFF == 27:  # 10 ms delay for smooth playback
       break

    frameNo += 1

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
# %%



# ### Step 6
# Compare the results:
# * Do you see any differences? If so, what are they?

# Yes, KSF tracker works faster but fails if template get distorted, the CSRT works even in case of minor jbject rotation or perspective distortion.

# * Does one tracker perform better than the other? In what way?

# All trackers works pretty good in general MV application in machinery with stable ROI and small perspecpective distortion. 
# The KSF seems the fastest one but less robust than others and fails quickly if object angle changed.
# But tt was challenging to find a machinery video where KSF fails.
# However the processing speed is the key factor in robotics and machinery MV inspection applications
# and KSF looks pretty good for such applications
# CSRT is amazingly robust  but works relatively slow
# MIL tracker should be invariant to background changes but it is not so robust as CSRT for this video with about the same speed

