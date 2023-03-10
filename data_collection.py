import cv2

# open video capture device
cap = cv2.VideoCapture("/dev/video2")

# set the frame size to 96x96
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 96)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)

frame_num = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, [96,96])
    if not ret:
        break

    # increment the frame number
    frame_num += 1

    filename = f"Data/New/1/frame_{frame_num:04d}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")
    # display the frame in a window
    cv2.imshow('frame', frame)

    # wait for a key press and check if the 'q' key was pressed
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break