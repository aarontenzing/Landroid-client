import cv2 as cv

if __name__ == "__main__":
    # capture video from camera
    cam = cv.VideoCapture(9)

    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()
        cv.imwrite("frame.jpg", frame)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if cv.waitKey(1) == ord("q"):
            break

    cam.release()
    out.release()
    cv.destroyAllWindows()


