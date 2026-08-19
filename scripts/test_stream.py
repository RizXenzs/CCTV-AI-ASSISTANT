import cv2

url = "https://stream.palembang.go.id/cam28/main_stream.m3u8"
cap = cv2.VideoCapture(url)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("CCTV Palembang Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Mencoba me-connect ulang stream...")
        cap.open(url)

cap.release()
cv2.destroyAllWindows()