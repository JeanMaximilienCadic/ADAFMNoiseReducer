from ADAFMNoiseReducer import ADAFMNoiseReducer
import cv2

if __name__ == "__main__":
    reducer = ADAFMNoiseReducer()
    img_path = "input.jpg"
    img = cv2.imread(img_path)
    img = reducer(img)
    cv2.imwrite("result.jpg", img)
    cv2.imshow("", img)
    cv2.waitKey()
