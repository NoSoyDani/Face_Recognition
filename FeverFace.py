
import face_recognition
import cv2
import numpy as np




video_capture = cv2.VideoCapture(0)
image = face_recognition.load_image_file("foto_personal.jpg")
face_encoding = face_recognition.face_encodings(image)[0]


known_face_encodings = [
    face_encoding,

]
known_face_names = [
    "Daniel",

]

scale=4;
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)

    rgb_small_frame = small_frame[:, :, ::-1]



    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame,model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "???"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        cv2.rectangle(frame, (left, top), (right, bottom), (251, 0, 255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (top,top), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
