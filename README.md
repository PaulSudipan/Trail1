## Mobile Net Transfer Learning
## Architechture
The architect of this model is  that it uses MobileNet V2, which is considered a lightweight model. It was trained on the ImageNet dataset, and it can predict many objects,including humans.

### arch refined
We removed the last 2-3 layers of Mobilenet and added some layers of our own, tuned to predict 7 classes.

## prediction
We are dependent on OpenCV built-in face detection module: CascadeClassifer
```python
# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract face region
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        # Predict emotion
        preds = model.predict(face)
        emotion_label = emotions[np.argmax(preds)]
```

