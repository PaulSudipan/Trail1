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
## version 2
In version two, we didn't freeze all the base MobileNet layers, but we let them be tunable.

### stopping criterion
If the validation loss does not decrease per 3 epochs, we scale down the learning rate by 0.5, so in this case my learning rate diminished 2 zero at 12th epoch.
I had a good training accuracy, but the validation is a mess.

## Training results version two
```cmd
Almost 4 hours of training
718/718 ━━━━━━━━━━━━━━━━━━━━ 2245s 3s/step - accuracy: 0.4207 - loss: 1.5160 - val_accuracy: 0.4024 - val_loss: 4.6760 - learning_rate: 0.0010
Epoch 2/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1708s 2s/step - accuracy: 0.5682 - loss: 1.1740 - val_accuracy: 0.3447 - val_loss: 2.5368 - learning_rate: 0.0010
Epoch 3/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1721s 2s/step - accuracy: 0.6081 - loss: 1.0879 - val_accuracy: 0.2416 - val_loss: 5.3716 - learning_rate: 0.0010
Epoch 4/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1712s 2s/step - accuracy: 0.6152 - loss: 1.0426 - val_accuracy: 0.3534 - val_loss: 2.8870 - learning_rate: 0.0010
Epoch 5/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1719s 2s/step - accuracy: 0.6569 - loss: 0.9332 - val_accuracy: 0.5245 - val_loss: 1.5071 - learning_rate: 5.0000e-04
Epoch 6/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1717s 2s/step - accuracy: 0.6837 - loss: 0.8599 - val_accuracy: 0.5518 - val_loss: 1.3953 - learning_rate: 5.0000e-04
Epoch 7/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1717s 2s/step - accuracy: 0.7034 - loss: 0.8058 - val_accuracy: 0.6088 - val_loss: 1.1472 - learning_rate: 5.0000e-04
Epoch 8/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1709s 2s/step - accuracy: 0.7223 - loss: 0.7705 - val_accuracy: 0.6123 - val_loss: 1.1347 - learning_rate: 5.0000e-04
Epoch 9/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1711s 2s/step - accuracy: 0.7361 - loss: 0.7284 - val_accuracy: 0.6102 - val_loss: 1.1925 - learning_rate: 5.0000e-04
Epoch 10/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1722s 2s/step - accuracy: 0.7548 - loss: 0.6753 - val_accuracy: 0.6325 - val_loss: 1.1186 - learning_rate: 5.0000e-04
Epoch 11/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1702s 2s/step - accuracy: 0.7661 - loss: 0.6437 - val_accuracy: 0.5816 - val_loss: 1.3820 - learning_rate: 5.0000e-04
Epoch 12/20
718/718 ━━━━━━━━━━━━━━━━━━━━ 1706s 2s/step - accuracy: 0.7870 - loss: 0.5944 - val_accuracy: 0.6211 - val_loss: 1.2341 - learning_rate: 5.0000e-04
Epoch 13/20
 67/718 ━━━━━━━━━━━━━━━━━━━━ 25:36 2s/step - accuracy: 0.8312 - loss: 0.4757
```
