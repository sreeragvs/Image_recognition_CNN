import cv2
import numpy as np
from tensorflow.keras import models

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load trained model
model = models.load_model("image_classifier.keras")

# Load any input image (original size)
img = cv2.imread('frog.jpg')  # original image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR → RGB for consistency

# Resize to CIFAR-10 size (32x32) → for model prediction
img_resized = cv2.resize(img_rgb, (32, 32))
img_array = np.expand_dims(img_resized, axis=0) / 255.0  # (1, 32, 32, 3)

# Prediction
prediction = model.predict(img_array)
index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("Predicted class index:", index)
print("Predicted class name:", class_names[index])
print(f"Confidence: {confidence:.2f}%")

# Overlay prediction text on the ORIGINAL image (not resized)
output_img = cv2.putText(img.copy(),
                         f"{class_names[index]} ({confidence:.1f}%)",
                         (20, 40),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1.2, (0, 255, 0), 3, cv2.LINE_AA)

# Show result
cv2.imshow("Result", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
