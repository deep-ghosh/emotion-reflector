def resize_image(image, target_size=(48, 48)):
    return cv2.resize(image, target_size)

def normalize_image(image):
    return image / 255.0

def preprocess_image(image):
    image = resize_image(image)
    image = normalize_image(image)
    image = image.reshape(1, 48, 48, 1)  # Reshape for model input
    return image