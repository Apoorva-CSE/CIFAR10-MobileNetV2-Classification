import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import os

# Define CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(image_path, model_path='cifar10_mobilenetv2.keras'):
    """Loads a saved model and predicts the class of an image."""

    # 1. Load the saved model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error loading model. Did you run the training script? Details: {e}")
        return

    # 2. Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((32, 32))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 32, 32, 3)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return

    # 3. Make the prediction
    predictions = model.predict(img_array, verbose=0)

    # 4. Interpret the results
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = class_names[predicted_idx]

    print(f"\nImage: {image_path}")
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print("\nAll class probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
        bar = '█' * int(prob * 40)
        print(f"  {name:12s}: {prob*100:5.2f}%  {bar}")

    return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classifier")
    parser.add_argument("--image_path", type=str, default="OIP.jpg",
                        help="Path to the image file to classify")
    parser.add_argument("--model_path", type=str, default="cifar10_mobilenetv2.keras",
                        help="Path to the saved Keras model")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at '{args.image_path}'")
    else:
        predict_image(args.image_path, args.model_path)
