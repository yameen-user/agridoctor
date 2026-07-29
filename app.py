import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Load trained AI model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Class names
class_names = ["early_blight", "healthy", "leaf_spot"]


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        uploaded_image = request.files.get("image")

        if uploaded_image and uploaded_image.filename != "":

            # Open uploaded image using Pillow
            img = Image.open(uploaded_image.stream)

            # Convert image to RGB
            img = img.convert("RGB")

            # Resize image
            img = img.resize((224, 224))

            # Convert image to array
            img_array = np.array(img)

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Normalize image
            img_array = img_array / 255.0

            # AI prediction
            prediction = model.predict(img_array)

            predicted_index = np.argmax(prediction[0])

            disease = class_names[predicted_index]

            confidence = float(
                np.max(prediction[0]) * 100
            )

            recommendations = {
    "early_blight": "Remove infected leaves and use a suitable fungicide.",
    "healthy": "The plant is healthy. Continue regular watering and care.",
    "leaf_spot": "Remove affected leaves and use a suitable fungicide."
        }

            recommendation = recommendations[disease]

            result = (
            f"Disease: {disease.replace('_', ' ').title()} | "
            f"Confidence: {confidence:.2f}% | "
            f"Recommendation: {recommendation}"
            )

        else:
            result = "Please select an image."

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)