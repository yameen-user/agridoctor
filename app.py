import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(
    model_path="plant_disease_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names
class_names = [
    "early_blight",
    "healthy",
    "leaf_spot"
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        uploaded_image = request.files.get("image")

        if uploaded_image and uploaded_image.filename != "":

            # Open image
            img = Image.open(uploaded_image.stream)

            # Convert to RGB
            img = img.convert("RGB")

            # Resize image
            img = img.resize((224, 224))

            # Convert image to NumPy array
            img_array = np.array(
                img,
                dtype=np.float32
            )

            # Normalize image
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Give image to AI model
            interpreter.set_tensor(
                input_details[0]["index"],
                img_array
            )

            # Run prediction
            interpreter.invoke()

            # Get prediction
            prediction = interpreter.get_tensor(
                output_details[0]["index"]
            )

            predicted_index = int(
                np.argmax(prediction[0])
            )

            disease = class_names[
                predicted_index
            ]

            confidence = float(
                np.max(prediction[0]) * 100
            )

            recommendations = {
                "early_blight":
                    "Remove infected leaves and use a suitable fungicide.",

                "healthy":
                    "The plant is healthy. Continue regular watering and care.",

                "leaf_spot":
                    "Remove affected leaves and use a suitable fungicide."
            }

            recommendation = recommendations[
                disease
            ]

            result = (
                f"Disease: "
                f"{disease.replace('_', ' ').title()} | "
                f"Confidence: "
                f"{confidence:.2f}% | "
                f"Recommendation: "
                f"{recommendation}"
            )

        else:
            result = "Please select an image."

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )