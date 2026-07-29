import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# Model ka exact path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.tflite")

# Load TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Classes
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
            img = Image.open(uploaded_image).convert("RGB")
            img = img.resize((224, 224))

            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            interpreter.set_tensor(
                input_details[0]["index"],
                img_array
            )

            interpreter.invoke()

            prediction = interpreter.get_tensor(
                output_details[0]["index"]
            )

            predicted_index = int(
                np.argmax(prediction[0])
            )

            confidence = float(
                np.max(prediction[0]) * 100
            )

            disease = class_names[predicted_index]

            result = (
                f"Disease: {disease.replace('_', ' ').title()} | "
                f"Confidence: {confidence:.2f}%"
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
        port=int(os.environ.get("PORT", 5000))
    )