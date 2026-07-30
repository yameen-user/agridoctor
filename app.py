import os
import base64
import io
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.tflite")

# Load TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["early_blight", "healthy", "leaf_spot"]

# Knowledge base for college demonstration & farmer advisory
DISEASE_INFO = {
    "early_blight": {
        "name": "Early Blight (Alternaria solani)",
        "severity": "Moderate to Severe",
        "badge_class": "badge-danger",
        "description": "Early Blight is a destructive fungal leaf disease that causes target-shaped dark brown spots on mature leaves, wilting, and reduced crop yield.",
        "symptoms": [
            "Dark brown to black spots with concentric rings (target pattern).",
            "Yellow halo surrounding leaf lesions.",
            "Lower/older leaves dry up and drop off early."
        ],
        "organic_remedy": [
            "Spray Neem Oil solution (5-10 ml per liter of water) every 7 days.",
            "Prune and safely discard affected lower leaves.",
            "Apply organic copper fungicide spray."
        ],
        "chemical_remedy": [
            "Apply Chlorothalonil or Mancozeb (2g per liter water).",
            "Spray Copper Oxychloride 50% WP during early morning."
        ],
        "prevention": [
            "Avoid overhead irrigation; water at soil level.",
            "Practice 3-year crop rotation with non-solanaceous crops.",
            "Apply straw mulch to prevent soil spores from splashing onto leaves."
        ]
    },
    "healthy": {
        "name": "Healthy Plant Leaf 🌿",
        "severity": "Optimal Health",
        "badge_class": "badge-success",
        "description": "No symptoms of fungal, bacterial, or viral disease detected. The leaf exhibits healthy chlorophyll levels and cell structure.",
        "symptoms": [
            "Vibrant green leaf blade and stem.",
            "No leaf spots, wilting, or necrosis."
        ],
        "organic_remedy": [
            "Continue applying vermicompost or organic liquid fertilizer monthly.",
            "Ensure 6-8 hours of direct sunlight daily."
        ],
        "chemical_remedy": [
            "No chemical pesticides or sprays required."
        ],
        "prevention": [
            "Maintain balanced N-P-K soil nutrients.",
            "Keep growing space weed-free and well-ventilated."
        ]
    },
    "leaf_spot": {
        "name": "Leaf Spot Disease (Septoria / Bacterial Spot)",
        "severity": "Moderate",
        "badge_class": "badge-warning",
        "description": "Leaf Spot causes small water-soaked lesions that turn into circular dark spots, weakening photosynthesis and leaf vitality.",
        "symptoms": [
            "Small circular dark grey/brown spots on foliage.",
            "Yellowing around leaf lesions.",
            "Leaves turn brittle and fall off under high humidity."
        ],
        "organic_remedy": [
            "Spray Baking Soda solution (1 tbsp baking soda + 1 tsp liquid soap per 4L water).",
            "Apply Bio-fungicide containing Bacillus subtilis."
        ],
        "chemical_remedy": [
            "Spray Copper-based fungicide or Zineb spray.",
            "Apply Dithane M-45 as per dosage instructions."
        ],
        "prevention": [
            "Ensure proper spacing between crops for adequate ventilation.",
            "Disinfect pruning shears with alcohol between uses.",
            "Use certified disease-resistant crop seeds."
        ]
    }
}

@app.route("/", methods=["GET", "POST"])
def home():
    data = None
    error_msg = None

    if request.method == "POST":
        uploaded_image = request.files.get("image")

        if uploaded_image and uploaded_image.filename != "":
            try:
                # Read image file
                raw_bytes = uploaded_image.read()
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                
                # Convert to base64 for direct preview on page
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Preprocess for model (224x224)
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Run inference
                interpreter.set_tensor(input_details[0]["index"], img_array)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]["index"])

                predicted_index = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0]) * 100)

                disease_key = class_names[predicted_index]
                info = DISEASE_INFO.get(disease_key, DISEASE_INFO["healthy"])

                data = {
                    "disease_key": disease_key,
                    "confidence": round(confidence, 2),
                    "image_b64": img_base64,
                    "info": info,
                    "all_scores": [
                        {"name": DISEASE_INFO[key]["name"], "score": round(float(prediction[0][i]) * 100, 1)}
                        for i, key in enumerate(class_names)
                    ]
                }

            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
        else:
            error_msg = "Please select a valid plant leaf image."

    return render_template(
        "index.html",
        data=data,
        error_msg=error_msg
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"AgriDoctor AI server running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)