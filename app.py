from flask import Flask, render_template, request

import os
import io
import base64

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image


# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)

# Maximum uploaded image size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==========================================
# Device
# ==========================================

device = torch.device("cpu")

print("Using device:", device)


# ==========================================
# Class Names
# نفس ترتيب التدريب
# ==========================================

class_names = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]


display_names = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor"
}


# ==========================================
# Information About Each Class
# ==========================================

class_information = {

    "glioma": {
        "title": "Glioma",
        "description": (
            "Glioma is a type of brain tumor that develops from glial cells, "
            "which support and protect nerve cells in the brain and nervous system. "
            "Gliomas can vary in location, size, and growth pattern. MRI imaging "
            "is commonly used to evaluate their appearance and characteristics."
        )
    },

    "meningioma": {
        "title": "Meningioma",
        "description": (
            "Meningioma is a tumor that develops from the meninges, the protective "
            "membranes surrounding the brain and spinal cord. Many meningiomas grow "
            "slowly, although their size and location may affect nearby brain structures. "
            "MRI scans are commonly used to evaluate their location and appearance."
        )
    },

    "pituitary": {
        "title": "Pituitary Tumor",
        "description": (
            "A pituitary tumor develops in the pituitary gland, a small gland located "
            "at the base of the brain that helps regulate hormones in the body. "
            "Depending on its size and characteristics, it may affect hormone production "
            "or nearby structures. MRI imaging is commonly used for evaluation."
        )
    },

    "notumor": {
        "title": "About This Classification",
        "description": (
            "The AI model classified this MRI image as No Tumor. This means it did not "
            "identify imaging patterns matching the three tumor categories it was trained "
            "to recognize: Glioma, Meningioma, or Pituitary tumor. This classification "
            "does not confirm the absence of disease and is not a medical diagnosis."
        )
    }

}


# ==========================================
# Temperature Calibration
# ==========================================

temperature = 1.1327744722366333


# ==========================================
# Model Architecture
# ResNet50 + Dropout + Linear
# ==========================================

model = models.resnet50(weights=None)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 4)
)


# ==========================================
# Model Path
# ==========================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_brain_model.pth"
)


# ==========================================
# Load Trained Model
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()

for param in model.parameters():
    param.requires_grad = False

print("Brain Tumor model loaded successfully!")


# ==========================================
# Image Preprocessing
# نفس test_transform المستخدم أثناء التدريب
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# Prediction Function
# ==========================================

def predict_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.inference_mode():

        outputs = model(image_tensor)

        # Temperature Scaling
        calibrated_outputs = outputs / temperature

        # Probabilities
        probabilities = torch.softmax(
            calibrated_outputs,
            dim=1
        )[0]

    # Best class
    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = class_names[
        predicted_index
    ]

    # Highest probability
    confidence = (
        probabilities[predicted_index].item()
        * 100
    )

    # All class probabilities
    all_probabilities = {}

    for i in range(len(class_names)):

        display_name = display_names[
            class_names[i]
        ]

        probability = (
            probabilities[i].item()
            * 100
        )

        all_probabilities[display_name] = round(
            probability,
            2
        )

    return (
        predicted_class,
        confidence,
        all_probabilities
    )


# ==========================================
# Create Image Preview
# حتى تبقى الصورة ظاهرة بعد Analyze
# ==========================================

def create_image_preview(image):

    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded_image
    )


# ==========================================
# Home Page
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    probabilities = None
    image_preview = None
    disease_info = None
    error = None

    # ======================================
    # User clicked Analyze MRI
    # ======================================

    if request.method == "POST":

        # Check if image exists
        if "mri_image" not in request.files:

            error = "Please select an MRI image."

        else:

            file = request.files["mri_image"]

            # Check filename
            if file.filename == "":

                error = "Please select an MRI image."

            else:

                try:

                    # ==================================
                    # Open Uploaded Image
                    # ==================================

                    image = Image.open(
                        file.stream
                    ).convert("RGB")

                    # ==================================
                    # Keep Image Visible
                    # ==================================

                    image_preview = create_image_preview(
                        image
                    )

                    # ==================================
                    # Prediction
                    # ==================================

                    (
                        predicted_class,
                        confidence,
                        probabilities
                    ) = predict_image(
                        image
                    )

                    # ==================================
                    # Display-Friendly Name
                    # ==================================

                    prediction = display_names[
                        predicted_class
                    ]

                    # ==================================
                    # Information About Result
                    # ==================================

                    disease_info = class_information[
                        predicted_class
                    ]

                except Exception as e:

                    print(
                        "Prediction error:",
                        e
                    )

                    error = (
                        "Unable to process this image. "
                        "Please upload a valid MRI image."
                    )

    # ======================================
    # Send Everything to index.html
    # ======================================

    return render_template(

        "index.html",

        prediction=prediction,

        confidence=confidence,

        probabilities=probabilities,

        image_preview=image_preview,

        disease_info=disease_info,

        error=error
    )


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )