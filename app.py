from flask import Flask, render_template, request

import os
import io
import base64

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image, ImageOps


# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)

# Maximum uploaded image size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==========================================
# CPU Optimization
# ==========================================

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# Tumor Class Names
# نفس ترتيب تدريب المودل الأساسي
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
# Paths
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TUMOR_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_brain_model.pth"
)

VALIDATOR_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_mri_validator.pth"
)


# ==========================================
# MRI Validator Settings
# ==========================================

MRI_THRESHOLD = 0.80


# ==========================================
# Tumor Model
# ResNet50 + Dropout + Linear
# ==========================================

tumor_model = models.resnet50(
    weights=None
)

num_features = tumor_model.fc.in_features

tumor_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 4)
)


tumor_checkpoint = torch.load(
    TUMOR_MODEL_PATH,
    map_location=device,
    weights_only=True
)

tumor_model.load_state_dict(
    tumor_checkpoint
)

del tumor_checkpoint

tumor_model = tumor_model.to(device)

tumor_model.eval()

print("Brain Tumor model loaded successfully!")


# ==========================================
# MRI Validator Model
# MobileNetV3 Small
# ==========================================

validator_model = models.mobilenet_v3_small(
    weights=None
)

validator_in_features = (
    validator_model.classifier[3].in_features
)

validator_model.classifier[3] = nn.Linear(
    validator_in_features,
    2
)


validator_checkpoint = torch.load(
    VALIDATOR_MODEL_PATH,
    map_location=device,
    weights_only=True
)

validator_model.load_state_dict(
    validator_checkpoint
)

del validator_checkpoint

validator_model = validator_model.to(device)

validator_model.eval()

print("MRI Validator model loaded successfully!")


# ==========================================
# Tumor Model Temperature Calibration
# ==========================================

temperature = 1.1327744722366333


# ==========================================
# Tumor Model Transform
# ==========================================

tumor_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# Validator Transform
# نفس test_transform المستخدم أثناء التدريب
# ==========================================

validator_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# Prepare Uploaded Image
# ==========================================

def prepare_uploaded_image(image):

    # Fix orientation from EXIF metadata
    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert("RGB")

    # Prevent extremely large images from
    # consuming too much RAM.
    max_dimension = 2000

    if (
        image.width > max_dimension
        or image.height > max_dimension
    ):

        image.thumbnail(
            (
                max_dimension,
                max_dimension
            ),
            Image.Resampling.LANCZOS
        )

    return image


# ==========================================
# MRI Validation Function
# ==========================================

def validate_brain_mri(image):

    image = image.convert("RGB")

    image_tensor = validator_transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():

        outputs = validator_model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Class index 1 = Brain MRI
        brain_probability = (
            probabilities[0, 1].item()
        )

    is_valid_brain_mri = (
        brain_probability >= MRI_THRESHOLD
    )

    del image_tensor
    del outputs
    del probabilities

    return (
        is_valid_brain_mri,
        brain_probability
    )


# ==========================================
# Tumor Prediction Function
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = tumor_transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():

        outputs = tumor_model(
            image_tensor
        )

        # Temperature Scaling
        calibrated_outputs = (
            outputs / temperature
        )

        probabilities = torch.softmax(
            calibrated_outputs,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        probabilities[predicted_index].item()
        * 100
    )

    all_probabilities = {}

    for i in range(
        len(class_names)
    ):

        display_name = display_names[
            class_names[i]
        ]

        probability = (
            probabilities[i].item()
            * 100
        )

        all_probabilities[
            display_name
        ] = round(
            probability,
            2
        )

    del image_tensor
    del outputs
    del calibrated_outputs
    del probabilities

    return (
        predicted_class,
        confidence,
        all_probabilities
    )


# ==========================================
# Create Image Preview
# ==========================================

def create_image_preview(image):

    display_image = image.copy()

    # Keep preview lightweight for the web
    display_image.thumbnail(
        (800, 800),
        Image.Resampling.LANCZOS
    )

    buffer = io.BytesIO()

    display_image.save(
        buffer,
        format="JPEG",
        quality=80,
        optimize=True
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

    if request.method == "POST":

        if "mri_image" not in request.files:

            error = (
                "Please select an MRI image."
            )

        else:

            file = request.files[
                "mri_image"
            ]

            if file.filename == "":

                error = (
                    "Please select an MRI image."
                )

            else:

                try:

                    # ==================================
                    # Open Uploaded Image
                    # ==================================

                    image = Image.open(
                        file.stream
                    )


                    # ==================================
                    # Fix Orientation + Limit Size
                    # ==================================

                    image = prepare_uploaded_image(
                        image
                    )


                    # ==================================
                    # Keep Image Visible
                    # ==================================

                    image_preview = (
                        create_image_preview(
                            image
                        )
                    )


                    # ==================================
                    # Step 1:
                    # Validate Brain MRI
                    # ==================================

                    (
                        is_valid_brain_mri,
                        brain_probability
                    ) = validate_brain_mri(
                        image
                    )

                    print(
                        "Brain MRI probability:",
                        round(
                            brain_probability * 100,
                            2
                        ),
                        "%"
                    )


                    # ==================================
                    # Reject Non-Brain Images
                    # ==================================

                    if not is_valid_brain_mri:

                        error = (
                            "The uploaded image does not appear "
                            "to be a valid brain MRI scan. "
                            "Please upload a clear brain MRI image."
                        )

                    else:

                        # ==============================
                        # Step 2:
                        # Tumor Classification
                        # ==============================

                        (
                            predicted_class,
                            confidence,
                            probabilities
                        ) = predict_image(
                            image
                        )


                        prediction = (
                            display_names[
                                predicted_class
                            ]
                        )


                        disease_info = (
                            class_information[
                                predicted_class
                            ]
                        )


                except Exception as e:

                    print(
                        "Prediction error:",
                        e
                    )

                    error = (
                        "Unable to process this image. "
                        "Please upload a valid MRI image."
                    )


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