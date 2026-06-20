import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# Custom Preprocessing class definition matching training pipeline
class RetinalPreprocessing(object):
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        resized = cv2.resize(img_np, (224, 224))
        green = resized[:, :, 1]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(green)
        
        blurred = cv2.GaussianBlur(clahe_img, (5,5), 0)
        processed_3ch = cv2.merge([blurred, blurred, blurred])
        return Image.fromarray(processed_3ch)

gradio_transform = transforms.Compose([
    RetinalPreprocessing(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes

# Load weights locally (Hugging Face space runs on CPU by default)
MODEL_FILE = "dr_classifier_bias_reduced.pth"
if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()
    print("Model Loaded Successfully ✅")
else:
    print(f"Error: Model file {MODEL_FILE} not found!")

classes = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']

precautions_map = {
    "Healthy": [
        "Maintain healthy lifestyle and regular checkups.",
        "Have annual eye checkups.",
        "Follow a balanced diet and exercise regularly."
    ],
    "Mild DR": [
        "Control blood sugar strictly.",
        "Visit an eye specialist every 6-12 months.",
        "Control blood pressure and cholesterol."
    ],
    "Moderate DR": [
        "Frequent retinal examinations (every 3-6 months).",
        "Strict diabetes control required.",
        "Avoid high sugar and processed foods."
    ],
    "Proliferate DR": [
        "Urgent specialist consultation required.",
        "Possible laser or injection treatment.",
        "Strict diabetes management is critical."
    ],
    "Severe DR": [
        "Immediate medical attention required.",
        "Possible laser treatment evaluation.",
        "Do not ignore vision changes."
    ]
}

def predict(image):
    if image is None:
        return "Please upload an image first", "", {}
        
    try:
        image = image.convert("RGB")
        x = gradio_transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(output, 1).item()

        stage = classes[pred_idx]
        precaution_list = precautions_map[stage]
        precaution_text = "\n".join([f"• {p}" for p in precaution_list])
        
        # Format confidence percentages for gr.Label
        confidences = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        return stage, precaution_text, confidences

    except Exception as e:
        return f"Error: {str(e)}", "Check model / input", {}

# Custom CSS for high-quality dark mode and glassmorphism styling
custom_css = """
body {
    background-color: #0b0f19 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
.header-container {
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 2.5rem;
    backdrop-filter: blur(10px);
}
.header-container h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    font-weight: 800;
}
.header-container p {
    color: #94a3b8;
    font-size: 1.1rem;
}
.card-box {
    background-color: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(15px) !important;
    padding: 1.5rem !important;
}
.submit-btn {
    background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    cursor: pointer;
}
.submit-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
}
.clear-btn {
    background-color: rgba(148, 163, 184, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    cursor: pointer;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Group(elem_classes="header-container"):
        gr.Markdown("# Diabetic Retinopathy Detection System")
        gr.Markdown("Upload a retinal fundus image or choose from the categorized sample images below to test the diagnostic system.")
        
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card-box"):
                gr.Markdown("### Input Image")
                input_image = gr.Image(type="pil", label="Retinal Fundus Image")
                with gr.Row():
                    clear_btn = gr.Button("Clear", elem_classes="clear-btn")
                    submit_btn = gr.Button("Analyze Image", elem_classes="submit-btn")
                    
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card-box"):
                gr.Markdown("### Diagnostic Result")
                output_stage = gr.Textbox(label="Predicted DR Severity Stage", interactive=False)
                output_precautions = gr.Textbox(label="Recommended Clinical Precautions", interactive=False, lines=4)
                output_confidences = gr.Label(label="Classification Confidence", num_top_classes=5)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Sample Images to Try")
            with gr.Tabs():
                with gr.Tab("Healthy"):
                    gr.Examples(
                        examples=[
                            ["samples/Healthy/Healthy_sample_1_dr.png"],
                            ["samples/Healthy/Healthy_sample_2_dr.png"],
                            ["samples/Healthy/Healthy_sample_3_dr.png"],
                            ["samples/Healthy/Healthy_sample_4_dr.png"],
                            ["samples/Healthy/Healthy_sample_5_dr.png"]
                        ],
                        inputs=input_image,
                        outputs=[output_stage, output_precautions, output_confidences],
                        fn=predict,
                        cache_examples=False
                    )
                with gr.Tab("Mild DR"):
                    gr.Examples(
                        examples=[
                            ["samples/Mild_DR/Mild_DR_sample_1_dr.png"],
                            ["samples/Mild_DR/Mild_DR_sample_2_dr.png"],
                            ["samples/Mild_DR/Mild_DR_sample_3_dr.png"],
                            ["samples/Mild_DR/Mild_DR_sample_4_dr.png"],
                            ["samples/Mild_DR/Mild_DR_sample_5_dr.png"]
                        ],
                        inputs=input_image,
                        outputs=[output_stage, output_precautions, output_confidences],
                        fn=predict,
                        cache_examples=False
                    )
                with gr.Tab("Moderate DR"):
                    gr.Examples(
                        examples=[
                            ["samples/Moderate_DR/Moderate_DR_sample_1_dr.png"],
                            ["samples/Moderate_DR/Moderate_DR_sample_2_dr.png"],
                            ["samples/Moderate_DR/Moderate_DR_sample_3_dr.png"],
                            ["samples/Moderate_DR/Moderate_DR_sample_4_dr.png"],
                            ["samples/Moderate_DR/Moderate_DR_sample_5_dr.png"]
                        ],
                        inputs=input_image,
                        outputs=[output_stage, output_precautions, output_confidences],
                        fn=predict,
                        cache_examples=False
                    )
                with gr.Tab("Severe DR"):
                    gr.Examples(
                        examples=[
                            ["samples/Severe_DR/Severe_DR_sample_1_dr.png"],
                            ["samples/Severe_DR/Severe_DR_sample_2_dr.png"],
                            ["samples/Severe_DR/Severe_DR_sample_3_dr.png"],
                            ["samples/Severe_DR/Severe_DR_sample_4_dr.png"],
                            ["samples/Severe_DR/Severe_DR_sample_5_dr.png"]
                        ],
                        inputs=input_image,
                        outputs=[output_stage, output_precautions, output_confidences],
                        fn=predict,
                        cache_examples=False
                    )
                with gr.Tab("Proliferate DR"):
                    gr.Examples(
                        examples=[
                            ["samples/Proliferate_DR/Proliferate_DR_sample_1_dr.png"],
                            ["samples/Proliferate_DR/Proliferate_DR_sample_2_dr.png"],
                            ["samples/Proliferate_DR/Proliferate_DR_sample_3_dr.png"],
                            ["samples/Proliferate_DR/Proliferate_DR_sample_4_dr.png"],
                            ["samples/Proliferate_DR/Proliferate_DR_sample_5_dr.png"]
                        ],
                        inputs=input_image,
                        outputs=[output_stage, output_precautions, output_confidences],
                        fn=predict,
                        cache_examples=False
                    )
            
    submit_btn.click(fn=predict, inputs=input_image, outputs=[output_stage, output_precautions, output_confidences])
    clear_btn.click(fn=lambda: (None, "", {}), inputs=None, outputs=[input_image, output_stage, output_precautions, output_confidences])

if __name__ == "__main__":
    demo.launch()
