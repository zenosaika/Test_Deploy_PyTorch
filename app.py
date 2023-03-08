from flask import Flask, jsonify, request

import io
import json
from PIL import Image
from torchvision.transforms import transforms
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval() # Inference Only Mode

imagenet_class_index = json.load(open('static/imagenet_class_index.json'))

@app.route('/')
def hello():
    return 'Hello Flask!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes)
    return jsonify({'class_id': class_id, 'class_name': class_name})

def transform_image(image_bytes):
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform_pipeline(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    max_element, max_idx = outputs.max(1)
    return imagenet_class_index[str(max_idx.item())]

if __name__ == '__main__':
    app.run()
