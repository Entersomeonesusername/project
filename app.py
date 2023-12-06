from flask import Flask, render_template, request, send_file
from io import BytesIO
import cv2
import torch
import numpy as np
import RRDBNet_arch as arch
import base64

app = Flask(__name__)

model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

@app.route('/')
def index():
    return render_template('index.html', processed_image=None)

@app.route('/process_image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Read the input image
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        # Convert the processed image to a base64-encoded string
        _, img_encoded = cv2.imencode('.jpg', output)
        processed_image = base64.b64encode(img_encoded).decode('utf-8')

        return render_template('p.html', processed_image=processed_image)

@app.route('/download_image', methods=['POST'])
def download_image():
    try:
        # Decode the base64-encoded image
        processed_image_data = request.form.get('processed_image', type=str)
        img_decoded = base64.b64decode(processed_image_data)

        # Create an in-memory file-like object
        output = BytesIO(img_decoded)

        # Send the file for download
        return send_file(output, as_attachment=True, download_name='processed_image.jpg')

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {str(e)}")
        return str(e)

if __name__ == '__main__':
    app.add_url_rule('/download_image', 'download_image', download_image)
    app.run(debug=True)
