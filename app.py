import os
import math
import cv2
from flask import Flask, request, render_template, url_for, redirect, flash
from werkzeug.utils import secure_filename
from soot_foil_image_tool import measure_image

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Define folder paths for uploads and outputs
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create folders if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Validate file presence
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    # Get additional parameters from the form
    dimension = request.form.get('dimension', 'h')  # default to 'h'
    size = request.form.get('size', '10.0')
    step = request.form.get('step', '10')
    
    try:
        size = float(size)
        step = int(step)
    except ValueError:
        flash("Invalid size or step value. Size must be a number and step an integer.")
        return redirect(url_for('index'))
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    # down‑scaling so Render doesn’t run out of RAM/CPU
    img = cv2.imread(upload_path)
    h, w = img.shape[:2]

    MAX_PIXELS      = 100000   
    MIN_SHORT_EDGE  = 250       

    current_pixels  = h * w
    scale_pix  = math.sqrt(MAX_PIXELS / current_pixels) if current_pixels > MAX_PIXELS else 1.0
    scale_edge = (MIN_SHORT_EDGE / min(h, w)) if min(h, w) > MIN_SHORT_EDGE else 1.0

    # choose the smaller factor – guarantees pixel cap, keeps as much detail as possible
    scale = min(scale_pix, scale_edge)

    if scale < 1.0:                         
        new_w = int(w * scale)
        new_h = int(h * scale)
        img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(upload_path, img)

    
    # Get the absolute path required by measure_image
    absolute_path = os.path.abspath(upload_path)
    
    try:
        # Call measure_image using the user-provided parameters.
        # The function signature is: measure_image(image_path, dimension, size, step=10, debug=0, ...)
        annotated_image, measurements = measure_image(absolute_path, dimension, size, step=step)
    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(url_for('index'))
    
    # Save the annotated image to the outputs folder
    annotated_filename = 'annotated_' + filename
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], annotated_filename)
    cv2.imwrite(output_path, annotated_image)
    
    # Generate a URL for the processed (annotated) image
    processed_image_url = url_for('static', filename=f'outputs/{annotated_filename}')
    contours_url = url_for('static', filename='outputs/contours.png')
    measurements_image_url = url_for('static', filename='outputs/Measurements.png')
    plot_url = url_for('static', filename='outputs/output_plot_1.png')
    csv_url = url_for('static', filename='outputs/measurements.csv')
    
    # Render the template and pass all output URLs and measurements
    return render_template('index.html', 
                           processed_image_url=processed_image_url, 
                           contours_url=contours_url,
                           measurements_image_url=measurements_image_url,
                           plot_url=plot_url,
                           measurements=measurements,
                           csv_url=csv_url)

if __name__ == '__main__':
    app.run(debug=True)