from flask import Flask, render_template
from flask import request, jsonify
from werkzeug.utils import secure_filename

from torch_utils import get_prediction, preprocess_upload

app = Flask(__name__)

Class = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # 파일 확장자 검사
    ext = filename.split('.')[-1]
    if ext in ALLOWED_EXTENSIONS:
        return True
    else:
        return False

# html 렌더링
@app.route('/upload')
def render_html():
    return render_template('upload.html')

# 업로드한 파일 처리
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # error check
    if request.method == 'POST':
        f = request.files['file']
        f_savepath = r"C:\Users\labadmin\Desktop\app\app\static\inference_images" + secure_filename(f.filename)
        f.save(f_savepath)

        if f is None or f.filename == "":
            return 'no file'
        if not allowed_file(f.filename):
            return 'format not supported'

        tensor = preprocess_upload(f)
        prediction = get_prediction(tensor)
        pred = prediction.item()
        label =Class.get(pred)
        
        return render_template('inference_result.html', file_path=f_savepath,
                               prediction_value=label)



if __name__ == '__main__':
    app.run(debug=True)

# set FLASK_APP=upload.py # windos에서. 그외는 export
# set FLASK_ENV=development # windos에서. 그외는 export
# flask run # 실행