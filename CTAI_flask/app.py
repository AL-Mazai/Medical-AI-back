import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
import torch
from flask import *
import core.main
import base64
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename

from CTAI_flask.core.net.model import EffB0_UNet
from CTAI_model.net import unet as net
from hjd.doctor import doctor
from hjd.AI import AI
UPLOAD_FOLDER = r'uploads'


ALLOWED_EXTENSIONS = set(['dcm'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.register_blueprint(doctor)
app.register_blueprint(AI)
werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)
# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

######################################erunet#################################################
model = EffB0_UNet()
PATH = 'core/ERU-Net_liver8_new.pth'
model.load_state_dict(torch.load(PATH))
model.eval()
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])


def process_single_image(image_path, model, transform):
    input_image = Image.open(image_path)
    input_tensor = transform(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    output_array = output.squeeze().numpy()

    return output_array

# def init_model():
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # model = net.Unet(1, 1).to(device)
#     model = EffB0_UNet()
#     path = 'core/ERU-Net_liver8_new.pth'
#     if torch.cuda.is_available():
#         model.load_state_dict(torch.load(path))
#     else:
#         model.load_state_dict(torch.load(path, map_location='cpu'))
#     model.eval()
#     return model

@app.route('/image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        try:
            output_image_array = process_single_image(file_path, model, transform_img)
            output_image = Image.fromarray((output_image_array * 255).astype('uint8'))
            output_file_path = os.path.join("outputs", "processed_" + filename)
            output_image.save(output_file_path)

            # 可以选择将处理后的图片作为文件发送回去，或者将其保存在服务器上并提供下载链接
            # 这里仅提供下载链接的逻辑
            return jsonify({"message": "Image processed successfully",
                            "download_url": "/download_erunet/" + "processed_" + filename}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file"}), 400


@app.route('/download_erunet/<filename>')
def download_file_erunet(filename):
    return send_from_directory("outputs", filename, as_attachment=True)
######################################################################################

# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        # 给文件名加时间戳，防止重名
        timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename, extension = os.path.splitext(file.filename)
        new_filename = f"{filename}_{timestamp_suffix}{extension}"
        # 保存文件
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(src_path)
        shutil.copy(src_path, 'tmp/ct')
        image_path = os.path.join('tmp/ct', new_filename)
        pid, image_info = core.main.c_main(image_path, current_app.model)

        # 将识别出来的文件转换为base64编码
        with open(f'tmp/image/{pid}', "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        with open(f'tmp/draw/{pid}', "rb") as draw_file:
            draw_base64 = base64.b64encode(draw_file.read()).decode("utf-8")

        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:5003/tmp/image/' + pid,
                        'draw_url': 'http://127.0.0.1:5003/tmp/draw/' + pid,
                        'image_base64' : 'data:image/png;base64,'+image_base64,
                        'draw_base64' : 'data:image/png;base64,'+draw_base64,
                        'image_info': image_info
                        })

    return jsonify({'status': 0})


@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data_test', 'testfile.zip', as_attachment=True)


# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    # print(file)
    if request.method == 'GET':
        if file is None:
            pass
        else:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass




if __name__ == '__main__':
    with app.app_context():
        current_app.model = model
    app.run(host='10.103.205.72', port=5003, debug=True)
