import os.path

from flask import Flask, request
from flask_cors import cross_origin
from PIL import Image
from predict import Predict

app = Flask(__name__)
model_dir = './output'
model_name = 'swin_t.pth'
cls_model = Predict(model_dir, model_name, device='cpu')

@app.route('/')
@cross_origin()
def index():
    return app.send_static_file('index.html')


@app.route('/test')
@cross_origin()
def test():
    return 'test'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # model_type = request.args['modelType']
    code = 200
    msg = ''
    # if model_type == '':
    #     code = 201
    #     msg = '未选择预测模型'
    #     return {'code': code, 'msg': msg}
    file = request.files.get('file')
    if file is None:
        code = 202
        msg = '未上传预测图片'
        return {'code': code, 'msg': msg}
    # path = './predictImg'
    # img_name = file.filename
    # img_path = os.path.join(path, img_name)
    # file.save(img_path)
    # 读取图像
    image = Image.open(file)
    image = image.convert('RGB')
    msg = cls_model(image)
    print(msg)
    return {'code': code, 'msg': [msg]}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)
