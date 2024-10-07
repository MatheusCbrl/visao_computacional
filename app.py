from flask import Flask, render_template, Response
from  yolodetection  import get_frame #poses

app = Flask(__name__)

# Rota para a página principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para o stream de vídeo
@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
