from flask import request, jsonify, Flask
import numpy as np
import cv2
from scoreCounter import countScore

app = Flask(__name__)

@app.route("/scorecounter", methods=["POST"])
def scoreCounter():
    if 'picture' not in request.files:
        return jsonify({"error": "No picture"}), 400

    file = request.files['picture']

    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Cant open the picture"}), 400

    scores = countScore(img)

    return jsonify({"message": "Picture verified", "scores": scores}), 201

if __name__ == "__main__":
    app.run(debug=True)