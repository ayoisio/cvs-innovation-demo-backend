import os
import time

from dotenv import load_dotenv
from flask import Flask, Response
from firebase_admin import credentials, initialize_app
from flask_cors import CORS
from random_word import RandomWords

from routes.chat import chat_bp


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
cors = CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["OPTIONS", "POST", "GET"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True
    }
})


# Initialize Firebase admin
cred = credentials.ApplicationDefault()
initialize_app(cred)

app.register_blueprint(chat_bp)


@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204


@app.route("/hello-world")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"


@app.route("/test-stream", methods=["POST"])
def test_stream():
    """Test streaming response with random words."""
    def mock_generate():
        r = RandomWords()
        while True:
            random_word = r.get_random_word()
            if random_word:
                yield random_word + "\n"
            time.sleep(2)

    return Response(mock_generate(), mimetype="text/plain")


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
