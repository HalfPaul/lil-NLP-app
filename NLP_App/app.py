from flask import Flask, render_template, url_for, request
import json
import requests
app = Flask(__name__)


@app.route("/", methods=["GET","POST"])
def home():
    return render_template("home.html", title="Home")

@app.route("/about", methods=["GET","POST"])
def about():
    return render_template("about.html", title="About")

@app.route("/john", methods=["GET","POST"])
def john():
    return render_template("john.html", title="John")

@app.route("/sam", methods=["GET","POST"])
def sam():
    return render_template("sam.html", title="Sam")


@app.route("/sam/get")
def chatbot_request():
    user_text = request.args.get('msg')

    payload = {'sentence': str(user_text)}
    response = requests.post("https://chatbot-api52.herokuapp.com/predict",  data=json.dumps(payload))
    return str(response.text)




if __name__ == "__main__":
	app.run(debug=True)