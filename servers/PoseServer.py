"""Flask Server Used to retrieve human pose data from imput camera"""
from flask import Flask, request

app = Flask(__name__)

@app.route('/pose',methods=['POST','GET'])
def get_pose():
    if request.method == 'POST':
        current_pose = request.data

    e





if __name__ == "__main__":
    app.run(port=5000)
