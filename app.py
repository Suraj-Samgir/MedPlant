from flask import Flask,request, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

import numpy as np
# from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from PIL import Image

app = Flask(__name__)

#Database Configurations...
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///Database.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)   
app.app_context().push()

#Creating Table and its columns in database...
class Feedback(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    gender = db.Column(db.String(7),nullable=False)
    country = db.Column(db.String(20),nullable=False)
    email = db.Column(db.String(35),nullable=False)
    phone = db.Column(db.Integer,nullable=False)
    feed = db.Column(db.String(250),nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

#Creating the Table for Order and its columns...
class Order(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50),nullable=False)
    plant = db.Column(db.String(25),nullable=False)
    email = db.Column(db.String(35),nullable=False)
    phone = db.Column(db.Integer,nullable=False)
    date = db.Column(db.String(12),nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

#Creating the class to display the information of plant...
class Information(db.Model):
    sno = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(50),nullable=False)
    info = db.Column(db.String(1000),nullable=False)

#Loading the deep learning model...
model = load_model('Medicinal_plants/plant.h5')
img_size = (256,256)

#function for preprocessing...
def preprocess_image(image_path):
    img = image.load_img(image_path,target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    return img_array

def predict_image(img_path):
    processed_image = preprocess_image(img_path)
    preds = model.predict(processed_image)

    class_names = ['Alpinia Galanga (Rasna)',
    'Amaranthus Viridis (Arive-Dantu)',
    'Artocarpus Heterophyllus (Jackfruit)',
    'Azadirachta Indica (Neem)',
    'Basella Alba (Basale)',
    'Brassica Juncea (Indian Mustard)',
    'Carissa Carandas (Karanda)',
    'Citrus Limon (Lemon)',
    'Ficus Auriculata (Roxburgh Fig)',
    'Ficus Religiosa (Peepal Tree)',
    'Rosa-sinensis (Hibiscus)',
    'Jasminum (Jasmine)',
    'Mangifera Indica (Mango)',
    'Mentha (Mint)',
    'Moringa Oelifera (Drumstick)',
    'Muntingia Calbura (Jamaica Cherry-Gasagase)',
    'Murraya Koenigli (Curry)',
    'Nerium Oleander (Oleander)',
    'Nyctanthes Arbor-tristis (Parijata)',
    'Ocimum Tenuiflorum (Tulsi)',
    'Piper Betle (Betel)',
    'Plectranthus Amboinicus (Mexican Mint)',
    'Pongamia Pinnata (Indian Beech)',
    'Psidium Guajava (Guava)',
    'Punica Granatum (Pomogranate)',
    'Santalum Album (Sandalwood)',
    'Syzygium Cumini (Jamun)',
    'Syzygium Jambos (Rose Apple)',
    'Tabernaemontana Divaricata (Crape Jasmine)',
    'Trigonella Foenum-graecum (Fenugreek)']

    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)*100
    
    return predicted_class,confidence

@app.route('/', methods = ['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/contact', methods = ['GET','POST'])
def contact():
    if request.method == 'POST':
        print("post")   
        tempname = request.form['Name']
        tempgender = request.form['Gender']
        tempcountry = request.form['Country']
        tempemail = request.form['Email']
        tempphone = request.form['Phone']
        tempfeed = request.form['Feedback']

        ins = Feedback(name=tempname, gender=tempgender, country=tempcountry, email=tempemail, phone=tempphone,feed=tempfeed)

        db.session.add(ins)
        db.session.commit()

    return render_template('contact.html')

@app.route('/order', methods = ['GET','POST'])
def order():
    if request.method == 'POST':
        tempname = request.form['Name']
        tempgender = request.form['Plant']
        tempemail = request.form['Email']
        tempphone = request.form['Phone']
        tempdate = request.form['Date']

        ins = Order(name=tempname, plant=tempgender, email=tempemail, phone=tempphone,date=tempdate)
        db.session.add(ins)
        db.session.commit()
        return render_template('booked.html')
    
    return render_template('order.html')

@app.route('/submit', methods = ['POST','GET'])
def get_output():
    if request.method == 'POST':
        img = request.files['inputfile']
        if img.filename != '':
            img_path = "static/uploads/"+img.filename
            img.save(img_path)
            result,accuracy = predict_image(img_path)
            acc=round(accuracy,2)
            plant_data = Information.query.filter_by(name=result).all()
            data = [{'info':item.info} for item in plant_data]
            value = data[0]
            info = value['info']
            return render_template('index.html',result=result,accuracy=acc,info=info)
    
    return "Please enter a image"

if __name__ == "__main__":
    app.run(debug=True)