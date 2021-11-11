from flask import Flask, render_template, request, Markup
import os
import random
from numpy import random
import torch
from torchvision import transforms
from PIL import Image
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Parameter Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224
with open("cat_to_name.json", 'r') as f:
    cat_to_name = json.load(f)

labels_map = {
    0:'1',
    1:'10',
    2:'100',
    3:'101',
    4:'102',
    5:'11',
    6:'12',
    7:'13',
    8:'14',
    9:'15',
    10:'16',
    11:'17',
    12:'18',
    13:"19",
    14:'2',
    15:'20',
    16:'21',
    17:'22',
    18:'23',
    19:'24',
    20:'25',
    21:'26',
    22:'27',
    23:'28',
    24:'29',
    25:'3',
    26:'30',
    27:'31',
    28:'32',
    29:'33',
    30:'34',
    31:'35',
    32:'36',
    33:'37',
    34:'38',
    35:'39',
    36:'4',
    37:'40',
    38:'41',
    39:'42',
    40:'43',
    41:'44',
    42:'45',
    43:'46',
    44:'47',
    45:'48',
    46:'49',
    47:'5',
    48:'50',
    49:'51',
    50:'52',
    51:'53',
    52:'54',
    53:'55',
    54:'56',
    55:'57',
    56:'58',
    57:'59',
    58:'6',
    59:'60',
    60:'61',
    61:'62',
    62:'63',
    63:'64',
    64:'65',
    65:'66',
    66:'67',
    67:'68',
    68:'69',
    69:'7',
    70:'70',
    71:'71',
    72:'72',
    73:'73',
    74:'74',
    75:'75',
    76:'76',
    77:'77',
    78:'78',
    79:'79',
    80:'8',
    81:'80',
    82:'81',
    83:'82',
    84:'83',
    85:'84',
    86:'85',
    87:'86',
    88:'87',
    89:'88',
    90:'89',
    91:'9',
    92:'90',
    93:'91',
    94:'92',
    95:'93',
    96:'94',
    97:'95',
    98:'96',
    99:'97',
    100:'98',
    101:'99',
}

# Load model
model = torch.load("models/best.pt")

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        extra = ""
        image = request.files['file']
        if image:
            print(image.filename)
            print(app.config['UPLOAD_FOLDER'])
            source = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            print("Save =", source)
            image.save(source)
            img0 = Image.open(source)
            trans = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
            img0 = trans(img0)
            img0 = img0.unsqueeze(0)
            print(img0)
            with torch.no_grad():
                pred=model(img0)
                predicted = cat_to_name[labels_map[pred[0].argmax(0).item()]]
            
            extra += predicted
            
        return render_template('index.html', user_image = image.filename, rand = random.random(), msg = "Upload file successfully", extra=Markup(extra))
    else:
        return render_template("index.html")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8080, debug=True)
            
