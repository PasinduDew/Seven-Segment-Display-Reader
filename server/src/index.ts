
// Import Libraries
// import * as admin from "firebase-admin";
import * as express from "express";
import * as bodyParser from "body-parser";
import * as cors from "cors";
// import * as tf from '@tensorflow/tfjs';

import * as tf from '@tensorflow/tfjs-node'

const PORT = 8000;
//initialize express server
const app = express();
const main = express();

// The Firebase Admin SDK to access Cloud Firestore.
// admin.initializeApp(functions.config().firebase);

// Storage Bucket



//initialize the database and the collection
// const db = admin.firestore();

// Automatically allow cross-origin requests
app.use(cors({ origin: true }));
// Middleware that transforms the raw string of req.body into json
main.use('/api/v1', app);

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

let model;
// let probability_model

// ###########################################################################################################################################
//                                                  REST End points - Functions
// ###########################################################################################################################################

// Invoke -> https://us-central1-ucsc-e-procurement.cloudfunctions.net/api/test
app.get("/test", cors(), (req, res) => {

    res.send("Ayubowan!, From Cloud Functions. I'm Working :)");


});

app.post("/digit-prediction", cors(), async (req, res) => {

    console.log("************************************ Request Received to 'digit-prediction'");
    const data = eval(req.body.numImageArray)
    console.log(data[0][0][0][0]);

    // let numImageArray = JSON.parse(req.body)

    // console.log(numImageArray);


    let predictions = [];
    for(let i = 0; i < data.length; i++){
        const prediction = await model.predict(data[i]);
        predictions.push(prediction)
    }
    
    console.log(predictions);

    



});



main.listen(PORT, async () => {
    console.log(`⚡️[server]: Server is running at https://localhost:${PORT}`);
    model = await tf.loadLayersModel('file://E:/Computer Vision/Projects/Seven-Segment-Display-Reader/TFJS_Saved/model.json');
    // probability_model = tf.Sequential([model, tf.keras.layers.Softmax()])

    console.log("TF Model: Loaded Successfully");
    

});

