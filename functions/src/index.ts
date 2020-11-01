// Import Libraries
import * as functions from 'firebase-functions';
import * as admin from "firebase-admin";
import * as express from "express";
import * as bodyParser from "body-parser";
import * as cors from "cors";

//initialize express server
const app = express();
const main = express();

// The Firebase Admin SDK to access Cloud Firestore.
admin.initializeApp(functions.config().firebase);

//initialize the database and the collection
// const db = admin.firestore();

// Automatically allow cross-origin requests
app.use(cors({ origin: true }));

// Middleware that transforms the raw string of req.body into json
main.use('/v1', app);
main.use(bodyParser.json());
main.use(bodyParser.urlencoded({ extended: false }));



// ###########################################################################################################################################
//                                                  REST End points - Functions
// ###########################################################################################################################################

// Invoke -> https://us-central1-ucsc-e-procurement.cloudfunctions.net/api/test
app.get("/test", cors(), (req, res) => {
  
    res.send("Ayubowan!, From Cloud Functions. I'm Working :)");

});


export const api = functions.https.onRequest(main);