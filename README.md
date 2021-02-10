# AI Fabric Demos
This is a demo set for using AI fabric in UiPath orchestrator.
This demo set contains:
- A Model Template, will guide you how to write your own model 
- A Image recognition demo implemented with Tensorflow 2.0
- A Text classification demo, with both training and inferencing functions, with Keras
- A Multilabel text classification demo, with bert

## Pre-request: 
You have to have a orch tenant with AI Fab

## Call the predict with demo Model in folder P
1. Zip the Model folder, which contains the transfer-learning python script, and upload it to AI Fab as ML package
   
2. Create a ML skill in AI Fab, select the uploaded ML package, takes 2~10min to finish deploy

3. Connect the robot to the Orch with AI Fab, select the ML skill acitivity from UiPath Studio, find the AI skill which you had just created.

4. Now you will be able to 

## Retrain your own dataset locally
1. Set up the python enviroment, with tensorflow 2.0, or Tensorflow-gpu if you have an GPU of Nvidia

## Retrain your own dataset in Colab
TBD

## TBD as next step
TBD

## Resources
https://www.uipath.com/product/rpa-ai-integration-with-ai-center
https://docs.uipath.com/ai-fabric/v0/docs/about-ai-center
https://docs.uipath.com/ai-fabric/docs/about-ai-fabric
https://academy.uipath.com/learningpath-detail/2910/3/0/3

## License
This project is under [Apache License](http://www.apache.org/licenses/LICENSE-2.0), since the retrain.py is based on the demo of Google Tensorflow hub
