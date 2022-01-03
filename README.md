# my-photomath

Photomath pre-interview assignment solution. The goal is to implement a simple Photomath 'application', capable of calculating basic math expressions from photographs of hand drawn symbols. 

The algorithm processes images to a black and white palette, finds larger contours (to counter rogue pixels) and then finds their bounding boxes. The images inside bounding boxes are sorted component and fed into a neural network which returns a symbol which the bound contour represents. Symbols are then fed into a math parser as a single string which calculates the expression. 

Instructions:
- Unzip math-examples.zip and dataset.zip to the project root directory
- To train the model, run model_train.py. Model_train.py saves the model as model.pth and class id dictionary as class-ids.txt.
- To see how the OCR works, run OCR_and_processing_demo.py. The demo uses math-examples. Press any key to continue with image loading.
- To solve a math expression from an image, run photomath.py.


Current milestones:
- created a simple math symbol dataset consisting of 600 hand drawn symbols: [0-9], +, -, /, *, (, ).
- created a basic PyTorch neural network configuration. 
- created a simple OCR using openCV.
- math parser functionality added.
- photomath.py added. Outputs a math expression and its result from a given image.


To-do: 
- improve image preprocessing. Performance depends greatly on image brightness. Ignore surrounding symbols inside the bounding box.
- experiment with different neural network architectures. Test their performance compared to the number of parameters.
- expand math parser/solver functionality to include other operators and more complex expressions, expression validity checking,
  catching border cases such as division by zero, etc... 
- develop a proper flask web application to showcase the project
- improve and automate the workflow.
- increase overall modularity and refactor

