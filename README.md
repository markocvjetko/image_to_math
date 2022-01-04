# my-photomath

Photomath pre-interview assignment solution. The goal is to implement a simple Photomath 'application', capable of calculating basic math expressions from photographs of hand drawn symbols. 

This project is not meant to replace the Photomath app. It only works with very prettily written math expressions and even then, some adjustments might need to be made depending on the image brightness. The scope of math problems which this version can solve is limited to integers and basic math operators, such as addition, subtraction, division, multiplication and brackets. 

The algorithm processes images to a black and white palette, finds larger contours (to counter rogue pixels) and then finds their bounding boxes. The images inside bounding boxes are sorted by the x component and fed into a neural network that outputs a symbol which the bound contour represents. Symbols, as a single string, are then fed into a math parser that calculates the expression. 

Instructions:
- Unzip math-examples.zip and dataset.zip to the project root directory
- To train the model, run model_train.py. Model_train.py saves the model as model.pth and class id dictionary as class-ids.txt.
- To see how the OCR works, run OCR_and_processing_demo.py. Press any key to continue with image loading.
- To solve a math expression from an image, run photomath.py.


Current milestones:
- created a simple math symbol dataset consisting of 600 hand drawn symbols: [0-9], +, -, /, *, (, ).
- created a basic PyTorch neural network configuration. 
- created a simple OCR using openCV.
- math parser functionality added.
- photomath.py added. Outputs a math expression and its result from a given image.


To-do: 
- improve image preprocessing. Performance depends greatly on image brightness. Make it so that pixels not part of the bounded contour are ignored.
- expand math parser/solver functionality to include other operators and more complex expressions: working with floating point numbers, expression validity checking, catching border cases such as division by zero, etc... 
- experiment with different neural network architectures. Test their performance compared to the number of parameters.
- develop a proper flask web application to showcase the project.
- improve and automate the workflow.
- increase overall modularity and refactor.

