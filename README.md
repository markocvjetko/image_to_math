# my-photomath

Photomath pre-interview assignment solution.


Current milestones:

- created a simple math symbol dataset consisting of 600 hand drawn symbols: [0-9], +, -, /, *, (, ).
- created a basic PyTorch neural network configuration. 
- created a simple OCR using openCV.
- math parser functionality added.
- photomath.py added. Outputs a math expression and its result from a given image.

To-do: 
- improve image preprocessing. Performance depends on image brightness. 
- experiment with different neural network architectures. Test their performance compared to the number of parameters.
- expand math parser/solver functionality to include other operators and more complex expressions, expression validity checking,
  catching border cases such as division by zero, etc... 
- develop a proper flask web application to showcase the project
- improve and automate the workflow.
- increase overall modularity and refactor

