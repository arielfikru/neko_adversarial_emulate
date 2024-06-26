# Neko Adversarial Emulate

## Introduction

This project aims to emulate the workings of adversarial attacks but with a simpler and more cost-effective approach. The purpose is to understand and replicate the core concepts of adversarial attacks without the extensive computational requirements typically associated with such tasks.

## Comparison Plot

Below is the comparison plot generated by our script, illustrating the differences between the original ("Vanilla Image") and the adversarially emulated image ("Neko Adversarial Emulate").

![Comparison Plot](comparison_plot.png)

## TODO List

1. **Optimize Performance**
   - [ ] Parallelize the dithering process using multi-threading or multi-processing to speed up the operation on large images.
   - [ ] Investigate GPU-based optimization for the overlaying process to leverage CUDA capabilities fully.

2. **Error Handling**
   - [ ] Implement robust error handling for file operations (e.g., file not found, unsupported file formats).
   - [ ] Add try-except blocks to handle potential exceptions during tensor operations and image transformations.

3. **Input Validation**
   - [ ] Validate input image path to ensure it exists and is accessible.
   - [ ] Check if the input image is in a valid format (e.g., PNG, JPEG) before processing.

4. **Output Format Flexibility**
   - [ ] Allow users to specify the output image format (e.g., PNG, JPEG) via a function parameter or command-line argument.
   - [ ] Add functionality to preserve the alpha channel in output images if required.

5. **Logging and Progress Monitoring**
   - [ ] Implement a logging mechanism to record the progress and any issues encountered during the image processing steps.
   - [ ] Enhance the progress bar to show more detailed information about the processing stages (e.g., current image section being processed).

6. **Code Modularization**
   - [ ] Refactor the code into separate modules (e.g., dithering, noise generation, overlay) for better maintainability and readability.
   - [ ] Create a main function to orchestrate the workflow, making it easier to test and extend.

