# OSFM NPU Inference Script
## _Documentation generated by [FastWrite](https://fastwrite-py.netlify.app)_

**Introduction**
===============

This documentation outlines the functionality and usage of a Python application designed to utilize Natural Processing Units (NPU) for text generation. The application automatically detects the available NPU platform (Intel or AMD) and loads the corresponding model for inference.

**Overview**
------------

The application consists of two primary functions:

*   **NPU Detection and Model Loading**: This function detects the available NPU platform and loads the corresponding model based on the provided model name.
*   **Main Application Loop**: This function initializes the NPU detection and model loading process, then enters a loop where it prompts the user for input, generates a response using the loaded model, and prints the response.

**File-Level Documentation**
---------------------------

The application is contained within a single Python file (`main.py`). This file includes the necessary import statements, function definitions, and the main application loop.

**Function-Level Documentation**
-------------------------------

### detect_and_load

*   **Purpose**: Detects the available NPU platform (Intel or AMD) and loads the corresponding model based on the provided model name.
*   **Parameters**:
    *   `model_name` (str): The name of the model to load.
*   **Returns**:
    *   `tokenizer`: The tokenizer object for the loaded model.
    *   `model`: The loaded model object.
    *   `generate_fn`: The function used to generate text using the loaded model.

### main

*   **Purpose**: Initializes the NPU detection and model loading process, then enters a loop where it prompts the user for input, generates a response using the loaded model, and prints the response.
*   **Parameters**: None
*   **Returns**: None

**Usage**
---------

1.  Install the required dependencies (e.g., intel_extension_for_pytorch or optimum[amd]) to utilize the NPU backend.
2.  Run the application using Python (e.g., `python main.py`).
3.  Interact with the application by typing input prompts, and the application will generate responses using the loaded model.
4.  Type 'exit' or 'quit' to terminate the application.