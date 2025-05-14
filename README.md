# Neural Network Project

This project is a neural network implementation in raw C++ and its integration with Godot Game Engine. It provides tools for training, testing, and visualizing neural networks, with support for MNIST data and Godot-based UI components.

## Features

- **C++ Core**: Implements core neural network functionality using libraries like Eigen and nlohmann_json.
- **Python Utilities**: Includes Python scripts for data preprocessing and additional neural network implementations using NumPy and PyTorch.
- **Godot Integration**: Provides a Godot-based UI for visualization and interaction with neural networks.
- **MNIST Support**: Tools for working with the MNIST dataset.

## Project Structure

- **`libnn/`**: Core neural network implementation in C++.
- **`python/`**: Python scripts for neural network experiments and data handling.
- **`godot/`**: Godot project files for UI and visualization.
- **`tools/`**: Additional tools and utilities, including a main executable.
- **`gdextension/`**: Godot C++ extensions for neural network integration.

## Dependencies

Project dependencies are managed via [Nix](https://nixos.org/) package manager.
This will drop you into a shell with all required dependencies available:

```
nix develop
```

Alternatively, you can provide the dependencies via your favourite approach and hope for the best. 
See [flake.nix](flake.nix) for a list of packages used to develop this.

## Compilation

```
cmake --preset default
cmake --build build
```
