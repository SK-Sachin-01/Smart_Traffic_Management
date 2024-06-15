# Smart Traffic Management

Smart Traffic Management is a comprehensive project designed to optimize urban traffic flow and reduce congestion through intelligent control systems. This repository provides the necessary code and resources to implement a sophisticated traffic management system using Python, focusing on adaptability and real-time data analysis.

## Features

- **Adaptive Traffic Control:** This system dynamically adjusts traffic signals based on real-time data, optimizing traffic flow and reducing wait times. The adaptive control ensures efficient management of varying traffic conditions throughout the day.
- **Data-Driven:** Utilizes YAML configuration files, making it easy to customize traffic management settings. This approach allows for flexibility and precise control over traffic signal timings and flow rates.
- **Simulation:** Includes a detailed Jupyter Notebook that facilitates the simulation and visualization of various traffic scenarios. This feature is essential for testing and refining traffic control strategies before implementation.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/SK-Sachin-01/Smart_Traffic_Management.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Smart_Traffic_Management
    ```
3. Run the main script to start the simulation:
    ```bash
    jupyter notebook main_script.ipynb
    ```

### Configuration

The system can be configured using the `data_1.yaml` file. This file contains a variety of parameters for traffic control, including signal timings and traffic flow rates. By modifying this file, users can tailor the system to meet specific requirements and adapt to different traffic environments.

### Simulation

The Jupyter Notebook (`main_script.ipynb`) includes comprehensive steps to simulate traffic scenarios. This notebook not only helps visualize traffic flow but also demonstrates the effects of different control strategies, making it a valuable tool for planning and decision-making.

## Files

- `control.py`: Contains the core logic for controlling traffic signals. This script implements the adaptive algorithms that adjust signal timings based on real-time traffic data.
- `data_1.yaml`: A configuration file that sets up various parameters for traffic management, allowing users to customize the system to their needs.
- `main_script.ipynb`: A Jupyter Notebook designed for simulating and visualizing the traffic management system. It provides an interactive platform for experimenting with different traffic control strategies and observing their outcomes.

