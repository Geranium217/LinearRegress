# Linear Regression with CUDA

This project demonstrates how to perform linear regression using CUDA. The project includes a C++ program that uses the CUDA runtime to perform linear regression on a dataset.

**Building the Project**

To build the project, follow these steps:

1. Install the CUDA toolkit and the NVIDIA GPU driver.
2. Create a new directory for the project and navigate to it.
3. Clone the project repository using the command `git clone https://github.com/your-username/linear-regression-cuda.git`.
4. Navigate to the project directory and run the command `nvcc -o linear_regression linear_regression.cpp -lcudart`.
5. Run the program using the command `./linear_regression`.

**Running the Program**

To run the program, follow these steps:

1. Navigate to the project directory.
2. Run the program using the command `./linear_regression`.

**Dataset**

The program uses a sample dataset to perform linear regression. The dataset is stored in a file called `data.txt` and has the following format:

x1 y1 x2 y2 ... xn yn


Each row represents a data point, where `x` is the independent variable and `y` is the dependent variable.

**Linear Regression Algorithm**

The program uses the following linear regression algorithm:

1. Initialize the coefficients `a` and `b` to 0.
2. Iterate over the dataset and calculate the sum of the squared errors for each data point.
3. Update the coefficients `a` and `b` using the following formulas:

`a = (Σ(x_i * y_i) - n * mean(x) * mean(y)) / (Σ(x_i^2) - n * mean(x)^2)`
`b = mean(y) - a * mean(x)`

4. Repeat steps 2-3 until the coefficients converge.

**CUDA Implementation**

The program uses the CUDA runtime to perform the linear regression algorithm on the GPU. The CUDA implementation is as follows:

1. Allocate memory on the GPU for the dataset and the coefficients.
2. Copy the dataset from the host to the GPU.
3. Launch a kernel on the GPU to perform the linear regression algorithm.
4. Copy the coefficients from the GPU to the host.
5. Print the coefficients to the console.

**License**

This project is licensed under the MIT License.

**Authors**

* Your Name

**Acknowledgments**

* NVIDIA for providing the CUDA toolkit and the NVIDIA GPU driver.
* The authors of the linear regression algorithm for providing the algorithm
