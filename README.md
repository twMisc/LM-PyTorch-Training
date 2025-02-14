# LM-PyTorch-Training

This repository contains the code for training a PyTorch network inheriting from `torch.nn.Module` using the Levenberg–Marquardt algorithm.
The code utilizes the `torch.func` (previously known as `functorch`) to compute the Jacobian of the model with respect to its parameters.

## Requirements

This code requires torch>=2.0.0 to support the `torch.func` module.

## Usage

The following is an example of how to use the code to train a simple DNN to approximate the sine function.

```python
import torch
from torch.func import functional_call
from lm_train.network import DNN
from lm_train.training_module import training_LM

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DNN([1, 50, 1]).to(device)       # can be any model inheriting from torch.nn.Module
params = dict(model.named_parameters())  # the model parameters in a dictionary
for p in params.values():                # Set requires_grad to False for lower memory usage
    p.requires_grad = False              # we do not need to engage PyTorch's autograd when using torch.func 

x = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
y = torch.sin(2 * torch.pi * x).to(device)

def model_u(data, params):
    return functional_call(model, params, (data, )) 

def loss_mse(params, *args, **kwargs):
    "Mean squared error loss"
    data, target, = args
    output = model_u(data, params)
    loss_value = output.flatten() - target.flatten()
    return loss_value

losses = [loss_mse]                        # a list of loss functions
inputs = [[x, y]]                          # a list of lists of inputs for each loss function
kwargs = [{} for _ in range(len(losses))]  # a list of dictionaries of keyword arguments for each loss function
args = tuple(zip(losses, inputs, kwargs))  
params, lossval_all, loss_running, lossval_test = training_LM(
    params,
    device,
    args,
)

x_test = torch.linspace(0, 1, 100000).reshape(-1, 1).to(device)
output = model_u(x_test, params)
target = torch.sin(2 * torch.pi * x_test).to(device)
error = torch.linalg.norm(output - target, float('inf'))
print(f'The L_inf error is: {error:.4e}')
```

## Levenberg–Marquardt algorithm

The Levenberg–Marquardt algorithm minimizes the loss function given by

$$
\text{Loss}(\theta) = \sum_{i=1}^M (\mathcal{L}(u_{\theta}(x_i)))^2
$$

where $u_{\theta}(x_i)$ is the output of the model at the input $x_i$ and $\mathcal{L}$ is the loss function. (e.g. $\mathcal{L}(u_{\theta}(x_i)) = (u_{\theta}(x_i) - y_i)/\sqrt{M}$ for the mean squared error loss).

Given the Jacobian matrix $J := \frac{\partial \mathcal{L}}{\partial \theta}$, in each update step the algorithm solves the linear system

$$ (J^\top J + \mu I) \Delta \theta = -J^T \mathcal{L} $$

where $\mu$ is the damping parameter, and $\Delta \theta$ is the update to the parameters $\theta$.

In this implementation, we utilize the `torch.func.vmap` and `torch.func.jacrev` to compute the Jacobian matrix $J$. You will need to define $\mathcal{L}$ as a function that takes the model parameters and the input data as input and returns the loss value as in the example above.

## Notebooks

- `01_Function_approximation.ipynb`: Fitting the $\text{sinc}$ function using a simple DNN. Compare the results with the Adam optimizer.
- `02_Poisson_2D_PINNs.ipynb`: Training a 2D Poisson PINN using the Levenberg–Marquardt algorithm. Compare the results with the Adam optimizer.

## Results

### Function approximation

![Function approximation](img/sinc.png)

### Physics-informed neural network (2D Poisson)

![Poisson PINNs](img/2d_poisson.png)
