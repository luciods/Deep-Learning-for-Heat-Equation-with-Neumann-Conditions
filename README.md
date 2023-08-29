# Deep-Learning-for-Laplace-Equation
The universal approximation theorem guarantees the power of neural networks to approximate any function. This means solving partial differential equation too. 
In the simplest construction of the network, the solution $u(t,x)$ is given by $u(t,x)\simeq nn(t,x;\theta)$ where $nn(t,x;\theta)$ is the output and $\theta$ are the parameters that minimize best the loss function $$J(\theta)=\dfrac{1}{\Omega}\sum_{(t_i,x_i)\in [0,T]\times \Omega}\left(\mathcal{L}u-f(t,x)\right)$$
