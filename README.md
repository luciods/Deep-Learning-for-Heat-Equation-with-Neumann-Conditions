# Deep-Learning-for-Heat-Equation-with-Neumann-Conditions
The universal approximation theorem guarantees the power of neural networks to approximate any function. This means solving partial differential equations (PDEs) too. 
In the simplest construction of the network, the solution $u(t,x)$ is given by $u(t,x)\simeq nn(t,x;\theta)$ where $nn(t,x;\theta)$ is the output and $\theta$ are the parameters that minimize best the loss function $$J(\theta)=\dfrac{1}{|\Omega||[0,T]|}\sum_{(t_i,x_i)\in [0,T]\times \Omega}\left(\hat{\mathcal{L}}nn(t_i,x_i;\theta)-s(t_i,x_i)\right)^2+J_{IC}+J_{B}$$ where $\hat{\mathcal{L}}$ is the differential operator and $IC$ and $B$ are the initial and boundary conditions respectively. As an optimization problem, the possiblity of getting stuck into some local minimum is not negligible. An improvement could be the 'weighting' of the loss function, for example $J(\theta)=p_1 J_{\mathcal{L}}+p_2 J_{IC}+p_3 J_{B}$ with $p_1+p_2+p_3=1$, in order to force the optimizer to give higher account to some terms of the loss function. Another powerful way to improve the convergence is the construction of the solution in order to automatically satisfies initial and boundary conditions. Such minimization problem is called unconstrained, since the only term to be minimized is the loss of the differential equation. The solution could be written as $$u(t,x)=G(t,x)+D(t,x)nn(t,x;\theta)$$ where $G$ represents the initial and boundary conditions and $D$ is a distance function for $(t,x)\in\times[0,T]\times\Omega$ to $(t=0)\times\partial\Omega$. The ansatz is inspired by the first article below. Theoretically, the complexity of the process is due to the training of $nn(t,x;\theta)$, since $G(t,x)$ and $D(t,x)$ are defined as pre-trained functions with low capacity neural network. The effectiveness of this approach has been demonstrated in numerous articles, particularly when Dirichlet conditions are applied to certain partial differential equations (PDEs). The reason is that Dirichlet conditions are easily computed and the distance function could be written analytically as $D(x)=min_{x_0\in\partial\Omega}||x-x_0||$ in case of problems with no temporal dependence.

In this project, we try to solve the heat equation with Neumann conditions. The problem is
$$\partial_t u(t,x)=\alpha\partial^2_x u(t,x)$$ $$u(0,x)=f(x)$$ $$\partial_x u(t,L)=\partial_x u(t,-L)=0$$ with $(t,x)\in[0,T]\times[-L,L]$. Following the above ansatz, the functions $G$ and $D$ should obey 
$$G(0,x)=f(x),\quad \partial_x G(t,L)=\partial_x G(t,-L)=0$$ $$D(0,x)=0,\quad \partial_x D(t,L)=\partial_x D(t,-L)=0, \quad D(t,L)=D(t,-L)=0.$$ Such construction 


## References
[https://arxiv.org/abs/1711.06464]

[https://arxiv.org/abs/2006.08472]

[https://arxiv.org/abs/2104.08426]
