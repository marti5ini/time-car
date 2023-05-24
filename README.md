# The importance of Time in Causal Algorithmic Recourse

This repository contains the implementation of the work [The Importance of Time in Causal Algorithmic Recourse](https://) in which 
we motivate **the need to integrate the temporal dimension into causal algorithmic recourse methods** to enhance recommendations’ **plausibility and reliability**. 


# Quick Start

You can utilize the following Jupyter Notebook to observe the results presented in the paper: [TIME-CAR](https://github.com/marti5ini/time-car/blob/master/time-car.ipynb)


# Setup

The code requires a python version >=3.9, as well as some libraries listed in requirements file. For some additional functionalities, more libraries are needed for these extra functions and options to become available. 

```
git clone https://github.com/marti5ini/time-car.git
cd time-car
```

Dependencies are listed in requirements.txt, a virtual environment is advised:

```
python3 -m venv ./venv # optional but recommended
pip install -r requirements.txt
```

# SCM used

We consider a semi-synthetic SCM based on the German Credit dataset that consists of the following structural equations and noise distributions:

\begin{center}
    \resizebox{0.95\textwidth}{!}{%
        \begin{minipage}{\textwidth}
            \begin{align*}
            (Gender) \qquad G &:=U_G, \qquad &U_G \sim \operatorname{Bernoulli}(0.5) \\[0.3em]
            (Age) \qquad A &:=-35+U_A, \qquad &U_A \sim \operatorname{Gamma}(10,3.5)\\[0.3em]
            (Education) \qquad E &:= G+ A +U_E, \quad &U_E \sim \mathcal{N}(0,1)\\[0.3em]
            (Job) \qquad J &:= G + 2 A + 4 E + U_J, \quad &U_J \sim \mathcal{N}(0,2)\\[0.3em]
            (Loan\; Amount) \qquad L&:= A + 0.5 G+ U_L, \quad &U_L \sim \mathcal{N}(0,3)\\[0.3em]
            (Loan\; Duration) \qquad D&:=G - 0.5 A + 2L+U_D, \quad &U_D \sim \mathcal{N}(0,2)\\[0.3em]
            (Income) \qquad I&:=0.5G + A + 4 E + 5 J + U_I, \quad &U_I \sim \mathcal{N}(0,4)\\[0.3em]
            (Savings) \qquad S&:= 5 I +U_S \quad &U_S \sim \mathcal{N}(0,2)
            \end{align*}
         \end{minipage}
    }
\end{center}

