# coding: utf-8
"""
Escape Time Computation
=======================

Mean Exit Time For a SDE 

.. math::

   \begin{equation}\begin{pmatrix}{dx \\ dy}\end{pmatrix} = \begin{pmatrix}1 & 0 \\
   0 & 1\end{pmatrix}dW_t \end{equation}

\ 

By using two approaches:

-  kinetic Monte Carlo Simulations
-  solving the Boundary Value Problem with Finite Elements

We will illustrate that numerically computed Mean Exit Times for this 2D
example apparoximate the theoretical value of the exit time that is
\ :math:`0.5`\  \* Oksendal, Bernt. Stochastic differential equations:
an introduction with applications. Springer Science & Business Media,
2013.

Dependencies
^^^^^^^^^^^^

-  numpy pip install numpy
-  matplotlib pip install matplotlib
-  tqdm pip install tqdm
-  shapely pip install Shapely
-  joblib pip install joblib
-  seaborn pip install seaborn
-  fenics (for solving the Boundary Value Problem)
   `Installation_Instruction <https://fenicsproject.org/download/archive/>`__

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from shapely.geometry import Polygon, Point, MultiPoint
    from joblib import Parallel, delayed
    import seaborn as sns
    
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=21)
    plt.rc('font', family='serif')
    plt.rc('font', family='serif')
    plt.rcParams['image.cmap'] = 'Spectral'
    np.random.seed(2)
    rng = np.random.default_rng(5)

Define Equations For the Circle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def equation(theta,r):
        x = np.cos(theta)*r
        y = np.sin(theta)*r
        return x,y

The user chooses the number of points to sample on the circle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    N = int(input("Enter Number of Points to Create the Circle: "))
    T =  np.linspace(0,360,N)*np.pi/180
    Circle = np.array(equation(T,1)).T


.. parsed-literal::

    Enter Number of Points to Create the Circle: 1000


.. code:: ipython3

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.scatter(Circle[:,0],Circle[:,1],c='k',s=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ##The Circle is saved and will be used for the Finite Difference Code as well. 
    np.savetxt('Circle_Boundary.csv',Circle,delimiter=',')



.. image:: output_8_0.png


The Shapely Library is used to find based on the data sampled on the circle which are inside the circle and which are outside.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will help as to identify when a trajectory escapes from the circle.
A grid of points is created and we test if the points are inside or
outside of the circle

.. code:: ipython3

    poly = Polygon(Circle)
    X,Y = np.meshgrid(np.linspace(-1.1,1.1,200),np.linspace(-1.1,1.1,200))
    meshpoint = np.c_[X.reshape(-1,1),Y.reshape(-1,1)]
    meshpoint_ = MultiPoint(meshpoint)

Test if a point in the grid is inside or outside of the limit cycle.

.. code:: ipython3

    judge_list = []
    
    for i in tqdm(range(len(meshpoint))):
        result = poly.contains(meshpoint_.geoms[i])
        judge_list.append(result)


.. parsed-literal::

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 40000/40000 [00:01<00:00, 20398.44it/s]


Simulate a trajectory until it escapes from the circle starting from the
origin.

.. code:: ipython3

    t = 0.0
    dt=1e-5
    traj =[]
    x = np.zeros(2)
    traj.append(x)
    while True:
        traj.append(x)
        if poly.contains(Point(x)) == False:
            break
        traj.append(x)
        x = x+ np.sqrt(dt) * np.random.randn(2)
    traj = np.array(traj)

We visualize the points inside and outside of the circle based on Shapely and also a trajectory (starting from the origin) integrated with Euler-Maruyama until it escapes.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    judge_list = np.array(judge_list)
    
    inside_pts = meshpoint[judge_list == True]
    outside_pts = meshpoint[judge_list == False]
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.scatter(inside_pts[:, 0], inside_pts[:, 1], color='r', label='Interior',s=2,zorder=0)
    ax.scatter(outside_pts[:, 0], outside_pts[:, 1], color='b', label='Exterior',s=2,zorder=0)
    ax.scatter(Circle[:,0],Circle[:,1],c='k',s=5)
    ax.plot(traj[:,0],traj[:,1],'g-',label='Trajectory',linewidth=0.2)
    ax.plot(0,0,'ko',markersize=5)
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=15,frameon=True,loc = 'upper right')
    plt.tight_layout()



.. image:: output_16_0.png


kinetic Monte Carlo run Experiments - Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions: \* To compute the escape time with kinetic Monte Carlo the
user specifies the number of experiments (trajectories) to simulate and
also the time step (dt).

Suggestions: \* Use a time step dt~1e-5 \* Run about 10,000 experiments
to get a good sample of the statistics (this might involve running for
~20 mins)

.. code:: ipython3

    n = int(input("Enter Number of Experiments: "))  # For 10,000 experiments 10,000 (needs ~22mins in 24 cores)
    dt: float = float(input("Select Time Step: "))   # Time step length (suggested time step ~1e-5)
    
    def integrate_until_exit(random_seed):
        np.random.seed(random_seed)
    
        t = 0.0
        x = np.zeros(2)
    
        while True:
            if poly.contains(Point(x)) == False:
            # if np.linalg.norm(x) >= 1.0: 
            ### For this toy example we can use also the above as a stopping condition.
                break
            x += np.sqrt(dt) * np.random.randn(2)
            t += dt
    
        return t
    
    
    fpts = Parallel(n_jobs=-1, verbose=1)(
        delayed(integrate_until_exit)(i) for i in (range(0,n)))
    
    print('MFPT', np.mean(fpts), '+/-', np.std(fpts))
    
    fpts = np.array(fpts)


.. parsed-literal::

    Enter Number of Experiments: 100
    Select Time Step: 1e-3


.. parsed-literal::

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:    1.0s


.. parsed-literal::

    MFPT 0.5049199999999969 +/- 0.34225363927939706


.. parsed-literal::

    [Parallel(n_jobs=-1)]: Done  77 out of 100 | elapsed:    1.4s remaining:    0.4s
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    1.6s finished


.. code:: ipython3

    fig = plt.figure(figsize=(4*2,4))
    ax = fig.add_subplot(121)
    sns.kdeplot(fpts,fill=True);



.. image:: output_19_0.png


Imports the Code that Solves the Boundary Value Problem with Finite Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import BVP_Stationary as BV


.. parsed-literal::

    Solving linear variational problem.
    MFPT: 0.49996583014439444


"""