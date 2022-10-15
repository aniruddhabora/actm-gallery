# coding: utf-8
"""

Simulate Evolution of Graphs
============================

.. raw:: html

   <center>

The Notebook Below Simulates an Evolving Graph based on the
Susceptible-Infected Susceptible model

.. raw:: html

   </center>

###

.. raw:: html

   <center>

The code produces trajectories in terms of the fraction of the infected
population :math:`\theta_I` and the fraction of the edges between the
susceptible population $g_{ss} $

.. raw:: html

   </center>

References:

-  Gross, Thilo, and Ioannis G. Kevrekidis. “Robust oscillations in SIS
   epidemics on adaptive networks: Coarse graining by automated moment
   closure.” EPL (Europhysics Letters) 82.3 (2008): 38004.
-  Kattis, Assimakis A., et al. “Modeling epidemics on adaptively
   evolving networks: a data-mining perspective.” Virulence 7.2 (2016):
   153-162.

Dependencies
^^^^^^^^^^^^

-  numpy pip install numpy
-  matplotlib pip install matplotlib
-  tqdm pip install tqdm

.. raw:: html

   <center>

One sampled trajectory for :math:`p=0.00075` from the evolving graph is
shown in the figure below

.. raw:: html

   </center>

.. figure:: attachment:Trajectory_display.png
   :alt: Trajectory_display.png

   Trajectory_display.png

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    from Full_Network_Functions import * ## 
    
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=21)
    plt.rc('font', family='serif')
    plt.rc('font', family='serif')
    plt.rcParams['image.cmap'] = 'Spectral'
    np.random.seed(2)

Parameters Used For The Simulation Below
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Parameters governing: \* The number of Nodes, :math:`N=10,000` \*
The number of Edges, :math:`L=100,000` \* The inital fraction of
Infected people, :math:`Y_0 =0.5` are kept fixed on the
Full_Network_Functions.py file.

.. code:: ipython3

    w0 = 0.06 #rewiring parameter
    r = 0.0002 #recover probability
    p = 0.00075 #infection parameter

Load an initial configuration of the graph 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user can choose between initializing a random graph or loading an
existing graph

.. code:: ipython3

    data = load_initial_graph_question()


.. parsed-literal::

    Do you want to load an existing graph? (yes or no)yes


Run Simulations for the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`p=0.00075` you might need to run about 20,000 steps (~20
mins) to capture the stable limit cycle

.. code:: ipython3

    Number_of_Steps=int(input('Select the Number of Time Steps:' ))
    stat_list = []
    I_node = data[0];edge_list = data[1]
    stat_array, I_node, edge_list = iterate(I_node, edge_list, Number_of_Steps,r, p,w0)


.. parsed-literal::

    Select the Number of Time Steps:100


.. parsed-literal::

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:07<00:00, 12.93it/s]


.. code:: ipython3

    fig = plt.figure(figsize=(4*3,4))
    ax = fig.add_subplot(131)
    ax.plot(stat_array[:,0],'.k')
    ax.set_ylabel(r'$\theta_I$')
    ax.set_xlabel('Iterations')
    
    ax = fig.add_subplot(132)
    ax.plot(stat_array[:,1],'.k')
    ax.set_ylabel(r'$g_{ss}$')
    ax.set_xlabel('Iterations')
    
    ax = fig.add_subplot(133)
    ax.plot(stat_array[:,0],stat_array[:,1],'.k')
    ax.set_ylabel(r'$g_{ss}$')
    ax.set_xlabel(r'$\theta_I$')
    
    plt.tight_layout()



.. image:: output_14_0.png




"""


