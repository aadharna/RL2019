For Tabular CartPole:

To run this, all you need to do is call:

python Cartpole_Dharna.py 

To change the number of iterations that you learn the policy for, just edit the second argument of mc_control_GLIE.

The Fn signature is mc_control_GLIE(environment, number_of_episodes, alpha, gamma)

I have also saved the policy that I created using my code. Therefore, another way you could test this, rather than re-learning the policy, would just be to use my policy. 

It's saved as a .pkl file, and I have included the code to use my policy. To use my policy, keep the file as current. To build your own policy, comment out the load_obj call, and uncomment out the learning call which returns Pi and Q (optimal).  

----

For Policy Gradient Cartpole, see `rl-pg.ipynb` -- needs some documentation
