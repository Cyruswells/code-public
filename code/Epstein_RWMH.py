import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import mcmc
import time


#Environment
tfb = tfp.bijectors
tfe = tf.executing_eagerly
assert tfe()   

# Load Date
data = pd.read_csv(r'D:\code\data\rbc_1_200.csv').astype(np.float64)
c_t = data['c_obs'].values
i_t = data['i_obs'].values
y_t=c_t+i_t

# Known parameters  
delta = np.float64(0.025)
sigma = np.float64(0.1)

# Define the parameters of the Epstein-Zin utility function
psi = tf.constant(1.5, dtype=tf.float64)  # The inverse of the elasticity of substitution, here assumed to be 1.5
gamma = tf.constant(2.0, dtype=tf.float64)  # Relative risk aversion coefficient, assumed here to be 2.0

# Assuming the first investment observation is the initial capital stock
k_0 = data['i_obs'].iloc[0]
k_t_series = [k_0]
    
# Compute the capital
for i_t in data['i_obs'][1:]:  
   k_t_plus_1 = i_t + (1 - delta) * k_t_series[-1]
   k_t_series.append(k_t_plus_1)
    
k_t = pd.DataFrame({'k_t': k_t_series})
    

# Convert
c_t_tensor = tf.convert_to_tensor(c_t, dtype=tf.float64)
i_t_tensor = tf.convert_to_tensor(i_t, dtype=tf.float64)
k_t_tensor = tf.convert_to_tensor(k_t, dtype=tf.float64)
y_t_tensor = tf.convert_to_tensor(y_t, dtype=tf.float64)

# Prior distribution
#alpha
alpha_prior = tfd.TruncatedNormal(
    loc=tf.constant(0.3, dtype=tf.float64), 
    scale=tf.constant(0.025, dtype=tf.float64), 
    low=tf.constant(0.2, dtype=tf.float64), 
    high=tf.constant(0.5, dtype=tf.float64)
)

#beta_draw
beta_draw_prior = tfd.Gamma(
    concentration=tf.constant(2.5, dtype=tf.float64), 
    rate=tf.constant(10.0, dtype=tf.float64)  
)

#rho
mean = 0.5
std_dev = 0.2
variance = std_dev ** 2

alpha = mean * ((mean * (1 - mean) / variance) - 1)
beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1)

rho_prior = tfd.Beta(
    concentration0=tf.constant(alpha, dtype=tf.float64), 
    concentration1=tf.constant(beta, dtype=tf.float64)
)

#Define the joint log probability function
def joint_log_prob(c_t_tensor, i_t_tensor, k_t_tensor, y_t_tensor, alpha, beta_draw, rho):
    beta = 1 / (1 + beta_draw / 100)
    rho = rho / 100  # Assuming rho is given as a percentage, convert to actual value

    # Model
    # Epstein-Zin
    equation1 = tf.pow(c_t_tensor[:-1], -psi) - beta * tf.pow(alpha * tf.exp(k_t_tensor[1:] * (alpha - 1)) / c_t_tensor[1:], psi - 1)

    equation2 = c_t_tensor + i_t_tensor - y_t_tensor
    equation3 = y_t_tensor - tf.exp(k_t_tensor * alpha)
    # Here we multiply sigma by the random term
    equation4 = k_t_tensor[1:] - rho * k_t_tensor[:-1] - sigma * tf.random.normal(shape=k_t_tensor[1:].shape, dtype=tf.float64)

    # likelihood
    log_lik = -tf.reduce_sum(tf.square(equation1)) - tf.reduce_sum(tf.square(equation2)) - tf.reduce_sum(tf.square(equation3)) - tf.reduce_sum(tf.square(equation4))

    # Calculate the log probability of the prior
    rv_alpha = alpha_prior.log_prob(alpha)
    rv_beta_draw = beta_draw_prior.log_prob(beta_draw)
    rv_rho = rho_prior.log_prob(rho)

    return log_lik + rv_alpha + rv_beta_draw + rv_rho

# MCMC using Random Walk Metropolis-Hastings
number_of_steps = 1000
burnin = 300

# Define the RWMH kernel
rwmh_kernel = tfp.mcmc.RandomWalkMetropolis(
    target_log_prob_fn=lambda alpha, beta_draw, rho: joint_log_prob(c_t_tensor, i_t_tensor, k_t_tensor, y_t_tensor, alpha, beta_draw, rho)
)


# Initial state remains the same
initial_state = [
    tf.convert_to_tensor(0.3, dtype=tf.float64),
    tf.convert_to_tensor(0.25, dtype=tf.float64),
    tf.convert_to_tensor(0.5, dtype=tf.float64)
]

# Sampling function for RWMH with the seed passed to sample_chain
@tf.autograph.experimental.do_not_convert
@tf.function
def run_chain_rwmh():
    return tfp.mcmc.sample_chain(
        num_results=number_of_steps,
        current_state=initial_state,
        kernel=rwmh_kernel,
        num_burnin_steps=burnin,
        trace_fn=lambda _, pkr: pkr.is_accepted,
        seed=42  
    )

# Execute the MCMC chain with RWMH
start_time = time.time()
samples_rwmh, is_accepted_rwmh = run_chain_rwmh()
alpha_samples_rwmh, beta_draw_samples_rwmh, rho_samples_rwmh = samples_rwmh
end_time = time.time()

# Calculate means and stds as before
alpha_posterior_mean_rwmh = tf.reduce_mean(alpha_samples_rwmh).numpy()
alpha_posterior_std_rwmh = tf.math.reduce_std(alpha_samples_rwmh).numpy()
beta_draw_posterior_mean_rwmh = tf.reduce_mean(beta_draw_samples_rwmh).numpy()
beta_draw_posterior_std_rwmh = tf.math.reduce_std(beta_draw_samples_rwmh).numpy()
rho_posterior_mean_rwmh = tf.reduce_mean(rho_samples_rwmh).numpy()
rho_posterior_std_rwmh = tf.math.reduce_std(rho_samples_rwmh).numpy()

total_time_seconds_rwmh = end_time - start_time

print("Posterior means of parameters using RWMH:")
print("Alpha: Mean =", alpha_posterior_mean_rwmh, "Std =", alpha_posterior_std_rwmh)
print("Beta Draw: Mean =", beta_draw_posterior_mean_rwmh, "Std =", beta_draw_posterior_std_rwmh)
print("Rho: Mean =", rho_posterior_mean_rwmh, "Std =", rho_posterior_std_rwmh)
print("Total Time (seconds):", total_time_seconds_rwmh)