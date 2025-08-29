import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# System Parameters
# -----------------------------
M = 1.0     # Cart mass (kg)
m = 0.1     # Pendulum mass (kg)
l = 0.5     # Pendulum length (m)
g = 9.81    # Gravity (m/s^2)
dt = 0.01   # Time step (s)

# -----------------------------
# Dynamics of cart-pendulum
# -----------------------------
def dynamics(state, F):
    x, x_dot, theta, theta_dot = state
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    denom = M + m * (sin_theta**2)
    
    x_ddot = (F + m*sin_theta*(l*theta_dot**2 + g*cos_theta)) / denom
    theta_ddot = (-F*cos_theta - m*l*theta_dot**2*sin_theta*cos_theta - (M+m)*g*sin_theta) / (l*denom)
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

# -----------------------------
# PID Controller
# -----------------------------
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp*error + self.Ki*self.integral + self.Kd*derivative

# -----------------------------
# Simulation
# -----------------------------
# PID gains (tuned manually)
pid = PID(Kp=100, Ki=1, Kd=20)

# Initial state: cart at 0, pendulum tilted
state = np.array([0.0, 0.0, 0.1, 0.0])  # [x, x_dot, theta(rad), theta_dot]
T = 10  # total simulation time (s)
time = np.arange(0, T, dt)
states = []

for t in time:
    error = state[2]  # control based on pendulum angle (theta)
    F = -pid.control(error, dt)  # negative feedback
    deriv = dynamics(state, F)
    state = state + deriv * dt
    states.append(state)

states = np.array(states)

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(time, states[:,2], label="Pendulum Angle (rad)")
plt.axhline(0, color='k', linestyle='--')
plt.ylabel("Theta (rad)")
plt.legend()

plt.subplot(2,1,2)
plt.plot(time, states[:,0], label="Cart Position (m)")
plt.axhline(0, color='k', linestyle='--')
plt.ylabel("Cart Position (m)")
plt.xlabel("Time (s)")
plt.legend()

plt.suptitle("Inverted Pendulum Stabilisation with PID")
plt.tight_layout()
plt.show()
