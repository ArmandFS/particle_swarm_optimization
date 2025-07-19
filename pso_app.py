import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#define the same functions as before
def objective_function(params):
    x, y, z = params[0], params[1], params[2]
    return (x - 4)**2 + (y - 5)**2 + (z + 6)**2

st.title("Particle Swarm Optimization (PSO)")

#Sidebar input
n_particles = st.sidebar.slider("Number of particles", 5, 50, 10)
max_iter = st.sidebar.slider("Number of iterations", 10, 100, 30)
w = st.sidebar.slider("Inertia weight (w)", 0.0, 1.0, 0.5)
c1 = st.sidebar.slider("Cognitive coefficient (c1)", 0.0, 2.0, 0.8)
c2 = st.sidebar.slider("Social coefficient (c2)", 0.0, 2.0, 0.9)

#initialize Bounds in space
bounds = np.array([[-10, -10, -10], [10, 10, 10]])

#Initialize variables 
particles = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_particles, 3))
velocities = np.zeros_like(particles)
best_positions = particles.copy()
best_costs = np.array([objective_function(p) for p in particles])
global_best_index = np.argmin(best_costs)
global_best_position = best_positions[global_best_index].copy()
global_best_cost = best_costs[global_best_index]

#Store progress
cost_history = []

#Run PSO by using a for loop
for i in range(max_iter):
    r1 = np.random.rand(n_particles, 3)
    r2 = np.random.rand(n_particles, 3)

    cognitive = c1 * r1 * (best_positions - particles)
    social = c2 * r2 * (global_best_position - particles)
    velocities = w * velocities + cognitive + social
    particles += velocities
    particles = np.clip(particles, bounds[0], bounds[1])

    costs = np.array([objective_function(p) for p in particles])
    is_better = costs < best_costs
    best_positions[is_better] = particles[is_better]
    best_costs[is_better] = costs[is_better]

    global_best_index = np.argmin(best_costs)
    global_best_position = best_positions[global_best_index].copy()
    global_best_cost = best_costs[global_best_index]
    
    cost_history.append(global_best_cost)
    
st.subheader("Particle Positions in 3D (Final Iteration)")

fig = go.Figure(data=[go.Scatter3d(
    x=particles[:, 0],
    y=particles[:, 1],
    z=particles[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        opacity=0.7
    )
)])

# Add the global best as a red point
fig.add_trace(go.Scatter3d(
    x=[global_best_position[0]],
    y=[global_best_position[1]],
    z=[global_best_position[2]],
    mode='markers',
    marker=dict(
        size=8,
        color='red',
        symbol='cross',
        opacity=1.0
    ),
    name='Global Best'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(title='X', backgroundcolor='white', gridcolor='black', color='black'),
        yaxis=dict(title='Y', backgroundcolor='white', gridcolor='black', color='black'),
        zaxis=dict(title='Z', backgroundcolor='white', gridcolor='black', color='black')
    ),
    paper_bgcolor='white',
    plot_bgcolor='black',
    width=700,
    margin=dict(r=20, l=10, b=10, t=10)
)

st.plotly_chart(fig)

#output will be here
st.subheader("Best Solution Found!")
st.write("Position:", global_best_position)
st.write("Cost:", global_best_cost)

#plot the output
st.subheader("Cost Over Iterations")
fig, ax = plt.subplots()
ax.plot(range(1, max_iter + 1), cost_history)
ax.set_xlabel("Iteration")
ax.set_ylabel("Best Cost")
st.pyplot(fig)