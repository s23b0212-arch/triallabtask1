import streamlit as st
import numpy as np
import pandas as pd
import random

# -------------------------------
# Dataset
# -------------------------------
data = {
    "node_id": list(range(21)),
    "node_type": ["depot"] + ["customer"] * 20,
    "x": [
        0.5047266, 0.74249285, 0.3856868, 0.15904588, 0.43770394,
        0.6478709, 0.6376881, 0.050297342, 0.33051717, 0.5967194,
        0.65211993, 0.9405325, 0.7721941, 0.7314005, 0.38096368,
        0.33801734, 0.5327366, 0.7032734, 0.17540133, 0.5773808,
        0.090266675
    ],
    "y": [
        0.91304976, 0.48951122, 0.17313126, 0.57699025, 0.7376329,
        0.9100823, 0.50143725, 0.56456906, 0.51687074, 0.7492327,
        0.64236987, 0.099436976, 0.70185155, 0.622245, 0.00668953,
        0.95866734, 0.6609484, 0.75974447, 0.7630342, 0.119943485,
        0.92257214
    ],
    "demand": [
        0, 3, 4, 8, 9, 5, 4, 5, 3, 1,
        3, 2, 1, 1, 1, 2, 2, 4, 7, 8, 3
    ]
}

df = pd.DataFrame(data)
DEPOT = df[df["node_type"] == "depot"].iloc[0]
CUSTOMERS = df[df["node_type"] == "customer"]
VEHICLE_CAPACITY = 30

# -------------------------------
# Distance Function
# -------------------------------
def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# -------------------------------
# ACO Algorithm
# -------------------------------
def aco_vrp(ants, iterations):
    nodes = list(CUSTOMERS.index)
    pheromone = {(i, j): 1.0 for i in nodes for j in nodes if i != j}

    best_dist = float("inf")
    best_routes = []
    convergence = []

    for _ in range(iterations):
        for _ in range(ants):
            unvisited = nodes.copy()
            routes = []
            total_dist = 0

            while unvisited:
                load = 0
                route = []
                current = None

                while True:
                    feasible = [n for n in unvisited if load + CUSTOMERS.loc[n, "demand"] <= VEHICLE_CAPACITY]
                    if not feasible:
                        break

                    next_node = random.choice(feasible)
                    route.append(next_node)
                    unvisited.remove(next_node)
                    load += CUSTOMERS.loc[next_node, "demand"]
                    current = next_node

                routes.append(route)

            for r in routes:
                prev = (DEPOT.x, DEPOT.y)
                for c in r:
                    curr = (CUSTOMERS.loc[c, "x"], CUSTOMERS.loc[c, "y"])
                    total_dist += distance(prev, curr)
                    prev = curr
                total_dist += distance(prev, (DEPOT.x, DEPOT.y))

            if total_dist < best_dist:
                best_dist = total_dist
                best_routes = routes

        convergence.append(best_dist)

    return best_routes, best_dist, convergence

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🐜 Ant Colony Optimization (ACO) for VRP")
st.write("Dataset-based Vehicle Routing Problem")

ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
iterations = st.sidebar.slider("Iterations", 10, 200, 50)

if st.button("Run ACO"):
    routes, best_dist, convergence = aco_vrp(ants, iterations)

    st.success(f"Best Distance: {best_dist:.4f}")
    st.write(f"Vehicles Used: {len(routes)}")

    st.subheader("Convergence Curve")
    st.line_chart(convergence)

    st.subheader("Routes (Text Representation)")
    for i, r in enumerate(routes):
        st.write(f"Vehicle {i+1}: Depot → {r} → Depot")
