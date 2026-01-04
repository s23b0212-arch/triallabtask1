import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset (Hardcoded from your data)
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
def euclidean(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# -------------------------------
# ACO Algorithm for VRP
# -------------------------------
def aco_vrp(num_ants, iterations, alpha, beta, evaporation):
    nodes = list(CUSTOMERS.index)
    pheromone = {(i, j): 1.0 for i in nodes for j in nodes if i != j}

    best_distance = float("inf")
    best_routes = None
    convergence = []

    for _ in range(iterations):
        solutions = []

        for _ in range(num_ants):
            unvisited = nodes.copy()
            routes = []
            total_distance = 0

            while unvisited:
                load = 0
                route = []
                current = None

                while True:
                    feasible = [
                        j for j in unvisited
                        if load + CUSTOMERS.loc[j, "demand"] <= VEHICLE_CAPACITY
                    ]
                    if not feasible:
                        break

                    probs = []
                    for j in feasible:
                        tau = pheromone.get((current, j), 1.0)
                        prev = (
                            (DEPOT["x"], DEPOT["y"])
                            if current is None
                            else (CUSTOMERS.loc[current, "x"], CUSTOMERS.loc[current, "y"])
                        )
                        dist = euclidean(prev, (CUSTOMERS.loc[j, "x"], CUSTOMERS.loc[j, "y"]))
                        eta = 1 / (dist + 1e-6)
                        probs.append((j, (tau ** alpha) * (eta ** beta)))

                    total = sum(p[1] for p in probs)
                    probs = [(p[0], p[1] / total) for p in probs]
                    next_node = random.choices(
                        [p[0] for p in probs],
                        [p[1] for p in probs]
                    )[0]

                    route.append(next_node)
                    unvisited.remove(next_node)
                    load += CUSTOMERS.loc[next_node, "demand"]
                    current = next_node

                routes.append(route)

            # Calculate distance
            for r in routes:
                prev = (DEPOT["x"], DEPOT["y"])
                for c in r:
                    curr = (CUSTOMERS.loc[c, "x"], CUSTOMERS.loc[c, "y"])
                    total_distance += euclidean(prev, curr)
                    prev = curr
                total_distance += euclidean(prev, (DEPOT["x"], DEPOT["y"]))

            solutions.append((routes, total_distance))

        # Evaporation
        for key in pheromone:
            pheromone[key] *= (1 - evaporation)

        best_iter = min(solutions, key=lambda x: x[1])
        if best_iter[1] < best_distance:
            best_distance = best_iter[1]
            best_routes = best_iter[0]

        for r in best_iter[0]:
            for i in range(len(r) - 1):
                pheromone[(r[i], r[i + 1])] += 1 / best_iter[1]

        convergence.append(best_distance)

    return best_routes, best_distance, convergence

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🐜 Ant Colony Optimization for VRP")
st.write("**Dataset-based VRP | Algorithm: ACO**")

st.sidebar.header("ACO Parameters")
num_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
iterations = st.sidebar.slider("Iterations", 20, 300, 100)
alpha = st.sidebar.slider("Alpha (pheromone)", 0.5, 3.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic)", 0.5, 5.0, 2.0)
evaporation = st.sidebar.slider("Evaporation Rate", 0.1, 0.9, 0.3)

if st.button("Run ACO"):
    routes, best_dist, convergence = aco_vrp(
        num_ants, iterations, alpha, beta, evaporation
    )

    st.success(f"Best Total Distance: {best_dist:.4f}")
    st.write(f"Vehicles Used: {len(routes)}")

    # Convergence Plot
    st.subheader("Convergence Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(convergence)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Distance")
    st.pyplot(fig1)

    # Route Visualization
    st.subheader("Vehicle Routes")
    fig2, ax2 = plt.subplots()

    ax2.scatter(DEPOT["x"], DEPOT["y"], c="red", s=120, label="Depot")

    for _, row in CUSTOMERS.iterrows():
        ax2.scatter(row["x"], row["y"])
        ax2.text(row["x"] + 0.01, row["y"] + 0.01, str(row.name))

    for r in routes:
        path = [(DEPOT["x"], DEPOT["y"])]
        for c in r:
            path.append((CUSTOMERS.loc[c, "x"], CUSTOMERS.loc[c, "y"]))
        path.append((DEPOT["x"], DEPOT["y"]))
        xs, ys = zip(*path)
        ax2.plot(xs, ys, marker="o")

    ax2.legend()
    st.pyplot(fig2)
