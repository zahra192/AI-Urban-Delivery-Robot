# 🤖 AI Urban Delivery Robot Simulator

A Python-based intelligent path planning simulator that models a delivery robot navigating a dynamic city grid. The robot uses classic AI search algorithms to find optimal routes, avoid obstacles, and deliver packages — all visualized in real time.

---

## 🎯 Project Overview

This project simulates an urban delivery robot operating in a randomly generated city environment. The robot must navigate roads, avoid buildings, manage traffic zones, and reach delivery points — demonstrating how different AI algorithms perform under the same conditions.

---

## ✨ Features

- 🗺️ **Dynamic Grid Generation** — Every new map is randomly generated with roads, buildings, traffic zones, and delivery points
- 🧠 **5 AI Algorithms** — BFS, DFS, UCS, Greedy, and A* all implemented from scratch
- 🤖 **Real-time Animation** — Watch the robot explore and move step by step
- 📊 **Live Performance Stats** — Track cost, nodes explored, and execution time per delivery
- 🎮 **Interactive GUI** — Built with Tkinter; switch algorithms and speed on the fly
- 📦 **5 Deliveries Per Run** — Robot completes all deliveries in sequence from base station

---

## 🧠 Algorithms Implemented

| Algorithm | Strategy | Optimal? | Complete? | Cost-Aware? |
|-----------|----------|----------|-----------|-------------|
| **BFS** | Level-by-level exploration | ✅ (steps) | ✅ | ❌ |
| **DFS** | Deep path first | ❌ | ❌ | ❌ |
| **UCS** | Minimum cumulative cost | ✅ | ✅ | ✅ |
| **Greedy** | Heuristic (Euclidean) only | ❌ | ❌ | ❌ |
| **A\*** | g(n) + h(n) — Manhattan | ✅ | ✅ | ✅ |

> A* uses Manhattan distance as heuristic — best suited for grid-based movement.

---

## 🗂️ Project Structure

```
AI-Urban-Delivery-Robot/
│
├── main.py          # Entry point — launches the application
├── grid.py          # Grid environment, cell types, cost assignment
├── algorithms.py    # BFS, DFS, UCS, Greedy, A* implementations
├── gui.py           # Tkinter GUI — canvas, sidebar, animation
└── .gitignore
```

---

## 🌆 Grid Environment

| Cell Type | Description | Movement Cost |
|-----------|-------------|---------------|
| 🟫 Road | Normal path | 1–5 |
| ⬛ Building | Obstacle — blocked | ❌ Not traversable |
| 🟧 Traffic Zone | Slow path | 10–20 |
| 🟦 Delivery Point | Package destination | 1–5 |
| 🟪 Base Station | Robot start point | 1–5 |

---

## ▶️ How to Run

**Requirements:** Python 3.x (Tkinter included)

```bash
# Clone the repository
git clone https://github.com/zahra192/AI-Urban-Delivery-Robot.git

# Navigate to folder
cd AI-Urban-Delivery-Robot

# Run the simulator
python main.py
```

---

## 🖥️ How to Use

1. Launch the app with `python main.py`
2. Select an algorithm from the sidebar (BFS, DFS, UCS, Greedy, A*)
3. Adjust animation speed using the slider
4. Click **▶️ Start Deliveries** to begin
5. Click **↺ New Map** to generate a fresh random grid

---

## 📊 Performance Comparison

Each algorithm is evaluated on:
- **Total Path Cost** — sum of movement costs along the path
- **Nodes Explored** — how many cells were visited before reaching goal
- **Execution Time** — in milliseconds

> A* consistently finds the optimal path while exploring the fewest nodes.

---

## 📚 Course

**Artificial Intelligence** — BS Computer Science  
Phase 1 Project

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **GUI:** Tkinter
- **Data Structures:** heapq, deque, dict
- **Concepts:** Graph Search, Heuristics, Priority Queues