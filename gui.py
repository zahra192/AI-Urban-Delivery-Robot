import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading

from grid import (Grid, GRID_SIZE, CELL_SIZE, PADDING,
                  ROAD, BUILDING, TRAFFIC, DELIVERY, BASE,
                  PATH, EXPLORED, ROBOT, COLORS, LABEL_COLORS)
from algorithms import ALGORITHMS


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🤖 Intelligent Urban Delivery Robot")
        self.configure(bg="#0d0f1a")
        self.resizable(False, False)
        self.geometry("+100+100")

        self.grid_env      = Grid()
        self.display_cells = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.robot_pos     = self.grid_env.base
        self.deliveries    = list(self.grid_env.deliveries)
        self.delivery_num  = 0
        self.running       = False
        self.algo_var      = tk.StringVar(value="A*")
        self.speed_var     = tk.IntVar(value=60)

        self._build_ui()
        self._draw_grid()
        self._update_status("Ready. Select algorithm and click ▶️ Start Deliveries.")

    def _build_ui(self):
        hdr = tk.Frame(self, bg="#0d0f1a")
        hdr.pack(fill="x", padx=14, pady=(12,4))
        tk.Label(hdr, text="URBAN DELIVERY ROBOT", font=("Courier New", 17, "bold"),
                 fg="#f0a500", bg="#0d0f1a").pack(side="left")
        tk.Label(hdr, text="AI Path Planning Simulator", font=("Courier New", 9),
                 fg="#555e8a", bg="#0d0f1a").pack(side="left", padx=10, pady=6)

        main = tk.Frame(self, bg="#0d0f1a")
        main.pack(padx=14, pady=4)

        canvas_size = GRID_SIZE * CELL_SIZE + 2*PADDING
        self.canvas = tk.Canvas(main, width=canvas_size, height=canvas_size,
                                bg="#0d0f1a", highlightthickness=2,
                                highlightbackground="#f0a500")
        self.canvas.pack(side="left", padx=(0,14))

        side = tk.Frame(main, bg="#12152b", bd=0)
        side.pack(side="left", fill="y")
        self._build_sidebar(side)

        log_frame = tk.Frame(self, bg="#0d0f1a")
        log_frame.pack(fill="x", padx=14, pady=(4,12))
        tk.Label(log_frame, text="DELIVERY LOG", font=("Courier New", 8, "bold"),
                 fg="#555e8a", bg="#0d0f1a").pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=7, bg="#0b0d1a", fg="#a0aac8",
                                font=("Courier New", 9), relief="flat",
                                insertbackground="#f0a500", state="disabled",
                                wrap="word")
        self.log_text.pack(fill="x")
        sb = tk.Scrollbar(log_frame, command=self.log_text.yview, bg="#12152b")
        sb.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb.set)

    def _build_sidebar(self, parent):
        pad = {"padx": 12, "pady": 5}
        tk.Label(parent, text="ALGORITHM", font=("Courier New", 9, "bold"),
                 fg="#f0a500", bg="#12152b").pack(anchor="w", **pad)
        for algo in ALGORITHMS:
            tk.Radiobutton(parent, text=algo, variable=self.algo_var, value=algo,
                           font=("Courier New", 10), fg="#c0c8e8", bg="#12152b",
                           selectcolor="#1e2540", activebackground="#12152b",
                           activeforeground="#f0a500", indicatoron=True
                           ).pack(anchor="w", padx=18, pady=1)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=8)

        tk.Label(parent, text="ANIMATION SPEED", font=("Courier New", 9, "bold"),
                 fg="#f0a500", bg="#12152b").pack(anchor="w", **pad)
        sf = tk.Frame(parent, bg="#12152b")
        sf.pack(fill="x", padx=12)
        tk.Label(sf, text="Slow", font=("Courier New", 8),
                 fg="#555e8a", bg="#12152b").pack(side="left")
        tk.Scale(sf, from_=200, to=10, orient="horizontal",
                 variable=self.speed_var, showvalue=False,
                 bg="#12152b", fg="#c0c8e8", troughcolor="#1e2540",
                 highlightthickness=0, length=120).pack(side="left")
        tk.Label(sf, text="Fast", font=("Courier New", 8),
                 fg="#555e8a", bg="#12152b").pack(side="left")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=8)

        btn_cfg = dict(font=("Courier New", 10, "bold"), relief="flat",
                       cursor="hand2", padx=8, pady=5)
        self.start_btn = tk.Button(parent, text="▶️  Start Deliveries",
                                   bg="#f0a500", fg="#0d0f1a",
                                   command=self._start, **btn_cfg)
        self.start_btn.pack(fill="x", padx=12, pady=3)
        tk.Button(parent, text="↺  New Map", bg="#1e2540", fg="#a0aac8",
                  command=self._new_map, **btn_cfg).pack(fill="x", padx=12, pady=3)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=8)

        tk.Label(parent, text="STATS", font=("Courier New", 9, "bold"),
                 fg="#f0a500", bg="#12152b").pack(anchor="w", **pad)
        self.stat_vars = {}
        for label, init in [("Delivery","—"),("Total Cost","—"),
                             ("Nodes Explored","—"),("Time (ms)","—")]:
            row = tk.Frame(parent, bg="#12152b")
            row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=label+":", font=("Courier New", 8),
                     fg="#555e8a", bg="#12152b", width=14, anchor="w").pack(side="left")
            var = tk.StringVar(value=init)
            tk.Label(row, textvariable=var, font=("Courier New", 9, "bold"),
                     fg="#1abc9c", bg="#12152b").pack(side="left")
            self.stat_vars[label] = var

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=8)

        tk.Label(parent, text="LEGEND", font=("Courier New", 9, "bold"),
                 fg="#f0a500", bg="#12152b").pack(anchor="w", **pad)
        for color, name in [(COLORS[BASE],"Base Station"),(COLORS[DELIVERY],"Delivery Point"),
                            (COLORS[BUILDING],"Building"),(COLORS[TRAFFIC],"Traffic Zone"),
                            (COLORS[PATH],"Optimal Path"),(COLORS[EXPLORED],"Explored Nodes"),
                            (COLORS[ROBOT],"Robot")]:
            row = tk.Frame(parent, bg="#12152b")
            row.pack(anchor="w", padx=12, pady=1)
            tk.Label(row, bg=color, width=2, relief="flat").pack(side="left", padx=(0,6))
            tk.Label(row, text=name, font=("Courier New", 8),
                     fg="#a0aac8", bg="#12152b").pack(side="left")

        self.status_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self.status_var, font=("Courier New", 8),
                 fg="#e74c3c", bg="#12152b", wraplength=190,
                 justify="left").pack(anchor="w", padx=12, pady=8)

    def _draw_grid(self):
        self.canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c, self.grid_env.cells[r][c])
        self._draw_robot(*self.robot_pos)

    def _draw_cell(self, r, c, cell_type, override_color=None):
        x1 = PADDING + c*CELL_SIZE
        y1 = PADDING + r*CELL_SIZE
        x2, y2 = x1+CELL_SIZE, y1+CELL_SIZE
        color = override_color if override_color else COLORS[cell_type]
        tag = f"cell_{r}_{c}"
        self.canvas.delete(tag)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                     outline="#0d0f1a", width=1, tags=tag)
        label_map = {BASE:"BASE", DELIVERY:"📦", BUILDING:"■", TRAFFIC:"⚠"}
        if cell_type in label_map:
            fg = LABEL_COLORS.get(cell_type, "#ffffff")
            self.canvas.create_text((x1+x2)//2, (y1+y2)//2,
                                    text=label_map[cell_type],
                                    font=("Courier New", 8, "bold"),
                                    fill=fg, tags=tag)

    def _draw_robot(self, r, c):
        self.canvas.delete("robot")
        x = PADDING + c*CELL_SIZE + CELL_SIZE//2
        y = PADDING + r*CELL_SIZE + CELL_SIZE//2
        rad = CELL_SIZE//2 - 4
        self.canvas.create_oval(x-rad, y-rad, x+rad, y+rad,
                                fill=COLORS[ROBOT], outline="#f0a500",
                                width=2, tags="robot")
        self.canvas.create_text(x, y, text="🤖", font=("Arial", 11), tags="robot")

    def _color_cell(self, r, c, cell_type):
        self._draw_cell(r, c, cell_type)

    def _update_status(self, msg):
        self.status_var.set(msg)

    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _new_map(self):
        if self.running:
            messagebox.showwarning("Busy", "Wait for current simulation to finish.")
            return
        self.grid_env = Grid()
        self.robot_pos = self.grid_env.base
        self.deliveries = list(self.grid_env.deliveries)
        self.delivery_num = 0
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        for k in self.stat_vars:
            self.stat_vars[k].set("—")
        self._draw_grid()
        self._update_status("New map generated. Ready.")

    def _start(self):
        if self.running:
            return
        self.deliveries = list(self.grid_env.deliveries)
        self.robot_pos  = self.grid_env.base
        self.delivery_num = 0
        self._draw_grid()
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.running = True
        self.start_btn.configure(state="disabled")
        threading.Thread(target=self._run_all_deliveries, daemon=True).start()

    def _run_all_deliveries(self):
        algo_name  = self.algo_var.get()
        algo_fn    = ALGORITHMS[algo_name]
        total_cost = 0
        for i, dest in enumerate(self.deliveries):
            self.delivery_num = i + 1
            self.after(0, self._update_status,
                       f"Delivery {i+1}/5 → ({dest[0]},{dest[1]}) using {algo_name}")
            self.after(0, self._log, f"\n── Delivery {i+1}/5 ── Algo: {algo_name} ──")
            self.after(0, self._log, f"   From {self.robot_pos} → {dest}")
            path, cost, explored, elapsed = algo_fn(self.grid_env, self.robot_pos, dest)
            if path is None:
                self.after(0, self._log, f"   ❌ No path found to {dest}!")
                self.after(0, self._update_status, f"No path to delivery {i+1}.")
                continue
            self._animate_explored(explored)
            self._animate_path(path)
            self._animate_robot(path)
            total_cost += cost
            self.robot_pos = dest
            self.after(0, self._log,
                       f"   ✅ Cost: {cost}  |  Nodes: {len(explored)}  |  "
                       f"Time: {elapsed*1000:.2f} ms")
            self.after(0, self.stat_vars["Delivery"].set,        f"{i+1} / 5")
            self.after(0, self.stat_vars["Total Cost"].set,      str(total_cost))
            self.after(0, self.stat_vars["Nodes Explored"].set,  str(len(explored)))
            self.after(0, self.stat_vars["Time (ms)"].set,       f"{elapsed*1000:.2f}")
            time.sleep(0.4)
        self.after(0, self._update_status,
                   f"✅ All 5 deliveries done! Total cost: {total_cost}")
        self.after(0, self._log,
                   f"\n🎉 All deliveries completed. Total cost: {total_cost}")
        self.after(0, self.start_btn.configure, {"state": "normal"})
        self.running = False

    def _animate_explored(self, explored):
        delay = max(5, self.speed_var.get() // 4)
        for r, c in explored:
            if self.grid_env.cells[r][c] in (ROAD, TRAFFIC):
                self.after(0, self._color_cell, r, c, EXPLORED)
            time.sleep(delay / 1000)

    def _animate_path(self, path):
        delay = max(10, self.speed_var.get() // 3)
        for r, c in path[1:-1]:
            self.after(0, self._color_cell, r, c, PATH)
            time.sleep(delay / 1000)

    def _animate_robot(self, path):
        delay = self.speed_var.get()
        prev  = path[0]
        for r, c in path[1:]:
            pr, pc = prev
            if self.grid_env.cells[pr][pc] not in (BASE, DELIVERY):
                self.after(0, self._color_cell, pr, pc, PATH)
            self.after(0, self._draw_robot, r, c)
            time.sleep(delay / 1000)
            prev = (r, c)
        dr, dc = path[-1]
        self.after(0, self._mark_delivered, dr, dc)

    def _mark_delivered(self, r, c):
        self.canvas.delete(f"cell_{r}_{c}")
        x1 = PADDING + c*CELL_SIZE
        y1 = PADDING + r*CELL_SIZE
        x2, y2 = x1+CELL_SIZE, y1+CELL_SIZE
        tag = f"cell_{r}_{c}"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#145a32",
                                     outline="#0d0f1a", width=1, tags=tag)
        self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text="✓",
                                font=("Courier New", 13, "bold"),
                                fill="#2ecc71", tags=tag)
        self._draw_robot(r, c)