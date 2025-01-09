import re
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

ELEMENTARY_SIGNALS = {
    "Square Wave": "u(t) - u(t-1)",
    "Exponential (Positive)": "exp(t)",
    "Sine (0..2π)": "sin(t)*[u(t)-u(t-2*pi)]",
    "Negative Exponential": "exp(t)*u(-t)",
    "Delta": "delta(t)",
    "Box(0..0.5)": "u(t)-u(t-0.5)",
    "Triangle Wave": "triangle(t)",
    "Sawtooth Wave": "sawtooth(t)",
    "Cosine Wave": "cos(t)*u(t)",
    "Linear Ramp": "t*u(t)",
    "Sinc": "sinc(t)",
    "Sin": "sin(t)",
    "Cos": "cos(t)",#
    "Shifted Step": "u(t-2)"
}


def preprocess_expression(expr):
    expr = re.sub(r'(\d)(pi|\w+\()', r'\1*\2', expr)
    expr = re.sub(r'(pi)(\w+\()', r'\1*\2', expr)
    return expr


def eval_signal_function(expr, t_or_n, mode):
    try:
        expr = preprocess_expression(expr)
        if mode == "Discrete Time (n)":
            u = lambda x: np.where(x >= 0, 1.0, 0.0)
            delta = lambda x: np.where(x == 0, 1.0, 0.0)
        else:
            u = lambda x: np.heaviside(x, 1)
            delta = lambda x: np.where(np.isclose(x, 0, atol=1e-14), 1.0, 0.0)

        def box(x):
            T = 0.5
            return ((x > -T) & (x < T)).astype(float)

        def step(x):
            return (x > 0).astype(float)

        def triangle(x):
            return ((np.abs(x) < 1) * (1 - np.abs(x))).astype(float)

        def sawtooth(x):
            return (x - np.floor(x)).astype(float)

        allowed_names = {
            't': t_or_n, 'n': t_or_n,
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'pi': np.pi,
            'u': u, 'delta': delta, 'box': box, 'triangle': triangle, 'sawtooth': sawtooth,
            'sinc': np.sinc,
            'e': np.e
        }
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return np.array(result, dtype=float)
    except Exception as e:
        print(f"Hata: {e}")
        return np.zeros_like(t_or_n)


def parse_signal_input(text):
    text = text.strip()
    if not text:
        return None, None
    pairs = text.split('],')
    n_list = []
    val_list = []
    for p in pairs:
        p = p.strip().strip('[]')
        if not p:
            continue
        parts = p.split(',')
        if len(parts) != 2:
            continue
        try:
            n_val = float(parts[0].strip())
            v_val = float(parts[1].strip())
            n_list.append(n_val)
            val_list.append(v_val)
        except:
            pass
    if len(n_list) == 0:
        return None, None
    return np.array(n_list), np.array(val_list)


class SinyalIslemeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Application")


        self.auto_zoom_enabled = False
        self.manual_xlims = None
        self.manual_ylims = None

        self.is_drawing = False
        self.last_x = None
        self.current_ax = None

        self.is_panning = False
        self.pan_start_x = None
        self.original_xlims = None

        # SHIFT
        self.input1_shift = 0
        self.input2_shift = 0
        self.original_input1_data = []
        self.original_input2_data = []

        self.last_operation = None
        self.last_data1 = None
        self.last_data2 = None
        self.last_mode = None

        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(top_frame, text="Time Mode:").pack(side=tk.LEFT, padx=5)
        self.time_mode_selector = ttk.Combobox(top_frame, values=["Continuous Time (t)", "Discrete Time (n)"],
                                               state="readonly")
        self.time_mode_selector.current(0)
        self.time_mode_selector.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Operation:").pack(side=tk.LEFT, padx=5)
        self.operation_box = ttk.Combobox(top_frame,
                                          values=["Addition", "Subtraction", "Multiplication", "Convolution"],
                                          state="readonly")
        self.operation_box.current(0)
        self.operation_box.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Time Range (start, end):").pack(side=tk.LEFT, padx=5)
        self.start_time_input = tk.Entry(top_frame, width=5)
        self.start_time_input.insert(0, "-5")
        self.start_time_input.pack(side=tk.LEFT, padx=5)
        self.end_time_input = tk.Entry(top_frame, width=5)
        self.end_time_input.insert(0, "5")
        self.end_time_input.pack(side=tk.LEFT, padx=5)

        self.apply_op_button = tk.Button(top_frame, text="Apply", command=self.apply_operation)
        self.apply_op_button.pack(side=tk.LEFT, padx=5)

        self.clear_input_button = tk.Button(top_frame, text="Clear Inputs", command=self.clear_inputs)
        self.clear_input_button.pack(side=tk.LEFT, padx=5)

        wave_frame = tk.Frame(root)
        wave_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(wave_frame, text="Basic Wave:").pack(side=tk.LEFT, padx=5)
        self.elem_signal_combo = ttk.Combobox(wave_frame, values=list(ELEMENTARY_SIGNALS.keys()), state="readonly")
        self.elem_signal_combo.pack(side=tk.LEFT, padx=5)
        tk.Button(wave_frame, text="Apply to Input1", command=lambda: self.apply_elem_signal(1)).pack(side=tk.LEFT, padx=5)
        tk.Button(wave_frame, text="Apply to Input2", command=lambda: self.apply_elem_signal(2)).pack(side=tk.LEFT, padx=5)

        func_frame = tk.Frame(root)
        func_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(func_frame, text="Manual Function Input (e.g. sin(t)*u(t)):").pack(side=tk.LEFT, padx=5)
        self.func_input = tk.Entry(func_frame, width=30)
        self.func_input.pack(side=tk.LEFT, padx=5)
        tk.Button(func_frame, text="Plot on Input1", command=lambda: self.set_input_func(1)).pack(side=tk.LEFT, padx=5)
        tk.Button(func_frame, text="Plot on Input2", command=lambda: self.set_input_func(2)).pack(side=tk.LEFT, padx=5)

        conv_frame = tk.Frame(root, bd=2, relief=tk.GROOVE)
        conv_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(conv_frame, text="Convolution (Discrete):").pack(side=tk.LEFT, padx=5)
        self.x_conv_input = tk.Entry(conv_frame, width=50)
        self.x_conv_input.insert(0, "[0,1],[1,2],[2,1]")
        self.x_conv_input.pack(side=tk.LEFT, padx=5)
        self.h_conv_input = tk.Entry(conv_frame, width=50)
        self.h_conv_input.insert(0, "[0,1],[1,1],[2,1]")
        self.h_conv_input.pack(side=tk.LEFT, padx=5)

        self.start_conv_button = tk.Button(conv_frame, text="Start Convolution", command=self.start_convolution)
        self.start_conv_button.pack(side=tk.LEFT, padx=5)

        conv_slider_frame = tk.Frame(root)
        conv_slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(conv_slider_frame, text="Convolution Animation:").pack(side=tk.LEFT, padx=5)
        self.conv_slider = tk.Scale(conv_slider_frame, from_=0, to=10, orient=tk.HORIZONTAL, length=300)
        self.conv_slider.pack(side=tk.LEFT, padx=5)
        self.conv_slider.bind("<B1-Motion>", lambda e: self.update_convolution_animation())
        self.conv_slider.bind("<ButtonRelease-1>", lambda e: self.update_convolution_animation())

        shift_frame = tk.Frame(root)
        shift_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(shift_frame, text="Input1 Shift:").pack(side=tk.LEFT)
        self.input1_shiftbar = tk.Scale(shift_frame, from_=-50, to=50, orient=tk.HORIZONTAL, length=150,
                                        command=self.on_input1_shift)
        self.input1_shiftbar.set(0)
        self.input1_shiftbar.pack(side=tk.LEFT, padx=5)

        tk.Label(shift_frame, text="Input2 Shift:").pack(side=tk.LEFT)
        self.input2_shiftbar = tk.Scale(shift_frame, from_=-50, to=50, orient=tk.HORIZONTAL, length=150,
                                        command=self.on_input2_shift)
        self.input2_shiftbar.set(0)
        self.input2_shiftbar.pack(side=tk.LEFT, padx=5)


        self.fig = Figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, figure=self.fig)

        self.ax_in1 = self.fig.add_subplot(gs[0, 0])
        self.ax_in2 = self.fig.add_subplot(gs[0, 1])
        self.ax_out = self.fig.add_subplot(gs[1, :])  # İkinci satırda tüm sütunları kaplar

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.cid_scroll = self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.input1_data = []
        self.input2_data = []
        self.out_signal = None

        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.cid_move = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        self.conv_data = {
            'x_n': None, 'x_vals': None,
            'h_n': None, 'h_vals': None,
            'y_n': None, 'y_vals': None,
            'is_discrete': True
        }

        self.update_plots()

    # SHIFT
    def on_input1_shift(self, val):
        self.auto_zoom_enabled = False
        self.input1_shift = int(val)
        self.reapply_operation_and_update()

    def on_input2_shift(self, val):
        self.auto_zoom_enabled = False
        self.input2_shift = int(val)
        self.reapply_operation_and_update()

    def reapply_operation_and_update(self):
        if self.last_operation and self.last_data1 and self.last_data2:
            op = self.last_operation
            mode = self.last_mode
            d1 = self.apply_shift(self.last_data1, self.input1_shift)
            d2 = self.apply_shift(self.last_data2, self.input2_shift)
            self.input1_data = d1
            self.input2_data = d2
            self.perform_operation(op, mode, d1, d2)
        else:
            self.update_plots()

    def apply_shift(self, data, shift_val):
        return [(p[0] + shift_val, p[1]) for p in data]

    # CLEAR
    def clear_inputs(self):
        self.auto_zoom_enabled = False
        self.input1_data = []
        self.input2_data = []
        self.original_input1_data = []
        self.original_input2_data = []
        self.out_signal = None
        self.input1_shiftbar.set(0)
        self.input2_shiftbar.set(0)
        self.input1_shift = 0
        self.input2_shift = 0
        self.last_operation = None
        self.last_data1 = None
        self.last_data2 = None
        self.last_mode = None

        # reset zoom and pan setting
        self.manual_xlims = None
        self.manual_ylims = None

        self.update_plots()

    # MANUAL FUNCTION
    def set_input_func(self, which):
        self.auto_zoom_enabled = False
        func_str = self.func_input.get().strip()
        if not func_str:
            return
        mode = self.time_mode_selector.get()
        try:
            st = float(self.start_time_input.get())
            en = float(self.end_time_input.get())
        except:
            st, en = -5, 5
        if mode == "Continuous Time (t)":
            t = np.linspace(st, en, 500)
            x = eval_signal_function(func_str, t, mode)
            data = list(zip(t, x))
        else:
            st_i = int(st)
            en_i = int(en)
            n = np.arange(st_i, en_i + 1)
            x = eval_signal_function(func_str, n, mode)
            data = list(zip(n, x))

        if which == 1:
            self.input1_data = data
            self.original_input1_data = data[:]
        else:
            self.input2_data = data
            self.original_input2_data = data[:]
        self.out_signal = None
        self.last_operation = None
        self.last_data1 = None
        self.last_data2 = None
        self.last_mode = None
        self.update_plots()

    def apply_elem_signal(self, which):
        self.auto_zoom_enabled = False
        secilen = self.elem_signal_combo.get()
        if not secilen or secilen not in ELEMENTARY_SIGNALS:
            return
        expr = ELEMENTARY_SIGNALS[secilen]
        self.func_input.delete(0, tk.END)
        self.func_input.insert(0, expr)
        self.set_input_func(which)

    # OPERATION
    def apply_operation(self):
        self.auto_zoom_enabled = True
        op = self.operation_box.get()
        mode = self.time_mode_selector.get()
        d1 = self.input1_data
        d2 = self.input2_data
        if len(d1) == 0 or len(d2) == 0:
            messagebox.showinfo("Info", "Please create both input signals first.")
            return

        self.last_operation = op
        self.last_mode = mode
        self.last_data1 = self.original_input1_data if len(self.original_input1_data) > 0 else d1
        self.last_data2 = self.original_input2_data if len(self.original_input2_data) > 0 else d2
        self.perform_operation(op, mode, d1, d2)

    def perform_operation(self, op, mode, d1, d2):
        x1 = np.array([p[0] for p in d1])
        y1 = np.array([p[1] for p in d1])
        x2 = np.array([p[0] for p in d2])
        y2 = np.array([p[1] for p in d2])

        if op == "Convolution":
            if mode == "Discrete Time (n)":
                yy = np.convolve(y1, y2, 'full')
                n_min = x1.min() + x2.min()
                n_max = x1.max() + x2.max()
                n_out = np.arange(n_min, n_max + 1)
                self.conv_data = {
                    'x_n': x1, 'x_vals': y1,
                    'h_n': x2, 'h_vals': y2,
                    'y_n': n_out, 'y_vals': yy,
                    'is_discrete': True
                }
                self.conv_slider.config(from_=0, to=len(yy) - 1)
                self.out_signal = None
                self.update_convolution_animation()
            else:
                t_min = min(x1.min(), x2.min())
                t_max = max(x1.max(), x2.max())
                t = np.linspace(t_min * 2, t_max * 2, 500)
                dt = t[1] - t[0]
                y1i = np.interp(t, x1, y1, left=0, right=0)
                y2i = np.interp(t, x2, y2, left=0, right=0)
                yy = np.convolve(y1i, y2i, 'full') * dt
                t_out = np.linspace(t_min + t_min, t_max + t_max, len(yy))
                self.conv_data = {
                    'x_n': t, 'x_vals': y1i,
                    'h_n': t, 'h_vals': y2i,
                    'y_n': t_out, 'y_vals': yy,
                    'is_discrete': False
                }
                self.conv_slider.config(from_=0, to=len(yy) - 1)
                self.out_signal = None
                self.update_convolution_animation()
        else:
            # addition / subtraction / multiply
            if mode == "Discrete Time (n)":
                alln = sorted(set(x1.tolist() + x2.tolist()))

                def val_at(nn, xx, yy):
                    if nn in xx:
                        i = xx.tolist().index(nn)
                        return yy[i]
                    return 0

                out = []
                for nn in alln:
                    v1 = val_at(nn, x1, y1)
                    v2 = val_at(nn, x2, y2)
                    if op == "Addition":
                        out.append(v1 + v2)
                    elif op == "Subtraction":
                        out.append(v1 - v2)
                    else:
                        out.append(v1 * v2)
                self.out_signal = {'t': alln, 'x': out, 'is_discrete': True}
            else:
                t_min = min(x1.min(), x2.min())
                t_max = max(x1.max(), x2.max())
                t_lin = np.linspace(t_min, t_max, 500)
                xx1 = np.interp(t_lin, x1, y1, left=0, right=0)
                xx2 = np.interp(t_lin, x2, y2, left=0, right=0)
                if op == "Addition":
                    out = xx1 + xx2
                elif op == "Subtraction":
                    out = xx1 - xx2
                else:
                    out = xx1 * xx2
                self.out_signal = {'t': t_lin, 'x': out, 'is_discrete': False}

            self.update_plots()

    def start_convolution(self):
        self.auto_zoom_enabled = False
        x_str = self.x_conv_input.get()
        h_str = self.h_conv_input.get()
        nx, xv = parse_signal_input(x_str)
        nh, hv = parse_signal_input(h_str)
        if nx is None or nh is None:
            messagebox.showerror("Error", "Invalid format! [n,amplitude]")
            return
        yy = np.convolve(xv, hv, 'full')
        nn = range(int(nx.min() + nh.min()), int(nx.max() + nh.max()) + 1)
        self.conv_data = {
            'x_n': nx, 'x_vals': xv,
            'h_n': nh, 'h_vals': hv,
            'y_n': np.array(list(nn)), 'y_vals': yy,
            'is_discrete': True
        }
        self.conv_slider.config(from_=0, to=len(yy) - 1)
        self.out_signal = None
        self.update_convolution_animation()

    def update_convolution_animation(self):
        val = self.conv_slider.get()
        cd = self.conv_data
        if cd['y_n'] is None:
            return

        self.ax_in1.clear()
        self.ax_in2.clear()
        self.ax_out.clear()

        if cd['is_discrete']:
            x_n, x_vals = cd['x_n'], cd['x_vals']
            h_n, h_vals = cd['h_n'], cd['h_vals']
            y_n, y_vals = cd['y_n'], cd['y_vals']
            k = y_n[val]
            self.ax_in1.stem(x_n, x_vals, linefmt='b-', markerfmt='bo', basefmt='k-')
            self.ax_in1.set_title("x[n]")
            h_rev = h_vals[::-1]
            h_rev_n = -h_n[::-1]
            h_shifted_n = h_rev_n + k
            self.ax_in2.stem(x_n, x_vals, linefmt='b-', markerfmt='bo', basefmt='k-')
            self.ax_in2.stem(h_shifted_n, h_rev, linefmt='g-', markerfmt='go', basefmt='k-')
            self.ax_in2.set_title(f"x[n]&h[-n], k={k}")

            partial_y = y_vals[:val + 1]
            partial_n = y_n[:val + 1]
            self.ax_out.stem(partial_n, partial_y, linefmt='r-', markerfmt='ro', basefmt='k-')
            self.ax_out.set_title(f"y[n] first {val + 1} steps")
        else:
            # C.T. animation
            t_out, y_vals = cd['y_n'], cd['y_vals']
            x_t, x_arr = cd['x_n'], cd['x_vals']
            h_t, h_arr = cd['h_n'], cd['h_vals']
            if val < len(y_vals):
                part_y = y_vals[:val + 1]
                part_t = t_out[:val + 1]
                self.ax_out.plot(part_t, part_y, 'r-')
                self.ax_out.set_title(f"y(t) first {val + 1} steps")
            else:
                self.ax_out.plot(t_out, y_vals, 'r-')
                self.ax_out.set_title("y(t)")

            # 1. axis => x(t)
            self.ax_in1.plot(x_t, x_arr, 'b-')
            self.ax_in1.set_title("x(t)")

            # 2. axis => flip+shift
            # Calculation as "h(current_t - tau)"
            if val < len(t_out):
                cur_t = t_out[val]
            else:
                cur_t = t_out[-1]
            self.ax_in2.plot(x_t, x_arr, 'b-', label="x(t)")
            h_shifted = []
            for tau in x_t:
                hh = np.interp(cur_t - tau, h_t, h_arr, left=0, right=0)
                h_shifted.append(hh)
            h_shifted = np.array(h_shifted)
            product = x_arr * h_shifted
            self.ax_in2.plot(x_t, h_shifted, 'g-', label=f"h({cur_t:.2f}-t)")
            self.ax_in2.fill_between(x_t, product, alpha=0.3, color='purple', label="x*h area")
            self.ax_in2.set_title(f"Flip+Shift @ t={cur_t:.2f}")
            self.ax_in2.legend()

        # Original scale
        self.restore_manual_scale(*self.get_time_range(), self.time_mode_selector.get())
        self.update_output_yaxis()

        self.canvas.draw_idle()

    def update_plots(self):
        self.ax_in1.clear()
        self.ax_in2.clear()
        self.ax_out.clear()

        mode = self.time_mode_selector.get()

        # SHIFT
        d1 = self.original_input1_data if len(self.original_input1_data) > 0 else self.input1_data
        d2 = self.original_input2_data if len(self.original_input2_data) > 0 else self.input2_data
        d1_sft = self.apply_shift(d1, self.input1_shift)
        d2_sft = self.apply_shift(d2, self.input2_shift)
        self.input1_data = d1_sft
        self.input2_data = d2_sft

        if len(self.input1_data) > 0:
            self.input1_data.sort(key=lambda p: p[0])
            xv = [p[0] for p in self.input1_data]
            yv = [p[1] for p in self.input1_data]
            if mode == "Continuous Time (t)":
                self.ax_in1.plot(xv, yv, 'b-')
            else:
                self.ax_in1.stem(xv, yv, linefmt='b-', markerfmt='bo', basefmt='k-')

        if len(self.input2_data) > 0:
            self.input2_data.sort(key=lambda p: p[0])
            xv = [p[0] for p in self.input2_data]
            yv = [p[1] for p in self.input2_data]
            if mode == "Continuous Time (t)":
                self.ax_in2.plot(xv, yv, 'g-')
            else:
                self.ax_in2.stem(xv, yv, linefmt='g-', markerfmt='go', basefmt='k-')

        if self.out_signal and len(self.out_signal['t']) > 0:
            to = self.out_signal['t']
            xo = self.out_signal['x']
            arr = sorted(zip(to, xo), key=lambda z: z[0])
            to, xo = zip(*arr)
            if self.out_signal.get('is_discrete', False):
                self.ax_out.stem(to, xo, linefmt='r-', markerfmt='ro', basefmt='k-')
            else:
                self.ax_out.plot(to, xo, 'r-')

        if self.auto_zoom_enabled:
            self.auto_zoom_to_data(mode, *self.get_time_range())
        else:
            self.restore_manual_scale(*self.get_time_range(), mode)

        # 3. axis y-limit
        self.update_output_yaxis()

        # SHIFT range
        st, en = self.get_time_range()
        max_shift = (en - st) * 2
        self.input1_shiftbar.config(from_=-max_shift, to=max_shift)
        self.input2_shiftbar.config(from_=-max_shift, to=max_shift)

        self.canvas.draw_idle()

    def update_output_yaxis(self):

        lines_out = self.ax_out.lines
        if not lines_out:
            return
        all_ys = []
        for ln in lines_out:
            all_ys.extend(ln.get_ydata())
        if not all_ys:
            return
        miny = min(all_ys)
        maxy = max(all_ys)

        if miny > 0:
            miny = 0
        if maxy < 0:
            maxy = 0
        if np.isclose(miny, maxy):
            miny -= 1
            maxy += 1
        self.ax_out.set_ylim(miny, maxy)
        self.ax_out.axhline(0, color='black', linewidth=1)
        self.ax_out.axvline(0, color='black', linewidth=1)
        self.ax_out.grid(True)

    def on_mouse_press(self, event):
        if event.button == 1:
            self.auto_zoom_enabled = False
            if event.inaxes not in [self.ax_in1, self.ax_in2]:
                return
            mode = self.time_mode_selector.get()
            if event.xdata is None or event.ydata is None:
                return
            self.current_ax = event.inaxes
            if mode == "Continuous Time (t)":
                self.is_drawing = True
                self.last_x = event.xdata
                self.add_point_continuous(event.xdata, event.ydata)
                self.update_plots()
            else:
                nn = int(round(event.xdata))
                aa = int(round(event.ydata))
                self.add_point_discrete(nn, aa)
                self.update_plots()
        elif event.button == 2:
            self.is_panning = True
            if event.inaxes and event.xdata is not None:
                self.pan_start_x = event.xdata
                self.original_xlims = (
                    self.ax_in1.get_xlim(),
                    self.ax_in2.get_xlim(),
                    self.ax_out.get_xlim()
                )

    def on_mouse_move(self, event):
        if self.is_drawing and event.button == 1:
            mode = self.time_mode_selector.get()
            if mode == "Continuous Time (t)":
                if event.inaxes == self.current_ax and event.xdata is not None and event.ydata is not None:
                    if event.xdata < self.last_x:
                        return
                    self.last_x = event.xdata
                    self.add_point_continuous(event.xdata, event.ydata)
                    self.update_plots()
        elif self.is_panning and event.button == 2:
            if event.inaxes and event.xdata is not None:
                dx = event.xdata - self.pan_start_x
                for ax, (oxlim) in zip([self.ax_in1, self.ax_in2, self.ax_out], self.original_xlims):
                    ax.set_xlim(oxlim[0] + dx, oxlim[1] + dx)
                self.canvas.draw_idle()

    def on_mouse_release(self, event):
        if event.button == 1:
            self.is_drawing = False
            self.last_x = None
            self.current_ax = None
            self.update_plots()
        elif event.button == 2:
            self.is_panning = False
            self.pan_start_x = None
            self.original_xlims = None
            self.manual_xlims = (
                self.ax_in1.get_xlim(),
                self.ax_in2.get_xlim(),
                self.ax_out.get_xlim()
            )
            self.manual_ylims = (
                self.ax_in1.get_ylim(),
                self.ax_in2.get_ylim(),
                self.ax_out.get_ylim()
            )

    def on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if event.button == 'up':
            scale_factor = 0.9
        elif event.button == 'down':
            scale_factor = 1.1
        else:
            scale_factor = 1.0

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - (1 - relx) * new_width, xdata + relx * new_width])
        ax.set_ylim([ydata - (1 - rely) * new_height, ydata + rely * new_height])

        self.manual_xlims = (
            self.ax_in1.get_xlim(),
            self.ax_in2.get_xlim(),
            self.ax_out.get_xlim()
        )
        self.manual_ylims = (
            self.ax_in1.get_ylim(),
            self.ax_in2.get_ylim(),
            self.ax_out.get_ylim()
        )
        self.canvas.draw_idle()

    def add_point_continuous(self, x, y):
        if self.current_ax == self.ax_in1:
            self.input1_data.append((x, y))
            self.original_input1_data = self.input1_data[:]
        else:
            self.input2_data.append((x, y))
            self.original_input2_data = self.input2_data[:]
        self.last_operation = None
        self.last_data1 = None
        self.last_data2 = None
        self.last_mode = None

    def add_point_discrete(self, n, amp):
        if self.current_ax == self.ax_in1:
            ex = [p[0] for p in self.input1_data]
            if n not in ex:
                self.input1_data.append((n, amp))
                self.original_input1_data = self.input1_data[:]
        else:
            ex = [p[0] for p in self.input2_data]
            if n not in ex:
                self.input2_data.append((n, amp))
                self.original_input2_data = self.input2_data[:]


        self.fill_discrete_gaps()

        self.last_operation = None
        self.last_data1 = None
        self.last_data2 = None
        self.last_mode = None


    def fill_discrete_gaps(self):


        if len(self.input1_data) > 0:
            n_vals = sorted([int(p[0]) for p in self.input1_data])
            min_n = n_vals[0]
            max_n = n_vals[-1]
            filled_data = []
            for n in range(min_n, max_n + 1):
                amps = [p[1] for p in self.input1_data if int(p[0]) == n]
                if amps:
                    filled_data.append((n, amps[0]))
                else:
                    filled_data.append((n, 0.0))
            self.input1_data = filled_data
            self.original_input1_data = filled_data[:]


        if len(self.input2_data) > 0:
            n_vals = sorted([int(p[0]) for p in self.input2_data])
            min_n = n_vals[0]
            max_n = n_vals[-1]
            filled_data = []
            for n in range(min_n, max_n + 1):
                amps = [p[1] for p in self.input2_data if int(p[0]) == n]
                if amps:
                    filled_data.append((n, amps[0]))
                else:
                    filled_data.append((n, 0.0))
            self.input2_data = filled_data
            self.original_input2_data = filled_data[:]

    def get_time_range(self):
        try:
            st = float(self.start_time_input.get())
            en = float(self.end_time_input.get())
        except:
            st, en = -5, 5
        return st, en

    def restore_manual_scale(self, start_time, end_time, mode):

        if self.manual_xlims and self.manual_ylims:
            xlims = self.manual_xlims
            ylims = self.manual_ylims
            self.ax_in1.set_xlim(xlims[0])
            self.ax_in2.set_xlim(xlims[1])
            self.ax_out.set_xlim(xlims[2])
            self.ax_in1.set_ylim(ylims[0])
            self.ax_in2.set_ylim(ylims[1])
            self.ax_out.set_ylim(ylims[2])
        else:
            if mode == "Continuous Time (t)":
                for ax in [self.ax_in1, self.ax_in2, self.ax_out]:
                    ax.set_xlim(start_time, end_time)
                    ax.set_ylim(-5, 5)
            else:
                for ax in [self.ax_in1, self.ax_in2, self.ax_out]:
                    ax.set_xlim(-5, 5)
                    ax.set_ylim(-5, 5)
            self.manual_xlims = (
                self.ax_in1.get_xlim(),
                self.ax_in2.get_xlim(),
                self.ax_out.get_xlim()
            )
            self.manual_ylims = (
                self.ax_in1.get_ylim(),
                self.ax_in2.get_ylim(),
                self.ax_out.get_ylim()
            )

        for ax in [self.ax_in1, self.ax_in2, self.ax_out]:
            ax.axhline(0, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            ax.grid(True)

    def auto_zoom_to_data(self, mode, start_time, end_time):


        for ax in [self.ax_in1, self.ax_in2, self.ax_out]:
            ax.set_xlim(start_time, end_time)
            ax.relim()
            ax.autoscale_view()


        all_y = []
        for ax in [self.ax_in1, self.ax_in2, self.ax_out]:
            lines = ax.get_lines()
            for line in lines:
                all_y.extend(line.get_ydata())
        if all_y:
            min_y = min(all_y)
            max_y = max(all_y)

            min_y = min(min_y, 0)
            max_y = max(max_y, 0)
            for ax in [self.ax_in1, self.ax_in2, self.ax_out]:
                ax.set_ylim(min_y, max_y)
        else:
            for ax in [self.ax_in1, self.ax_in2, self.ax_out]:
                ax.set_ylim(-5, 5)


if __name__ == "__main__":
    root = tk.Tk()
    app = SinyalIslemeApp(root)
    root.mainloop()
