import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import sympy as sp

# Señales
def square_coeffs(N):
    """Onda cuadrada"""
    coeffs = []
    for n in range(1, N+1, 2):
        coeffs.append((4/np.pi) * (1/n))
    return coeffs, "seno"

def triangle_coeffs(N):
    """Onda triangular"""
    coeffs = []
    for n in range(1, N+1, 2):
        coeffs.append((8/(np.pi**2)) * ((-1)**((n-1)//2) / (n**2)))
    return coeffs, "seno"

def sawtooth_coeffs(N):
    """Diente de sierra: f(x) = -(2/π) * Σ [(-1)^n / n] * sin(nx)"""
    coeffs = []
    for n in range(1, N+1):
        coeffs.append((-2/np.pi) * ((-1)**n) / n)
    return coeffs, "seno"

def sinusoid_coeffs(N):
    """Sin"""
    coeffs = [1.0] + [0]*(N-1)
    return coeffs, "seno"

def two_sines_coeffs(N):
    """Suma de dos senos: sin(t) + 0.5*sin(3t)"""
    coeffs = []
    for n in range(1, N+1):
        if n == 1:
            coeffs.append(1.0)  # sin(t)
        elif n == 3 and N >= 3:
            coeffs.append(0.5)  # 0.5*sin(3t)
        else:
            coeffs.append(0.0)
    return coeffs, "seno"

def parabola_coeffs(N):
    """Parábola: f(x) = x² para -π<x<π"""
    coeffs = []
    a0 = (2 * np.pi**2) / 3  # Término constante
    coeffs.append(a0 / 2)  # a0/2 para el término constante
    for n in range(1, N+1):
        an = (4 * (-1)**n) / (n**2)
        coeffs.append(an)
    return coeffs, "coseno"

# Señal de referencia
def perfect_square(t):
    return np.sign(np.sin(t))

def perfect_triangle(t):
    return (2/np.pi) * np.arcsin(np.sin(t))

def perfect_sawtooth(t):
    return 2 * (t / (2*np.pi) - np.floor(t / (2*np.pi) + 0.5))

def perfect_parabola(t):
    """Parábola: f(x) = x² para -π<x<π, extendida periódicamente"""
    # Normalizar t al rango [-π, π]
    t_mod = t % (2*np.pi)
    t_shifted = np.where(t_mod < np.pi, t_mod, t_mod - 2*np.pi)
    return t_shifted**2

# app
class FourierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Series de Fourier")
        
        try:
            self.root.iconbitmap("spiral.ico")
        except:
            pass  
        
        self.root.configure(bg='white')

        self.N = tk.IntVar(value=7)
        self.signal_type = tk.StringVar(value="Cuadrada")
        
        # Parámetros para beats/interferencia
        self.f1 = tk.DoubleVar(value=10.0)  # Frecuencia 1
        self.f2 = tk.DoubleVar(value=11.0)  # Frecuencia 2
        self.A1 = tk.DoubleVar(value=1.0)   # Amplitud 1
        self.A2 = tk.DoubleVar(value=1.0)   # Amplitud 2

        # Señal dropdown
        ttk.Label(root, text="Tipo de señal:").grid(row=0, column=0, padx=5, pady=5)
        self.signal_menu = ttk.Combobox(root, textvariable=self.signal_type, values=["Cuadrada","Triangular","Sierra","Parábola","Beats"])
        self.signal_menu.grid(row=0, column=1, padx=5, pady=5)
        self.signal_menu.bind("<<ComboboxSelected>>", self.on_signal_change)

        # Frame para controles de Fourier
        self.fourier_frame = ttk.LabelFrame(root, text="Parámetros de Fourier")
        self.fourier_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Slider N (para Fourier)
        ttk.Label(self.fourier_frame, text="Número de términos N:").grid(row=0, column=0, padx=5, pady=5)
        self.slider = tk.Scale(self.fourier_frame, from_=1, to=50, orient="horizontal", variable=self.N, command=self.update_plot, length=400)
        self.slider.grid(row=0, column=1, padx=5, pady=5)
        
        # Frame para controles de Beats
        self.beats_frame = ttk.LabelFrame(root, text="Parámetros de Beats")
        self.beats_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Frecuencia 1
        ttk.Label(self.beats_frame, text="Frecuencia 1 (f₁):").grid(row=0, column=0, padx=5, pady=5)
        self.f1_scale = tk.Scale(self.beats_frame, from_=1.0, to=20.0, resolution=0.1, orient="horizontal", 
                                  variable=self.f1, command=self.update_plot, length=200)
        self.f1_scale.grid(row=0, column=1, padx=5, pady=5)
        
        # Frecuencia 2
        ttk.Label(self.beats_frame, text="Frecuencia 2 (f₂):").grid(row=0, column=2, padx=5, pady=5)
        self.f2_scale = tk.Scale(self.beats_frame, from_=1.0, to=20.0, resolution=0.1, orient="horizontal", 
                                  variable=self.f2, command=self.update_plot, length=200)
        self.f2_scale.grid(row=0, column=3, padx=5, pady=5)
        
        # Amplitud 1
        ttk.Label(self.beats_frame, text="Amplitud 1 (A₁):").grid(row=1, column=0, padx=5, pady=5)
        self.A1_scale = tk.Scale(self.beats_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", 
                                  variable=self.A1, command=self.update_plot, length=200)
        self.A1_scale.grid(row=1, column=1, padx=5, pady=5)
        
        # Amplitud 2
        ttk.Label(self.beats_frame, text="Amplitud 2 (A₂):").grid(row=1, column=2, padx=5, pady=5)
        self.A2_scale = tk.Scale(self.beats_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", 
                                  variable=self.A2, command=self.update_plot, length=200)
        self.A2_scale.grid(row=1, column=3, padx=5, pady=5)
        
        self.beats_frame.grid_remove()

        # subplots
        self.fig = None
        self.canvas = None
        self.ax_fourier = None
        self.ax_target = None
        self.ax_beats_y1 = None
        self.ax_beats_y2 = None
        self.ax_beats_sum = None
        
        # Serie de fórmulas
        self.fig_formula, self.ax_formula = plt.subplots(1, 1, figsize=(10, 2))
        self.canvas_formula = FigureCanvasTkAgg(self.fig_formula, master=root)
        self.canvas_formula.get_tk_widget().grid(row=4, column=0, columnspan=2, pady=10)
        
        # Crear gráficos iniciales
        self.create_fourier_plots()

        self.update_plot()
    
    def on_signal_change(self, event=None):
        """Maneja el cambio de tipo de señal"""
        signal_type = self.signal_type.get()
        if signal_type == "Beats":
            self.fourier_frame.grid_remove()
            self.beats_frame.grid()
            self.create_beats_plots()
        else:
            self.fourier_frame.grid()
            self.beats_frame.grid_remove()
            self.create_fourier_plots()
        self.update_plot()
    
    def create_fourier_plots(self):
        """Crea los gráficos que sean Fourier"""
        if self.fig is not None and self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            plt.close(self.fig)
        self.fig, (self.ax_fourier, self.ax_target) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=5)
        self.ax_beats_y1 = None
        self.ax_beats_y2 = None
        self.ax_beats_sum = None
    
    def create_beats_plots(self):
        """Crea los gráficos para Beats"""
        if self.fig is not None and self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            plt.close(self.fig)
        self.fig, (self.ax_beats_y1, self.ax_beats_y2, self.ax_beats_sum) = plt.subplots(3, 1, figsize=(10, 7), 
                                                                                         gridspec_kw={'hspace': 0.35})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=5)
        self.ax_fourier = None
        self.ax_target = None

    def get_coeffs(self):
        N = self.N.get()
        sig = self.signal_type.get()

        if sig == "Cuadrada":
            return square_coeffs(N)
        elif sig == "Triangular":
            return triangle_coeffs(N)
        elif sig == "Sierra":
            return sawtooth_coeffs(N)
        elif sig == "Parábola":
            return parabola_coeffs(N)
        else:
            return square_coeffs(N)  # Default

    def update_plot(self, event=None):
        signal_name = self.signal_type.get()
        
        # Modo Beats
        if signal_name == "Beats":
            if self.ax_beats_y1 is None or self.ax_beats_y2 is None or self.ax_beats_sum is None:
                return
            
            self.ax_beats_y1.clear()
            self.ax_beats_y2.clear()
            self.ax_beats_sum.clear()
            
            # Parámetros
            f1 = self.f1.get()
            f2 = self.f2.get()
            A1 = self.A1.get()
            A2 = self.A2.get()
            
            # Tiempo: mostrar varios periodos de beats
            f_beat = abs(f2 - f1)  # Frecuencia de beats
            if f_beat > 0:
                T_beat = 1.0 / f_beat  # Periodo de beats
                t_max = 3 * T_beat  # Mostrar 3 periodos de beats
            else:
                t_max = 2.0
            t = np.linspace(0, t_max, 5000)
            
            # Calcular señales
            y1 = A1 * np.sin(2 * np.pi * f1 * t)
            y2 = A2 * np.sin(2 * np.pi * f2 * t)
            y_sum = y1 + y2
            
            # Calcular envolvente
            # Para y1 = A1*sin(2πf1*t) y y2 = A2*sin(2πf2*t)
            # Cuando A1 = A2: y1 + y2 = 2A*sin(2π*(f1+f2)/2*t)*cos(2π*(f1-f2)/2*t)
            # La envolvente es: ±2A*|cos(2π*(f1-f2)/2*t)|
            f_diff = abs(f2 - f1) / 2
            if f_diff > 0:
                envelope_modulation = np.abs(np.cos(2 * np.pi * f_diff * t))
                #  cuando las señales están en fase, la amplitud es A1+A2
                # cuando están fuera de fase, es |A1-A2|
                max_amp = A1 + A2
                min_amp = abs(A1 - A2)
                envelope_amplitude = min_amp + (max_amp - min_amp) * envelope_modulation
                envelope_upper = envelope_amplitude
                envelope_lower = -envelope_amplitude
            else:
                # Si f1 = f2, no hay beats, amplitud constante
                envelope_upper = (A1 + A2) * np.ones_like(t)
                envelope_lower = -(A1 + A2) * np.ones_like(t)
            
            # Gráfica  y1
            self.ax_beats_y1.plot(t, y1, color='magenta', linewidth=1.5, label=f'y₁ = {A1:.1f}sin(2π·{f1:.2f}t)')
            self.ax_beats_y1.set_ylabel('Amplitud', fontsize=9)
            self.ax_beats_y1.set_title(f'Señal 1 (y₁) - f₁ = {f1:.2f} Hz, A₁ = {A1:.1f}', fontsize=10, fontweight='bold')
            self.ax_beats_y1.legend(loc='upper right', fontsize=8)
            self.ax_beats_y1.grid(True, alpha=0.3)
            self.ax_beats_y1.set_xlim(t[0], t[-1])
            self.ax_beats_y1.tick_params(labelsize=8)
            
            # Gráfica y2
            self.ax_beats_y2.plot(t, y2, color='red', linewidth=1.5, label=f'y₂ = {A2:.1f}sin(2π·{f2:.2f}t)')
            self.ax_beats_y2.set_ylabel('Amplitud', fontsize=9)
            self.ax_beats_y2.set_title(f'Señal 2 (y₂) - f₂ = {f2:.2f} Hz, A₂ = {A2:.1f}', fontsize=10, fontweight='bold')
            self.ax_beats_y2.legend(loc='upper right', fontsize=8)
            self.ax_beats_y2.grid(True, alpha=0.3)
            self.ax_beats_y2.set_xlim(t[0], t[-1])
            self.ax_beats_y2.tick_params(labelsize=8)
            
            # Gráfica y1 + y2 + envolvente
            self.ax_beats_sum.plot(t, y_sum, color='blue', linewidth=1.2, label='y₁ + y₂', alpha=0.8)
            self.ax_beats_sum.plot(t, envelope_upper, color='green', linewidth=1.5, linestyle='--', 
                                   label='Envolvente superior', alpha=0.7)
            self.ax_beats_sum.plot(t, envelope_lower, color='cyan', linewidth=1.5, linestyle='--', 
                                   label='Envolvente inferior', alpha=0.7)
            self.ax_beats_sum.set_xlabel('Tiempo t (s)', fontsize=9)
            self.ax_beats_sum.set_ylabel('Amplitud', fontsize=9)
            if f_beat > 0:
                title = f'Señal Resultante (y₁ + y₂) - T_beat = {T_beat:.3f} s, f_beat = {f_beat:.3f} Hz'
            else:
                title = 'Señal Resultante (y₁ + y₂) - Sin beats (f₁ = f₂)'
            self.ax_beats_sum.set_title(title, fontsize=10, fontweight='bold')
            self.ax_beats_sum.legend(loc='upper right', fontsize=8)
            self.ax_beats_sum.grid(True, alpha=0.3)
            self.ax_beats_sum.set_xlim(t[0], t[-1])
            self.ax_beats_sum.tick_params(labelsize=8)
            
            # Marcar puntos de interferencia constructiva y destructiva
            if f_beat > 0:
                max_indices = []
                for i in range(1, len(t)-1):
                    if envelope_amplitude[i] > envelope_amplitude[i-1] and envelope_amplitude[i] > envelope_amplitude[i+1]:
                        if envelope_amplitude[i] > 0.9:  # Cerca del máximo
                            max_indices.append(i)
                
                if len(max_indices) > 0:
                    for idx in max_indices[:3]:  #  los primeros 3
                        self.ax_beats_sum.axvline(x=t[idx], color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Actualizar fórmula
            self.ax_formula.clear()
            if f_beat > 0:
                formula_text = f"y₁ + y₂ = {A1:.1f}sin(2π·{f1:.2f}t) + {A2:.1f}sin(2π·{f2:.2f}t)"
                formula_text += f" | T_beat = {T_beat:.3f} s, f_beat = {f_beat:.3f} Hz"
            else:
                formula_text = f"y₁ + y₂ = {A1:.1f}sin(2π·{f1:.2f}t) + {A2:.1f}sin(2π·{f2:.2f}t) (f₁ = f₂, sin beats)"
            self.ax_formula.text(0.5, 0.5, formula_text, transform=self.ax_formula.transAxes, 
                               fontsize=11, ha='center', va='center')
            self.ax_formula.set_xlim(0, 1)
            self.ax_formula.set_ylim(0, 1)
            self.ax_formula.axis('off')
            self.canvas_formula.draw()
            return
        
        #  Fourier 
        if self.ax_fourier is None or self.ax_target is None:
            return
            
        self.ax_fourier.clear()
        self.ax_target.clear()
        
        T = 2*np.pi
        t = np.linspace(0, 4*np.pi, 1000)

        coeffs, kind = self.get_coeffs()
        y_sum = np.zeros_like(t)

        # Plot individual terms and cumulative sum
        cumulative_sum = np.zeros_like(t)
        term_count = 0
        
        has_const_term = signal_name in ["Parábola"]
        const_idx = 0
        
        for i, c in enumerate(coeffs):
            if has_const_term and i == 0:

                term = c * np.ones_like(t)
                cumulative_sum += term
                continue
            
            # Calcula n segn el tipo de señal
            if signal_name == "Sierra":
                n = i + 1
            elif signal_name == "Parábola":
                # El primer término (i=0) es constante, luego todos los términos
                n = i
            else:
                n = 2*i+1  # Para las impares (Cuadrada)

            if kind == "seno":
                term = c * np.sin(n*t)
            else:  # coseno
                term = c * np.cos(n*t)
            
            if abs(c) > 1e-10:  # Evitar términos prácticamente cero
                term_count += 1
                self.ax_fourier.plot(t, term, alpha=0.4, linewidth=1, linestyle='--', color='gray')
                
                cumulative_sum += term
                if term_count <= 3:
                    self.ax_fourier.plot(t, cumulative_sum, alpha=0.8, linewidth=1.5, 
                                       label=f"N = {term_count}")

        self.ax_fourier.plot(t, cumulative_sum, color="black", linewidth=3, label="Suma Final")

        self.ax_fourier.set_title(f"Serie de Fourier - {self.signal_type.get()} (N = {self.N.get()})", fontsize=12, fontweight='bold')
        self.ax_fourier.set_ylabel('Amplitud', fontsize=11)
        self.ax_fourier.legend()
        self.ax_fourier.grid()

        # Señal ref
        signal_type = self.signal_type.get()
        if signal_type == "Cuadrada":
            perfect_signal = perfect_square(t)
        elif signal_type == "Triangular":
            perfect_signal = perfect_triangle(t)
        elif signal_type == "Sierra":
            perfect_signal = perfect_sawtooth(t)
        elif signal_type == "Parábola":
            perfect_signal = perfect_parabola(t)
        else:
            perfect_signal = perfect_square(t)  # Default
        
        self.ax_target.plot(t, perfect_signal, color="red", linewidth=3, label="Señal Objetivo")
        self.ax_target.set_title(f"Señal Objetivo - {signal_type}", fontsize=12, fontweight='bold')
        self.ax_target.set_xlabel('x (rad)', fontsize=11)
        self.ax_target.set_ylabel('Amplitud', fontsize=11)
        self.ax_target.legend()
        self.ax_target.grid()
        
        # Error Cuadrático Medio mean square error
        mse = np.mean((perfect_signal - cumulative_sum)**2)
        
        # cajita del mse
        self.ax_fourier.text(0.98, 0.98, f'MSE = {mse:.6f}', 
                            transform=self.ax_fourier.transAxes,
                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.fig.tight_layout()
        self.canvas.draw()

        # Formulas
        self.ax_formula.clear()
        n, x = sp.symbols("n x")
        
        # Formula notación suma
        signal_name = self.signal_type.get()
        if signal_name=="Cuadrada":
            implicit_formula = r"$f(x) = \frac{4}{\pi} \sum_{n=1}^{" + str(self.N.get()) + r"} \frac{1}{n} \sin(nx)$"
        elif signal_name=="Triangular":
            implicit_formula = r"$f(x) = \frac{8}{\pi^2} \sum_{n=1,3,5,\ldots}^{" + str(self.N.get()) + r"} \frac{(-1)^{(n-1)/2}}{n^2} \sin(nx)$"
        elif signal_name=="Sierra":
            implicit_formula = r"$f(x) = -\frac{2}{\pi} \sum_{n=1}^{" + str(self.N.get()) + r"} \frac{(-1)^n}{n} \sin(nx)$"
        elif signal_name=="Parábola":
            implicit_formula = r"$f(x) = \frac{\pi^2}{3} + 4 \sum_{n=1}^{" + str(self.N.get()) + r"} \frac{(-1)^n}{n^2} \cos(nx)$"
        else:
            implicit_formula = r"$f(x) = \frac{4}{\pi} \sum_{n=1}^{" + str(self.N.get()) + r"} \frac{1}{n} \sin(nx)$"
        
        # expresion con terminos 
        terms_list = []
        coeffs, kind = self.get_coeffs()
        has_const = signal_name in ["Parábola"]
        
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-10:  # Saltar términos prácticamente cero
                continue
                
            if has_const and i == 0:
                term_str = f"{c:.3f}"
                terms_list.append(term_str)
                continue
            
            # Calcular n_val segun tipo de señal
            if signal_name == "Sierra":
                n_val = i + 1
            elif signal_name == "Parábola":
                n_val = i
            else:
                n_val = 2*i + 1  # Para las impares
            
            if kind == "seno":
                if c > 0:
                    term_str = f"+ {c:.3f} \\sin({n_val}x)"
                else:
                    term_str = f"- {abs(c):.3f} \\sin({n_val}x)"
            else:  
                if c > 0:
                    term_str = f"+ {c:.3f} \\cos({n_val}x)"
                else:
                    term_str = f"- {abs(c):.3f} \\cos({n_val}x)"
            
            terms_list.append(term_str)
        
        if terms_list:
            # Remover el primer + si existe
            if terms_list[0].startswith("+ "):
                terms_list[0] = terms_list[0][2:]
            
            max_terms = min(len(terms_list), 4)  # Mostrar máximo 4 terminos
            visible_terms = terms_list[:max_terms]
            
            if len(terms_list) > max_terms:
                explicit_formula = r"$f(x) = " + " + ".join(visible_terms) + r" + \cdots$"
            else:
                explicit_formula = r"$f(x) = " + " + ".join(visible_terms) + r"$"
        else:
            explicit_formula = r"$f(x) = 0$"
        
        self.ax_formula.text(0.5, 0.7, implicit_formula, transform=self.ax_formula.transAxes, 
                           fontsize=12, ha='center', va='center', weight='bold')
        self.ax_formula.text(0.5, 0.3, explicit_formula, transform=self.ax_formula.transAxes, 
                           fontsize=10, ha='center', va='center', style='italic')
        self.ax_formula.set_xlim(0, 1)
        self.ax_formula.set_ylim(0, 1)
        self.ax_formula.axis('off')
        
        self.canvas_formula.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = FourierApp(root)
    root.mainloop()
