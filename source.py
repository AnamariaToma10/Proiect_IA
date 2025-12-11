import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import make_interp_spline
import tkinter as tk
from tkinter import Tk, Button, Label, messagebox, Frame

#  CONFIGURARE SI VARIABILE GLOBALE 
X_DATE = None
Y_DATE = None
radacina = None # Fereastra principala

#  CLASA PSO (LOGICA MATEMATICA) 
class OptimizarePSO:
    def __init__(self, dim, limite, nr_particule, nr_iteratii, functie_cost, mod='gbest'):
        self.dim = dim
        self.limite = np.array(limite)
        self.nr_part = nr_particule
        self.max_iter = nr_iteratii
        self.cost = functie_cost
        self.mod = mod
        self.iter_curenta = 0

        # initializare
        jos = self.limite[:,0]
        sus = self.limite[:,1]
        self.pozitii = np.random.uniform(jos, sus, (nr_particule, dim))
        self.viteze = np.random.uniform(-0.1, 0.1, (nr_particule, dim))

        self.p_best = self.pozitii.copy()
        self.scor_p = np.array([self.cost(p) for p in self.pozitii])

        idx_gbest = np.argmin(self.scor_p)
        self.g_best = self.p_best[idx_gbest].copy()
        self.scor_g = float(self.scor_p[idx_gbest])

    def pas(self):
        if self.iter_curenta >= self.max_iter:
            return False

        # inertie adaptiva
        w = 0.9 - 0.5 * (self.iter_curenta / self.max_iter)

        for i in range(self.nr_part):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            # selectie sociala
            if self.mod == 'gbest':
                social = self.g_best
            else:  # lbest
                stanga = (i - 1) % self.nr_part
                dreapta = (i + 1) % self.nr_part
                vecini = [stanga, i, dreapta]
                idx = min(vecini, key=lambda j: self.scor_p[j])
                social = self.p_best[idx]

            cognitiv = self.p_best[i]

            # actualizare viteza si pozitie
            self.viteze[i] = (w * self.viteze[i] +2 * r1 * (cognitiv - self.pozitii[i]) +  2 * r2 * (social - self.pozitii[i]))

            lim_v = 0.2 * (self.limite[:,1] - self.limite[:,0])
            self.viteze[i] = np.clip(self.viteze[i], -lim_v, lim_v)

            self.pozitii[i] += self.viteze[i]
            self.pozitii[i] = np.clip(self.pozitii[i], self.limite[:,0], self.limite[:,1])

            # evaluare
            scor_nou = self.cost(self.pozitii[i])
            if scor_nou < self.scor_p[i]:
                self.scor_p[i] = scor_nou
                self.p_best[i] = self.pozitii[i].copy()
                if scor_nou < self.scor_g:
                    self.scor_g = scor_nou
                    self.g_best = self.pozitii[i].copy()

        self.iter_curenta += 1
        return True

#  LOGICA SPLINE 
def cost_spline(y_interne, x_control):
    global X_DATE, Y_DATE
    y0 = Y_DATE[0]
    yN = Y_DATE[-1]
    # construim setul complet de puncte
    x_full = np.r_[X_DATE.min(), x_control, X_DATE.max()]
    y_full = np.r_[y0, y_interne, yN]

    try:
        spline = make_interp_spline(x_full, y_full, k=3)
        pred = spline(X_DATE)
        eroare = np.sum((pred - Y_DATE)**2)
        return eroare
    except:
        return 1e9

def start_animatie_spline(mod_pso):
    global X_DATE, Y_DATE
    
    # Configurare PSO
    nr_interne = 5
    x_ctrl = np.linspace(0.1, 0.9, nr_interne)
    ymin, ymax = Y_DATE.min(), Y_DATE.max()
    marja = max(0.5, 0.2 * abs(ymax - ymin))
    limite = [(ymin - marja, ymax + marja)] * nr_interne

    func_wrapper = lambda y: cost_spline(y, x_ctrl)

    pso = OptimizarePSO(dim=nr_interne, limite=limite, nr_particule=30, nr_iteratii=120, functie_cost=func_wrapper, mod=mod_pso)

    # Configurare Grafic Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Optimizare PSO Spline - Mod: {mod_pso}")
    
    # Grafic 1: Particule
    ax1.set_title("Cautare Particule")
    ax1.plot(X_DATE, Y_DATE, 'co', label='Date Reale')
    linii_part = [ax1.plot([], [], 'y-', alpha=0.3)[0] for _ in range(pso.nr_part)]
    linie_best, = ax1.plot([], [], 'g-', lw=2, label='Best')
    ax1.legend()

    # Grafic 2: Rezultat
    ax2.set_title("Rezultat Spline")
    ax2.plot(X_DATE, Y_DATE, 'co', markersize=8)
    linie_rez, = ax2.plot([], [], 'r-', lw=3)
    
    txt_info = ax2.text(0.05, 0.9, "", transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    def update(frame):
        activ = pso.pas()
        if not activ: anim.event_source.stop()

        # Desenare particule
        for i, linie in enumerate(linii_part):
            linie.set_data(x_ctrl, pso.pozitii[i])
        
        # Desenare best
        gb = pso.g_best
        x_g = np.r_[X_DATE.min(), x_ctrl, X_DATE.max()]
        y_g = np.r_[Y_DATE[0], gb, Y_DATE[-1]]
        linie_best.set_data(x_g, y_g)

        # Spline fin
        try:
            spl = make_interp_spline(x_g, y_g, k=3)
            xx = np.linspace(X_DATE.min(), X_DATE.max(), 300)
            linie_rez.set_data(xx, spl(xx))
        except: pass

        txt_info.set_text(f"Iter: {pso.iter_curenta}\nEroare: {pso.scor_g:.9f}")
        return lines_part + [linie_best, linie_rez, txt_info]

    lines_part = linii_part 
    anim = FuncAnimation(fig, update, interval=50, repeat=False)
    plt.show()

#  INTERFATA GRAFICA 
def curata_fereastra():
    """Sterge tot continutul din fereastra principala"""
    for widget in radacina.winfo_children():
        widget.destroy()

def meniu_principal():
    curata_fereastra()
    radacina.title("Proiect PSO - Meniu Principal")
    
    lbl = Label(radacina, text="Selecteaza Modulul", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
    lbl.pack(pady=40)

    btn_spline = Button(radacina, text="1. Optimizare Spline", width=25, height=2, bg="#3498db", fg="white",command=meniu_spline_setup)
    btn_spline.pack(pady=10)

    #TO DO: Snake
    btn_snake = Button(radacina, text="2. Optimizare Snake AI (WIP)", width=25, height=2, bg="#7f8c8d", fg="white", command=meniu_snake_setup)
    btn_snake.pack(pady=10)

    btn_exit = Button(radacina, text="Iesire", width=15, bg="#c0392b", fg="white", command=radacina.quit)
    btn_exit.pack(pady=30)

#  MENIU 1: SPLINE 
def meniu_spline_setup():
    curata_fereastra()
    radacina.title("Configurare Date Spline")
    Label(radacina, text="Introdu puncte (x y) - unul pe linie:", bg="#2c3e50", fg="white").pack(pady=5)
    
    txt_input = tk.Text(radacina, height=10, width=40)
    txt_input.pack(pady=5)
    txt_input.insert(tk.END, "0.0 0.0\n0.1 0.8\n0.2 0.9\n0.35 0.7\n0.5 0.0\n0.65 -0.7\n0.8 -0.9\n0.9 -0.75\n1.0 0.0") #date default

    def proceseaza_si_start(mod):
        global X_DATE, Y_DATE
        continut = txt_input.get("1.0", tk.END).strip().split('\n')
        xl, yl = [], []
        try:
            for linie in continut:
                if not linie.strip(): continue
                vals = list(map(float, linie.split()))
                xl.append(vals[0])
                yl.append(vals[1])
            
            # Sortare
            perechi = sorted(zip(xl, yl))
            X_DATE = np.array([p[0] for p in perechi])
            Y_DATE = np.array([p[1] for p in perechi])
            
            # Porneste animatia
            start_animatie_spline(mod)
            
        except ValueError:
            messagebox.showerror("Eroare", "Format date invalid!")

    frame_btn = Frame(radacina, bg="#2c3e50")
    frame_btn.pack(pady=10)

    Button(frame_btn, text="Start GBEST", bg="#27ae60", fg="white", width=15, command=lambda: proceseaza_si_start('gbest')).grid(row=0, column=0, padx=5)
    Button(frame_btn, text="Start LBEST", bg="#e67e22", fg="white", width=15, command=lambda: proceseaza_si_start('lbest')).grid(row=0, column=1, padx=5)
    Button(radacina, text="Inapoi la Meniu", command=meniu_principal).pack(pady=10)
#TO DO ...
#  MENIU 2: SNAKE (PLACEHOLDER) 
def meniu_snake_setup():
    curata_fereastra()
    radacina.title("Snake AI Config")
    Label(radacina, text="Modul Snake AI - In Dezvoltare", font=("Arial", 14), bg="#2c3e50", fg="white").pack(pady=50)
    
    # TO DO... widget-uri specifice snake(marime grid, viteza etc)
    Button(radacina, text="Inapoi", command=meniu_principal).pack(pady=20)

#  MAIN 
if __name__ == "__main__":
    radacina = Tk()
    radacina.geometry("500x500")
    radacina.configure(bg="#2c3e50")
    
    # Pornire interfata
    meniu_principal() 
    radacina.mainloop()