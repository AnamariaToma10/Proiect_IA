import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import make_interp_spline
import tkinter as tk
from tkinter import Tk, Button, Label, messagebox, Frame, Radiobutton, StringVar
import pygame
import random

#  CONFIGURARE SI VARIABILE GLOBALE SPLINE
X_DATE = None
Y_DATE = None
radacina = None # Fereastra principala

# CONFIGURARE SNAKE
SNAKE_BLOCK = 20
SNAKE_W, SNAKE_H = 600, 600
GRID_W, GRID_H = SNAKE_W // SNAKE_BLOCK, SNAKE_H // SNAKE_BLOCK     # o tabla mai mic pentru PSO
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Sus, Jos, Stanga, Dreapta


#  CLASA PSO (LOGICA MATEMATICA) - ALGORITMUL PARTICLE SWARM OPTIMIZATION
class OptimizarePSO:
    """
    Implementare clasica a algoritmului Particle Swarm Optimization (PSO)
    cu suport pentru topologii gbest (global best) si lbest (local best).
    """
    def __init__(self, dim, limite, nr_particule, nr_iteratii, functie_cost, mod='gbest'):
        """
        Initializare PSO.
        
        Parametri:
            dim          : dimensiunea spatiului de cautare (numar de variabile)
            limite       : lista/tuplu de perechi (min, max) pentru fiecare dimensiune
            nr_particule : numarul de particule din roi
            nr_iteratii  : numarul maxim de iteratii
            functie_cost : functia obiectiv de minimizat
            mod          : 'gbest' pentru topologie globala, 'lbest' pentru topologie locala (inel)
        """
        self.dim = dim
        self.limite = np.array(limite)
        self.nr_part = nr_particule
        self.max_iter = nr_iteratii
        self.cost = functie_cost
        self.mod = mod
        self.iter_curenta = 0

        # initializare pozitii si viteze ale particulelor
        jos = self.limite[:,0]
        sus = self.limite[:,1]
        self.pozitii = np.random.uniform(jos, sus, (nr_particule, dim))
        self.viteze = np.random.uniform(-0.1, 0.1, (nr_particule, dim))

        # Cele mai bune pozitii personale si scorurile asociate
        self.p_best = self.pozitii.copy()
        self.scor_p = np.array([self.cost(p) for p in self.pozitii])

        # Cel mai bun global (gbest)
        idx_gbest = np.argmin(self.scor_p)
        self.g_best = self.p_best[idx_gbest].copy()
        self.scor_g = float(self.scor_p[idx_gbest])

    def pas(self):
        """
        Executa o iteratie a algoritmului PSO.
        Returneaza False daca s-a atins numarul maxim de iteratii.
        """
        if self.iter_curenta >= self.max_iter:
            return False

        # Greutate inertiala adaptiva (scade liniar de la 0.9 la 0.4)
        w = 0.9 - 0.5 * (self.iter_curenta / self.max_iter)

        for i in range(self.nr_part):
            r1 = np.random.rand(self.dim) #componenta cognitiva
            r2 = np.random.rand(self.dim) #componenta sociala

            # selectie sociala
            if self.mod == 'gbest':
                social = self.g_best
            else:  # lbest – topologie inel
                stanga = (i - 1) % self.nr_part
                dreapta = (i + 1) % self.nr_part
                vecini = [stanga, i, dreapta]
                idx = min(vecini, key=lambda j: self.scor_p[j])
                social = self.p_best[idx]

            cognitiv = self.p_best[i]

            # actualizare viteza si pozitie
            self.viteze[i] = (w * self.viteze[i] +2 * r1 * (cognitiv - self.pozitii[i]) +  2 * r2 * (social - self.pozitii[i]))

            # Limitare viteza pentru stabilitate
            lim_v = 0.2 * (self.limite[:,1] - self.limite[:,0])
            self.viteze[i] = np.clip(self.viteze[i], -lim_v, lim_v)

            # Actualizare pozitie si restrictionare in limite
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
    """
    Functie de cost pentru optimizarea punctelor interne ale unui spline cubic.
    Minimizeaza eroarea patratica intre spline si datele originale.
    """
    global X_DATE, Y_DATE
    y0 = Y_DATE[0]# Valoare la capatul stang (fix)
    yN = Y_DATE[-1]# Valoare la capatul drept (fix)
    # construim setul complet de puncte
    x_full = np.r_[X_DATE.min(), x_control, X_DATE.max()]
    y_full = np.r_[y0, y_interne, yN]

    try:
        spline = make_interp_spline(x_full, y_full, k=3)# Spline cubic
        pred = spline(X_DATE)# Predictii pe datele originale
        eroare = np.sum((pred - Y_DATE)**2)# Eroare patratica totala
        return eroare
    except:
        return 1e9

def start_animatie_spline(mod_pso):
    global X_DATE, Y_DATE

    # Configurare PSO
    nr_interne = 5# Numar puncte interne de optimizat
    x_ctrl = np.linspace(0.1, 0.9, nr_interne)# Pozitii fixe ale punctelor interne pe axa x
    ymin, ymax = Y_DATE.min(), Y_DATE.max()
    marja = max(0.5, 0.2 * abs(ymax - ymin))
    limite = [(ymin - marja, ymax + marja)] * nr_interne

    func_wrapper = lambda y: cost_spline(y, x_ctrl)# Wrapper pentru functia de cost

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

        # Spline final
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


class SnakeEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_W // 2, GRID_H // 2)]
        self.food = self.spawn_food()
        self.direction = 3  # Start spre Dreapta
        self.score = 0
        self.dead = False

    def spawn_food(self):
        while True:
            x, y = random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1)
            if (x, y) not in self.snake: return (x, y)

    def simuleaza_pasi(self, moves_idx):
        """ Functie folosita de PSO pentru a testa o traiectorie imaginara.
            Returneaza COSTUL (mai mic e mai bine). """
        sim_snake = list(self.snake)
        sim_head = sim_snake[0]
        sim_dir = self.direction

        # Distanta Manhattan initiala pana la mancare
        dist_start = abs(sim_head[0] - self.food[0]) + abs(sim_head[1] - self.food[1])
        min_dist = dist_start

        for move in moves_idx:
            # Prevenire intoarcere 180 grade
            if (sim_dir == 0 and move == 1) or (sim_dir == 1 and move == 0) or (sim_dir == 2 and move == 3) or (sim_dir == 3 and move == 2):
                move = sim_dir
            sim_dir = move

            dx, dy = DIRS[move]
            nx, ny = sim_head[0] + dx, sim_head[1] + dy

            # Verificare Coliziune (Moarte in simulare)
            if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H or (nx, ny) in sim_snake[:-1]:
                return 100  # Cost penalizare

            sim_snake.insert(0, (nx, ny))
            sim_head = (nx, ny)

            # Calcul distanta minima atinsa in acest plan
            dist = abs(nx - self.food[0]) + abs(ny - self.food[1])
            if dist < min_dist: min_dist = dist

            if (nx, ny) == self.food:
                return -10  # Recompensa (Scor negativ)
            else:
                sim_snake.pop()

        return min_dist # Daca n-a murit, costul este distanta ramasa

    def executa_mutare(self, move_idx):
        """ Aplica mutarea decisa de PSO in jocul REAL. """
        if (self.direction == 0 and move_idx == 1) or (self.direction == 1 and move_idx == 0) or \
           (self.direction == 2 and move_idx == 3) or (self.direction == 3 and move_idx == 2):
            move_idx = self.direction
        self.direction = move_idx

        dx, dy = DIRS[self.direction]
        nx, ny = self.snake[0][0] + dx, self.snake[0][1] + dy

        if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H or (nx, ny) in self.snake[:-1]:
            self.dead = True
            return

        self.snake.insert(0, (nx, ny))
        if (nx, ny) == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()


def ruleaza_snake_ai(tip_mod):
    pygame.init()
    screen = pygame.display.set_mode((SNAKE_W, SNAKE_H))
    pygame.display.set_caption(f"Snake AI - PSO: {tip_mod.upper()}")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont('Arial', 22)

    game = SnakeEngine()

    plt.ion()
    fig_p, ax_p = plt.subplots(figsize=(5, 5))
    ax_p.set_title(f"Spatiul Particulelor \nMod: {tip_mod}")
    ax_p.set_xlim(-0.5, 4.5);
    ax_p.set_ylim(-0.5, 4.5)
    scat = ax_p.scatter([], [], c='blue', alpha=0.5, label='Particule')
    gbest_dot = ax_p.scatter([], [], c='red', s=100, marker='X', label='Lider')
    ax_p.legend()

    ORIZONT = 12
    NR_PART = 25
    ITERATII = 15

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        if game.dead:
            game.reset()
            pygame.time.delay(300)
            continue

        # INTEGRARE PSO - transforma pozitiile particulelor in mutari valide (0-3)
        def snake_cost_wrapper(p):
            moves = np.clip(p, 0, 3.99).astype(int)
            return game.simuleaza_pasi(moves)

        if game.score > 20:
            ORIZONT = 20
            ITERATII = 25

        limite = [(0, 4)] * ORIZONT
        pso = OptimizarePSO(ORIZONT, limite, NR_PART, ITERATII, snake_cost_wrapper, mod=tip_mod)

        for _ in range(ITERATII):
            pso.pas()

        scat.set_offsets(pso.pozitii[:, :2])
        gbest_dot.set_offsets([pso.g_best[:2]])
        fig_p.canvas.draw()
        fig_p.canvas.flush_events()

        # Este luata prima mutare din secventa optima gasita si este executata in jocul real
        best_move = np.clip(pso.g_best[0], 0, 3.99).astype(int)
        game.executa_mutare(best_move)

        screen.fill((20, 20, 20))

        pygame.draw.rect(screen, (255, 50, 50),(game.food[0] * SNAKE_BLOCK, game.food[1] * SNAKE_BLOCK, SNAKE_BLOCK, SNAKE_BLOCK))

        for p in game.snake:
            pygame.draw.rect(screen, (50, 255, 50),(p[0] * SNAKE_BLOCK, p[1] * SNAKE_BLOCK, SNAKE_BLOCK, SNAKE_BLOCK))
            pygame.draw.rect(screen, (255, 255, 255),(p[0] * SNAKE_BLOCK, p[1] * SNAKE_BLOCK, SNAKE_BLOCK, SNAKE_BLOCK), 1)


        info_text = f"Scor: {game.score} | Cost PSO (Fitness): {pso.scor_g:.2f}"
        text_surface = font.render(info_text, True, (255, 255, 255))

        bg_rect = pygame.Rect(5, 5, text_surface.get_width() + 10, text_surface.get_height() + 5)
        pygame.draw.rect(screen, (50, 50, 50), bg_rect)

        screen.blit(text_surface, (10, 8))

        pygame.display.flip()
        clock.tick(15)

    plt.close(fig_p)
    plt.ioff()
    pygame.quit()

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

    btn_snake = Button(radacina, text="2. Optimizare Snake AI", width=25, height=2, bg="#27ae60", fg="white", command=meniu_snake_setup)
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

#  MENIU 2: SNAKE
def meniu_snake_setup():
    curata_fereastra()
    radacina.title("Configurare Snake PSO")

    Label(radacina, text="Alege Topologia PSO:", font=("Arial", 14), bg="#2c3e50", fg="white").pack(pady=20)

    v = StringVar(radacina, "gbest")

    rb_frame = Frame(radacina, bg="#2c3e50")
    rb_frame.pack(pady=10)

    Radiobutton(rb_frame, text="Global Best", variable=v, value="gbest",bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Arial", 12)).pack(anchor="w")
    Radiobutton(rb_frame, text="Local Best", variable=v, value="lbest",bg="#2c3e50", fg="white", selectcolor="#34495e", font=("Arial", 12)).pack(anchor="w")
    Button(radacina, text="Start Joc", bg="#2ecc71", fg="white", width=20, height=2,command=lambda: ruleaza_snake_ai(v.get())).pack(pady=20)
    Button(radacina, text="Inapoi la Meniu", command=meniu_principal).pack(pady=10)

#  MAIN
if __name__ == "__main__":
    radacina = Tk()
    radacina.geometry("500x500")
    radacina.configure(bg="#2c3e50")

    # Pornire interfata
    meniu_principal()
    radacina.mainloop()