import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

class DroneMPC:
    """
    Model-Predictive Controller for a quad-copter that regulates
    altitude (z) and attitude (roll, pitch, yaw).

    ------------------------------------------------------------
    States  (8) :  z,  dz,  roll, droll, pitch, dpitch, yaw, dyaw
    Inputs  (4) :  thrust,  Mroll,  Mpitch,  Myaw
    Refs    (4) :  z_ref, roll_ref, pitch_ref, yaw_ref
    """

    def __init__(self, dt: float = 0.02, horizon: int = 20):
        self.dt = dt          # [s]  control sampling time (50 Hz)
        self.N = horizon      # prediction horizon (steps)

        # === Physical parameters of the vehicle ===
        self.m   = 0.60   # [kg]   mass
        self.Ixx = 0.02   # [kg·m²] inertia around x (roll)
        self.Iyy = 0.02   # [kg·m²] inertia around y (pitch)
        self.Izz = 0.04   # [kg·m²] inertia around z (yaw)

        # === Actuator limits (tune for your vehicle) ===
        self.thrust_min = 0.0     # [N]
        self.thrust_max = 15.0    # [N]
        self.Mroll_min  = -0.5    # [N·m]
        self.Mroll_max  =  0.5
        self.Mpitch_min = -0.5
        self.Mpitch_max =  0.5
        self.Myaw_min   = -0.2
        self.Myaw_max   =  0.2

        # Pre-build the optimisation problem so each call is fast
        self._build_problem()

    def _build_problem(self):
        """Set up CVXPY variables, parameters and constraints."""
        n_x = 8   # state dimension
        n_u = 4   # input dimension
        N   = self.N
        dt  = self.dt

        A = np.eye(n_x)
        A[0, 1] = dt
        A[2, 3] = dt
        A[4, 5] = dt
        A[6, 7] = dt

        B = np.zeros((n_x, n_u))
        B[1, 0] = dt / self.m
        B[3, 1] = dt / self.Ixx
        B[5, 2] = dt / self.Iyy
        B[7, 3] = dt / self.Izz

        self.x = cp.Variable((n_x, N + 1))
        self.u = cp.Variable((n_u, N))

        self.x0_param  = cp.Parameter(n_x)
        self.ref_param = cp.Parameter(4)

        # Adjust these weights to match the desired behavior
        Qz = 10.0     # Weight for altitude tracking
        Qr = 8.0      # Weight for roll tracking
        Qp = 8.0      # Weight for pitch tracking
        Qy = 5.0      # Weight for yaw tracking
        
        R_thrust = 0.05  # Penalty on thrust usage
        R_moment = 0.05  # Penalty on roll/pitch moment usage
        R_yaw    = 0.02  # Penalty on yaw moment usage

        cost = 0
        constr = [self.x[:, 0] == self.x0_param]

        for k in range(N):
            constr += [self.x[:, k + 1] == A @ self.x[:, k] + B @ self.u[:, k]]
            constr += [self.u[0, k] >= self.thrust_min,
                       self.u[0, k] <= self.thrust_max,
                       self.u[1, k] >= self.Mroll_min,
                       self.u[1, k] <= self.Mroll_max,
                       self.u[2, k] >= self.Mpitch_min,
                       self.u[2, k] <= self.Mpitch_max,
                       self.u[3, k] >= self.Myaw_min,
                       self.u[3, k] <= self.Myaw_max]

            z_err    = self.x[0, k] - self.ref_param[0]
            roll_err = self.x[2, k] - self.ref_param[1]
            pitch_err = self.x[4, k] - self.ref_param[2]
            yaw_err  = self.x[6, k] - self.ref_param[3]

            cost += Qz * cp.square(z_err) \
                  + Qr * cp.square(roll_err) \
                  + Qp * cp.square(pitch_err) \
                  + Qy * cp.square(yaw_err) \
                  + R_thrust * cp.square(self.u[0, k]) \
                  + R_moment * cp.square(self.u[1, k]) \
                  + R_moment * cp.square(self.u[2, k]) \
                  + R_yaw    * cp.square(self.u[3, k])

        z_err_T    = self.x[0, N] - self.ref_param[0]
        roll_err_T = self.x[2, N] - self.ref_param[1]
        pitch_err_T = self.x[4, N] - self.ref_param[2]
        yaw_err_T  = self.x[6, N] - self.ref_param[3]

        cost += Qz * cp.square(z_err_T) \
              + Qr * cp.square(roll_err_T) \
              + Qp * cp.square(pitch_err_T) \
              + Qy * cp.square(yaw_err_T)

        self._prob = cp.Problem(cp.Minimize(cost), constr)

    def _solve_mpc(self):
        try:
            self._prob.solve(solver=cp.OSQP, warm_start=True)
        except cp.SolverError:
            print("Solver error, trying again without warm start")
            self._prob.solve(solver=cp.OSQP)

    def predict(self, entrada: np.ndarray) -> np.ndarray:
        """
        Compute the first optimal control action.

        Parameters
        ----------
        entrada : array-like, shape (12,)
            Concatenation of state estimate (8,) and references (4,).

        Returns
        -------
        u0 : ndarray, shape (4,)
            Control input to apply at the current step.
        """
        entrada = np.asarray(entrada).flatten()
        x_hat = entrada[:8]
        ref = entrada[8:]

        self.x0_param.value  = x_hat
        self.ref_param.value = ref

        self._solve_mpc()
        return self.u[:, 0].value


def simulate_test_scenario(total_time=30.0):
    """
    Simula el escenario de prueba similar a las gráficas proporcionadas.
    """
    # Crear el controlador MPC
    dt = 0.02  # 50 Hz
    mpc = DroneMPC(dt=dt, horizon=20)
    
    # Calcular el número de pasos
    steps = int(total_time / dt)
    
    # Inicializar historia de estados, entradas y referencias
    states = np.zeros((steps+1, 8))
    inputs = np.zeros((steps, 4))
    references = np.zeros((steps, 4))
    
    # Estado inicial (en el suelo, altitud 0.6m)
    states[0] = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Definir perfiles de referencia similares a la gráfica
    time_points = np.arange(0, total_time, dt)
    
    # Asegurarse de que time_points tenga exactamente 'steps' elementos
    time_points = time_points[:steps]
    
    # Altitud: Iniciar a 0.6m y subir a 1.2m
    alt_ref = np.ones_like(time_points) * 0.6
    alt_ref[time_points >= 1.0] = 1.2
    
    # Roll: Normalmente 0, pero con cambios de +3 a -3 grados entre 15-18s
    roll_ref = np.zeros_like(time_points)
    roll_ref[(time_points >= 15.0) & (time_points < 16.5)] = np.radians(3)
    roll_ref[(time_points >= 16.5) & (time_points < 18.0)] = np.radians(-3)
    
    # Pitch: Cambios entre 0, +15, -15 entre 8-12s
    pitch_ref = np.zeros_like(time_points)
    pitch_ref[(time_points >= 8.0) & (time_points < 9.5)] = np.radians(15)
    pitch_ref[(time_points >= 9.5) & (time_points < 12.0)] = np.radians(-15)
    
    # Yaw: Cambio a +20 grados en t=8s
    yaw_ref = np.zeros_like(time_points)
    yaw_ref[(time_points >= 8.0) & (time_points < 9.5)] = np.radians(20)
    
    # Asignar referencias para cada paso
    for k in range(steps):
        references[k] = [alt_ref[k], roll_ref[k], pitch_ref[k], yaw_ref[k]]
    
    # Simulación del modelo dinámico
    for k in range(steps):
        # Obtener referencias actuales
        ref = references[k]
        
        # Concatenar estado y referencia para el MPC
        mpc_input = np.concatenate([states[k], ref])
        
        # Calcular acción de control óptima
        control = mpc.predict(mpc_input)
        inputs[k] = control
        
        # Simular la dinámica del dron
        next_state = np.copy(states[k])
        
        # Gravedad (simplificada)
        gravity_acc = -9.81
        
        # Actualizar velocidades
        next_state[1] += dt * (control[0] / mpc.m + gravity_acc)  # dz (aceleración vertical)
        next_state[3] += dt * control[1] / mpc.Ixx  # droll
        next_state[5] += dt * control[2] / mpc.Iyy  # dpitch
        next_state[7] += dt * control[3] / mpc.Izz  # dyaw
        
        # Actualizar posiciones
        next_state[0] += dt * next_state[1]  # z
        next_state[2] += dt * next_state[3]  # roll
        next_state[4] += dt * next_state[5]  # pitch
        next_state[6] += dt * next_state[7]  # yaw
        
        # Guardar el estado para el siguiente paso
        states[k+1] = next_state
    
    # Crear el vector de tiempo para graficar (asegurarse que tenga la misma longitud que states)
    plot_time_points = np.linspace(0, total_time, steps+1)
    
    return plot_time_points, states, inputs, references


def plot_results(time_points, states, references):
    """
    Genera gráficas similares a las mostradas.
    """
    # Asegurarse de que references tenga la misma longitud que time_points para graficar
    ref_time_points = time_points[:-1]  # Quitar el último punto para que coincida con references
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    
    # Convertir radianes a grados para ángulos
    roll_deg = np.degrees(states[:, 2])
    pitch_deg = np.degrees(states[:, 4])
    yaw_deg = np.degrees(states[:, 6])
    
    roll_ref_deg = np.degrees(references[:, 1])
    pitch_ref_deg = np.degrees(references[:, 2])
    yaw_ref_deg = np.degrees(references[:, 3])
    
    # Roll
    axs[0].plot(time_points, roll_deg, 'purple', linewidth=2)
    axs[0].plot(ref_time_points, roll_ref_deg, 'purple', linestyle='--', linewidth=1)
    axs[0].set_title('Test Roll', color='purple')
    axs[0].set_ylim(-5, 5)
    axs[0].grid(True)
    
    # Pitch
    axs[1].plot(time_points, pitch_deg, 'red', linewidth=2)
    axs[1].plot(ref_time_points, pitch_ref_deg, 'red', linestyle='--', linewidth=1)
    axs[1].set_title('Test Pitch', color='red')
    axs[1].set_ylim(-20, 20)
    axs[1].grid(True)
    
    # Yaw
    axs[2].plot(time_points, yaw_deg, 'magenta', linewidth=2)
    axs[2].plot(ref_time_points, yaw_ref_deg, 'magenta', linestyle='--', linewidth=1)
    axs[2].set_title('Test Yaw', color='magenta')
    axs[2].set_ylim(-5, 25)
    axs[2].grid(True)
    
    # Altitud
    axs[3].plot(time_points, states[:, 0], 'blue', linewidth=2)
    axs[3].plot(ref_time_points, references[:, 0], 'blue', linestyle='--', linewidth=1)
    axs[3].set_title('Test Alt', color='blue')
    axs[3].set_ylim(0.5, 1.3)
    axs[3].grid(True)
    
    axs[3].set_xlabel('Time (sec)')
    
    plt.tight_layout()
    plt.savefig('drone_stabilization_test.png')
    plt.show()


def run_test():
    """
    Ejecuta la simulación y muestra los resultados.
    """
    print("Ejecutando simulación de estabilización del dron...")
    time_points, states, inputs, references = simulate_test_scenario(total_time=30.0)
    
    print("Simulación completada. Mostrando resultados...")
    plot_results(time_points, states, references)
    
    print("Test finalizado. Los resultados se han guardado en 'drone_stabilization_test.png'")
    

if __name__ == "__main__":
    run_test()