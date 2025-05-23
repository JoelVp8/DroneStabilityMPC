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
        self.g   = 9.81   # [m/s²] gravity acceleration

        # === Actuator limits (tune for your vehicle) ===
        self.thrust_min = 0.0     # [N]
        self.thrust_max = 15.0    # [N]
        self.Mroll_min  = -0.5    # [N·m]
        self.Mroll_max  =  0.5
        self.Mpitch_min = -0.5
        self.Mpitch_max =  0.5
        self.Myaw_min   = -0.2
        self.Myaw_max   =  0.2
        
        # Hover thrust - equilibrium point
        self.u_hover = self.m * self.g

        # Pre-build the optimisation problem so each call is fast
        self._build_problem()

    def _build_problem(self):
        """Set up CVXPY variables, parameters and constraints."""
        n_x = 8   # state dimension
        n_u = 4   # input dimension
        N   = self.N
        dt  = self.dt

        # System dynamics (discretized)
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

        # Gravity effect vector (as external disturbance)
        g_vec = np.zeros(n_x)
        g_vec[1] = -self.g * dt  # Effect of gravity on vertical velocity

        self.x = cp.Variable((n_x, N + 1))
        self.u = cp.Variable((n_u, N))
        
        # Parameters for initial state and reference
        self.x0_param  = cp.Parameter(n_x)
        self.ref_param = cp.Parameter(4)

        # Tuning weights for the cost function - increased to improve tracking
        Qz = 25.0     # Weight for altitude tracking (increased from 15)
        Qr = 15.0     # Weight for roll tracking (increased from 8)
        Qp = 15.0     # Weight for pitch tracking (increased from 8)
        Qy = 5.0      # Weight for yaw tracking
        
        R_thrust = 0.001  # Reduced from 0.005 to allow more aggressive control
        R_moment = 0.02   # Reduced from 0.05 to allow more aggressive control
        R_yaw    = 0.02   # Penalty on yaw moment usage
        
        # Rate of change penalty - reduced to allow more responsive control
        R_delta = 0.05    # Reduced from 0.1

        cost = 0
        constr = [self.x[:, 0] == self.x0_param]

        # Previous control input for first step rate limiting
        u_prev = cp.Parameter(n_u)
        self.u_prev_param = u_prev
        
        for k in range(N):
            # System dynamics with gravity
            constr += [self.x[:, k + 1] == A @ self.x[:, k] + B @ self.u[:, k] + g_vec]
            
            # Input constraints
            constr += [self.u[0, k] >= self.thrust_min,
                      self.u[0, k] <= self.thrust_max,
                      self.u[1, k] >= self.Mroll_min,
                      self.u[1, k] <= self.Mroll_max,
                      self.u[2, k] >= self.Mpitch_min,
                      self.u[2, k] <= self.Mpitch_max,
                      self.u[3, k] >= self.Myaw_min,
                      self.u[3, k] <= self.Myaw_max]
            
            # Safety constraints
            constr += [self.x[0, k] >= 0.05]  # Minimum altitude (5cm safety margin)
            
            # Velocity constraints to prevent abrupt movements - increased for faster response
            constr += [self.x[1, k] >= -1.5]  # Max descent velocity (increased from -1.0)
            constr += [self.x[1, k] <= 2.0]   # Max ascent velocity (increased from 1.5)
            
            # Rate of change constraint on first step - increased for faster response
            if k == 0:
                constr += [cp.abs(self.u[:, k] - self.u_prev_param) <= np.array([3.0, 0.3, 0.3, 0.1])]
            
            # Rate of change constraints between consecutive steps
            if k > 0:
                constr += [cp.abs(self.u[:, k] - self.u[:, k-1]) <= np.array([3.0, 0.3, 0.3, 0.1])]

            # Tracking error
            z_err    = self.x[0, k] - self.ref_param[0]
            roll_err = self.x[2, k] - self.ref_param[1]
            pitch_err = self.x[4, k] - self.ref_param[2]
            yaw_err  = self.x[6, k] - self.ref_param[3]
            
            # Stage cost
            cost += Qz * cp.square(z_err) \
                  + Qr * cp.square(roll_err) \
                  + Qp * cp.square(pitch_err) \
                  + Qy * cp.square(yaw_err) \
                  + R_thrust * cp.square(self.u[0, k] - self.u_hover) \
                  + R_moment * cp.square(self.u[1, k]) \
                  + R_moment * cp.square(self.u[2, k]) \
                  + R_yaw    * cp.square(self.u[3, k])
            
            # Add rate of change penalty for all but first step
            if k > 0:
                for i in range(n_u):
                    cost += R_delta * cp.square(self.u[i, k] - self.u[i, k-1])

        # Terminal cost
        z_err_T    = self.x[0, N] - self.ref_param[0]
        roll_err_T = self.x[2, N] - self.ref_param[1]
        pitch_err_T = self.x[4, N] - self.ref_param[2]
        yaw_err_T  = self.x[6, N] - self.ref_param[3]

        # Higher terminal weights for better reference tracking
        cost += 3 * Qz * cp.square(z_err_T) \
              + 3 * Qr * cp.square(roll_err_T) \
              + 3 * Qp * cp.square(pitch_err_T) \
              + 3 * Qy * cp.square(yaw_err_T)

        self._prob = cp.Problem(cp.Minimize(cost), constr)
        
        # Initialize previous control input
        self.prev_u = np.zeros(n_u)
        self.prev_u[0] = self.u_hover  # Initialize with hover thrust

    def _solve_mpc(self):
        try:
            self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            solved = True
        except cp.SolverError:
            print("Solver error, trying again without warm start")
            try:
                self._prob.solve(solver=cp.OSQP, verbose=False)
                solved = True
            except cp.SolverError:
                print("Solver failed again, using fallback control")
                solved = False
        
        return solved

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

        self.x0_param.value = x_hat
        self.ref_param.value = ref
        self.u_prev_param.value = self.prev_u

        solved = self._solve_mpc()
        
        if solved and self.u[:, 0].value is not None:
            control = self.u[:, 0].value
        else:
            # Fallback control: maintain hover thrust and zero moments
            control = np.zeros(4)
            control[0] = self.u_hover
            print("Using fallback control")
        
        # Store for next iteration
        self.prev_u = control
        
        return control


def simulate_test_scenario(total_time=30.0):
    """
    Simula el escenario de prueba con un despegue y maniobras seguras.
    """
    # Crear el controlador MPC
    dt = 0.02  # 50 Hz
    mpc = DroneMPC(dt=dt, horizon=20)  # Reducido a 20 pasos
    
    # Calcular el número de pasos
    steps = int(total_time / dt)
    
    # Inicializar historia de estados, entradas y referencias
    states = np.zeros((steps+1, 8))
    inputs = np.zeros((steps, 4))
    references = np.zeros((steps, 4))
    
    # Estado inicial (en el suelo, con pequeña altura segura)
    states[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Inicializar el empuje con valor de hover para evitar caída
    mpc.prev_u[0] = mpc.u_hover * 1.2  # Increased for better takeoff
    
    # Definir perfiles de referencia 
    time_points = np.arange(0, total_time, dt)
    time_points = time_points[:steps]  # Asegurar longitud correcta
    
    # Altitud: Perfil de referencia según lo requerido
    alt_ref = np.zeros_like(time_points)
    
    # Primero sube a 0.6m (hasta t=2s), luego sube gradualmente a 1.2m (hasta t=5s)
    for i, t in enumerate(time_points):
        if t < 0.5:
            alt_ref[i] = 0.0
        elif t < 2.0:
            # Subida inicial hasta 0.6m
            progress = (t - 0.5) / 1.5
            alt_ref[i] = progress * 0.6
        elif t < 5.0:
            # Subida gradual de 0.6m a 1.2m
            progress = (t - 2.0) / 3.0
            alt_ref[i] = 0.6 + progress * 0.6
        else:
            # Mantener en 1.2m
            alt_ref[i] = 1.2
    
    # Roll: Perfil según lo requerido
    roll_ref = np.zeros_like(time_points)
    
    # Baja a -3.8 grados, mantiene 2s, sube a +2 grados
    # Luego baja a -2.5 en 15s, mantiene 2s, sube a +2.5
    for i, t in enumerate(time_points):
        if t < 10.0:
            roll_ref[i] = 0.0
        elif t < 11.0:
            # Baja a -3.8 grados
            progress = (t - 10.0) / 1.0
            roll_ref[i] = -3.8 * progress
        elif t < 13.0:
            # Mantiene -3.8 grados
            roll_ref[i] = -3.8
        elif t < 14.0:
            # Sube a +2 grados
            progress = (t - 13.0) / 1.0
            roll_ref[i] = -3.8 + (5.8 * progress)
        elif t < 15.0:
            # Mantiene +2 grados
            roll_ref[i] = 2.0
        elif t < 17.0:
            # Baja a -2.5 grados
            progress = (t - 15.0) / 2.0
            roll_ref[i] = 2.0 - (4.5 * progress)
        elif t < 19.0:
            # Mantiene -2.5 grados
            roll_ref[i] = -2.5
        elif t < 20.0:
            # Sube a +2.5 grados
            progress = (t - 19.0) / 1.0
            roll_ref[i] = -2.5 + (5.0 * progress)
        else:
            # Mantiene +2.5 grados
            roll_ref[i] = 2.5
    
    # Convertir a radianes
    roll_ref = np.radians(roll_ref)
    
    # Pitch: Perfil según lo requerido
    pitch_ref = np.zeros_like(time_points)
    
    # En t=10s sube a +15 grados, mantiene 1s, baja a 0 y mantiene 2s,
    # baja a -15 grados, regresa a 0 después de 1s y se mantiene
    for i, t in enumerate(time_points):
        if t < 10.0:
            pitch_ref[i] = 0.0
        elif t < 10.5:
            # Sube a +15 grados
            progress = (t - 10.0) / 0.5
            pitch_ref[i] = 15.0 * progress
        elif t < 11.5:
            # Mantiene +15 grados
            pitch_ref[i] = 15.0
        elif t < 12.0:
            # Baja a 0 grados
            progress = (t - 11.5) / 0.5
            pitch_ref[i] = 15.0 - (15.0 * progress)
        elif t < 14.0:
            # Mantiene 0 grados
            pitch_ref[i] = 0.0
        elif t < 14.5:
            # Baja a -15 grados
            progress = (t - 14.0) / 0.5
            pitch_ref[i] = -15.0 * progress
        elif t < 15.5:
            # Mantiene -15 grados
            pitch_ref[i] = -15.0
        elif t < 16.0:
            # Regresa a 0 grados
            progress = (t - 15.5) / 0.5
            pitch_ref[i] = -15.0 + (15.0 * progress)
        else:
            # Mantiene 0 grados
            pitch_ref[i] = 0.0
    
    # Convertir a radianes
    pitch_ref = np.radians(pitch_ref)
    
    # Yaw: mantenemos como está
    yaw_ref = np.zeros_like(time_points)
    
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
        
        # Perturbación simulada (viento descendente constante) - reducida para mejor rendimiento
        wind_disturbance = -0.1  # [N] de empuje negativo (reducido de -0.15)

        # Ruido leve en la estimación de velocidad vertical
        dz_noise = np.random.normal(0, 0.01)  # Reducido de 0.02 m/s

        # Actualizar velocidad vertical con perturbación y ruido
        next_state[1] += dt * ((control[0] + wind_disturbance) / mpc.m - mpc.g) + dz_noise

        next_state[3] += dt * control[1] / mpc.Ixx  # droll
        next_state[5] += dt * control[2] / mpc.Iyy  # dpitch
        next_state[7] += dt * control[3] / mpc.Izz  # dyaw
        
        # Actualizar posiciones
        next_state[0] += dt * next_state[1]  # z
        next_state[2] += dt * next_state[3]  # roll
        next_state[4] += dt * next_state[5]  # pitch
        next_state[6] += dt * next_state[7]  # yaw
        
        # Restricción física: el dron no puede estar por debajo del suelo
        next_state[0] = max(0.0, next_state[0])
        
        # Si el dron toca el suelo, detener la velocidad vertical
        if next_state[0] <= 0.001:
            next_state[1] = 0.0
        
        # Guardar el estado para el siguiente paso
        states[k+1] = next_state
    
    # Crear el vector de tiempo para graficar
    plot_time_points = np.linspace(0, total_time, steps+1)
    
    return plot_time_points, states, inputs, references


def plot_results(time_points, states, inputs, references):
    """
    Genera gráficas detalladas del comportamiento del dron.
    """
    # Asegurarse de que references tenga la misma longitud que time_points para graficar
    ref_time_points = time_points[:-1]  # Quitar el último punto para que coincida con references
    
    # Crear subfiguras
    fig, axs = plt.subplots(5, 1, figsize=(12, 15))
    
    # Convertir radianes a grados para ángulos
    roll_deg = np.degrees(states[:, 2])
    pitch_deg = np.degrees(states[:, 4])
    yaw_deg = np.degrees(states[:, 6])
    
    roll_ref_deg = np.degrees(references[:, 1])
    pitch_ref_deg = np.degrees(references[:, 2])
    yaw_ref_deg = np.degrees(references[:, 3])
    
    # Altitud (primera para ver claramente)
    axs[0].plot(time_points, states[:, 0], 'blue', linewidth=2, label='Actual')
    axs[0].plot(ref_time_points, references[:, 0], 'blue', linestyle='--', linewidth=1, label='Reference')
    axs[0].set_title('Test Alt', color='blue')
    axs[0].set_ylim(-0.1, 1.3)
    axs[0].grid(True)
    axs[0].legend()
    
    # Velocidad vertical
    axs[1].plot(time_points, states[:, 1], 'green', linewidth=2)
    axs[1].set_title('Vertical Velocity (dz)', color='green')
    axs[1].set_ylim(-1.5, 1.5)
    axs[1].grid(True)
    
    # Roll
    axs[2].plot(time_points, roll_deg, 'purple', linewidth=2)
    axs[2].plot(ref_time_points, roll_ref_deg, 'purple', linestyle='--', linewidth=1)
    axs[2].set_title('Test Roll', color='purple')
    axs[2].set_ylim(-5, 5)
    axs[2].grid(True)
    
    # Pitch
    axs[3].plot(time_points, pitch_deg, 'red', linewidth=2)
    axs[3].plot(ref_time_points, pitch_ref_deg, 'red', linestyle='--', linewidth=1)
    axs[3].set_title('Test Pitch', color='red')
    axs[3].set_ylim(-20, 20)
    axs[3].grid(True)
    
    # Yaw
    axs[4].plot(time_points, yaw_deg, 'magenta', linewidth=2)
    axs[4].plot(ref_time_points, yaw_ref_deg, 'magenta', linestyle='--', linewidth=1)
    axs[4].set_title('Test Yaw', color='magenta')
    axs[4].set_ylim(-5, 25)
    axs[4].grid(True)
    
    axs[4].set_xlabel('Time (sec)')
    
    plt.tight_layout()
    plt.savefig('drone_stabilization_test.png')
    
    # Graficar los controles también
    fig2, axs2 = plt.subplots(4, 1, figsize=(12, 10))
    
    # Thrust
    axs2[0].plot(ref_time_points, inputs[:, 0], 'blue', linewidth=2)
    axs2[0].axhline(y=0.6*9.81, color='r', linestyle='--', label='Hover thrust')
    axs2[0].set_title('Thrust')
    axs2[0].grid(True)
    axs2[0].legend()
    
    # Roll moment
    axs2[1].plot(ref_time_points, inputs[:, 1], 'purple', linewidth=2)
    axs2[1].set_title('Roll Moment')
    axs2[1].grid(True)
    
    # Pitch moment
    axs2[2].plot(ref_time_points, inputs[:, 2], 'red', linewidth=2)
    axs2[2].set_title('Pitch Moment')
    axs2[2].grid(True)
    
    # Yaw moment
    axs2[3].plot(ref_time_points, inputs[:, 3], 'magenta', linewidth=2)
    axs2[3].set_title('Yaw Moment')
    axs2[3].grid(True)
    
    axs2[3].set_xlabel('Time (sec)')
    
    plt.tight_layout()
    plt.savefig('drone_control_inputs.png')
    
    plt.show()


def run_test():
    """
    Ejecuta la simulación y muestra los resultados.
    """
    print("Ejecutando simulación de estabilización del dron...")
    time_points, states, inputs, references = simulate_test_scenario(total_time=30.0)
    
    print("Simulación completada. Mostrando resultados...")
    plot_results(time_points, states, inputs, references)
    
    print("Test finalizado. Los resultados se han guardado en 'drone_stabilization_test.png' y 'drone_control_inputs.png'")
    

if __name__ == "__main__":
    run_test()