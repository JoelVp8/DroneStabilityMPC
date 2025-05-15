import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam

class DroneSimulator:
    """
    Simple drone simulator with dynamics similar to the original MPC.
    """
    def __init__(self, dt=0.02):
        self.dt = dt  # [s] control sampling time (50 Hz)
        
        # Physical parameters
        self.m = 0.60    # [kg] mass
        self.Ixx = 0.02  # [kg·m²] inertia around x (roll)
        self.Iyy = 0.02  # [kg·m²] inertia around y (pitch)
        self.Izz = 0.04  # [kg·m²] inertia around z (yaw)
        self.g = 9.81    # [m/s²] gravity acceleration
        
        # Hover thrust (equilibrium)
        self.u_hover = self.m * self.g
        
        # Control limits
        self.thrust_min = 0.0
        self.thrust_max = 15.0
        self.Mroll_min = -0.5
        self.Mroll_max = 0.5
        self.Mpitch_min = -0.5
        self.Mpitch_max = 0.5
        self.Myaw_min = -0.2
        self.Myaw_max = 0.2
    
    def step(self, state, control, add_disturbance=True):
        """
        Simulate one step of the drone dynamics.
        
        Parameters:
        -----------
        state: np.ndarray (8,)
            Current state [z, dz, roll, droll, pitch, dpitch, yaw, dyaw]
        control: np.ndarray (4,)
            Control inputs [thrust, Mroll, Mpitch, Myaw]
        add_disturbance: bool
            Whether to add wind disturbance and noise
            
        Returns:
        --------
        next_state: np.ndarray (8,)
            Next state after applying control
        """
        next_state = np.copy(state)
        
        # Apply actuator limits
        thrust = np.clip(control[0], self.thrust_min, self.thrust_max)
        Mroll = np.clip(control[1], self.Mroll_min, self.Mroll_max)
        Mpitch = np.clip(control[2], self.Mpitch_min, self.Mpitch_max)
        Myaw = np.clip(control[3], self.Myaw_min, self.Myaw_max)
        
        # Disturbance (if enabled)
        wind_disturbance = -0.15 if add_disturbance else 0.0
        dz_noise = np.random.normal(0, 0.02) if add_disturbance else 0.0
        
        # Update velocities
        next_state[1] += self.dt * ((thrust + wind_disturbance) / self.m - self.g) + dz_noise  # dz
        next_state[3] += self.dt * Mroll / self.Ixx   # droll
        next_state[5] += self.dt * Mpitch / self.Iyy  # dpitch
        next_state[7] += self.dt * Myaw / self.Izz    # dyaw
        
        # Update positions
        next_state[0] += self.dt * next_state[1]  # z
        next_state[2] += self.dt * next_state[3]  # roll
        next_state[4] += self.dt * next_state[5]  # pitch
        next_state[6] += self.dt * next_state[7]  # yaw
        
        # Ground constraint
        if next_state[0] <= 0.001:
            next_state[0] = 0.0
            next_state[1] = 0.0  # Stop vertical velocity if on ground
        
        return next_state


class SimpleDroneController:
    """
    Simple Neural Network Controller for drone control.
    Uses a Sequential model to avoid recursion issues.
    """
    def __init__(self):
        # Physical parameters
        self.m = 0.60    # [kg] mass
        self.g = 9.81    # [m/s²] gravity
        self.u_hover = self.m * self.g
        
        # Control limits
        self.thrust_min = 0.0
        self.thrust_max = 15.0
        self.Mroll_min = -0.5
        self.Mroll_max = 0.5
        self.Mpitch_min = -0.5
        self.Mpitch_max = 0.5
        self.Myaw_min = -0.2
        self.Myaw_max = 0.2
        
        # Previous control input for smooth transitions
        self.prev_u = np.array([self.u_hover, 0.0, 0.0, 0.0])
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Create a simple neural network model for drone control."""
        # Input shape: [state (8), reference (4), previous control (4)]
        input_dim = 16
        
        # Sequential model (simpler structure to avoid recursion issues)
        self.model = Sequential([
            # First hidden layer
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            
            # Second hidden layer
            Dense(64, activation='relu'),
            BatchNormalization(),
            
            # Output layer (raw outputs)
            Dense(4, activation='sigmoid')  # Use sigmoid to limit output range
        ])
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Print model summary
        self.model.summary()
    
    def predict(self, state, reference):
        """
        Compute control action based on current state and reference.
        
        Parameters:
        -----------
        state: np.ndarray (8,)
            Current state [z, dz, roll, droll, pitch, dpitch, yaw, dyaw]
        reference: np.ndarray (4,)
            References [z_ref, roll_ref, pitch_ref, yaw_ref]
            
        Returns:
        --------
        control: np.ndarray (4,)
            Control inputs [thrust, Mroll, Mpitch, Myaw]
        """
        # Combine inputs
        model_input = np.concatenate([state, reference, self.prev_u]).reshape(1, -1)
        
        # Get raw predictions (0 to 1 range from sigmoid)
        raw_output = self.model.predict(model_input, verbose=0)[0]
        
        # Scale to control ranges
        thrust = self.thrust_min + (self.thrust_max - self.thrust_min) * raw_output[0]
        Mroll = self.Mroll_min + (self.Mroll_max - self.Mroll_min) * raw_output[1]
        Mpitch = self.Mpitch_min + (self.Mpitch_max - self.Mpitch_min) * raw_output[2]
        Myaw = self.Myaw_min + (self.Myaw_max - self.Myaw_min) * raw_output[3]
        
        # Combine control outputs
        control = np.array([thrust, Mroll, Mpitch, Myaw])
        
        # Store for next iteration
        self.prev_u = control
        
        return control
    
    def train(self, train_X, train_y, epochs=50, batch_size=64, validation_split=0.2):
        """
        Train the model using provided data.
        
        Parameters:
        -----------
        train_X: np.ndarray
            Input training data
        train_y: np.ndarray
            Output training data
        epochs: int
            Number of training epochs
        batch_size: int
            Batch size for training
        validation_split: float
            Fraction of data to use for validation
        
        Returns:
        --------
        history: tf.keras.callbacks.History
            Training history
        """
        # Set up early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            train_X, 
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath):
        """Save the model to a file."""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a model from a file."""
        self.model = tf.keras.models.load_model(filepath)


def generate_training_data_from_pd(n_samples=50000):
    """
    Generate training data using a simple PD controller for drone control.
    This is a simpler alternative to using the MPC controller which can cause
    recursion errors with deepcopy.
    
    Parameters:
    -----------
    n_samples: int
        Number of training samples to generate
        
    Returns:
    --------
    X: np.ndarray
        Input data [state, reference, previous control]
    y: np.ndarray
        Target outputs (control actions)
    """
    print("Generating training data using PD controller...")
    
    # Create simulator
    simulator = DroneSimulator()
    
    # PD controller gains
    Kp_z = 5.0
    Kd_z = 2.0
    Kp_roll = 0.3
    Kd_roll = 0.1
    Kp_pitch = 0.3
    Kd_pitch = 0.1
    Kp_yaw = 0.2
    Kd_yaw = 0.1
    
    # Storage for data
    states = []
    references = []
    prev_controls = []
    controls = []
    
    # Previous control for first step
    prev_control = np.array([simulator.u_hover, 0.0, 0.0, 0.0])
    
    # Generate random scenarios
    for _ in range(int(n_samples / 100)):  # ~100 steps per scenario
        # Random initial state with small perturbations
        state = np.array([
            np.random.uniform(0.1, 0.5),  # z - start near ground
            np.random.uniform(-0.2, 0.2),  # dz
            np.random.uniform(-0.1, 0.1),  # roll
            np.random.uniform(-0.1, 0.1),  # droll
            np.random.uniform(-0.1, 0.1),  # pitch
            np.random.uniform(-0.1, 0.1),  # dpitch
            np.random.uniform(-0.1, 0.1),  # yaw
            np.random.uniform(-0.1, 0.1),  # dyaw
        ])
        
        # Random reference target
        reference = np.array([
            np.random.uniform(0.3, 1.5),    # z_ref
            np.random.uniform(-0.2, 0.2),   # roll_ref
            np.random.uniform(-0.3, 0.3),   # pitch_ref
            np.random.uniform(-0.5, 0.5),   # yaw_ref
        ])
        
        # Run simulation for multiple steps
        for _ in range(100):
            # Extract state components
            z, dz, roll, droll, pitch, dpitch, yaw, dyaw = state
            z_ref, roll_ref, pitch_ref, yaw_ref = reference
            
            # PD control calculation
            thrust = simulator.m * (simulator.g + Kp_z * (z_ref - z) + Kd_z * (-dz))
            Mroll = Kp_roll * (roll_ref - roll) + Kd_roll * (-droll)
            Mpitch = Kp_pitch * (pitch_ref - pitch) + Kd_pitch * (-dpitch)
            Myaw = Kp_yaw * (yaw_ref - yaw) + Kd_yaw * (-dyaw)
            
            # Apply control limits
            thrust = np.clip(thrust, simulator.thrust_min, simulator.thrust_max)
            Mroll = np.clip(Mroll, simulator.Mroll_min, simulator.Mroll_max)
            Mpitch = np.clip(Mpitch, simulator.Mpitch_min, simulator.Mpitch_max)
            Myaw = np.clip(Myaw, simulator.Myaw_min, simulator.Myaw_max)
            
            # Combined control action
            control = np.array([thrust, Mroll, Mpitch, Myaw])
            
            # Store the data point
            states.append(state)
            references.append(reference)
            prev_controls.append(prev_control)
            controls.append(control)
            
            # Update state and previous control
            state = simulator.step(state, control)
            prev_control = control
            
            # Occasionally change the reference to get more varied data
            if np.random.random() < 0.05:
                reference = np.array([
                    np.random.uniform(0.3, 1.5),
                    np.random.uniform(-0.2, 0.2),
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.5, 0.5),
                ])
    
    # Combine inputs [state, reference, prev_control]
    X = np.hstack([
        np.array(states),
        np.array(references),
        np.array(prev_controls)
    ])
    
    # Output control actions
    y = np.array(controls)
    
    # Normalize outputs to 0-1
    y_normalized = np.zeros_like(y)
    y_normalized[:, 0] = (y[:, 0] - simulator.thrust_min) / (simulator.thrust_max - simulator.thrust_min)
    y_normalized[:, 1] = (y[:, 1] - simulator.Mroll_min) / (simulator.Mroll_max - simulator.Mroll_min)
    y_normalized[:, 2] = (y[:, 2] - simulator.Mpitch_min) / (simulator.Mpitch_max - simulator.Mpitch_min)
    y_normalized[:, 3] = (y[:, 3] - simulator.Myaw_min) / (simulator.Myaw_max - simulator.Myaw_min)
    
    print(f"Generated {len(X)} training samples")
    
    return X, y_normalized


def plot_training_history(history):
    """Plot the training history."""
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('nn_training_history.png')
    plt.close()


def simulate_test_scenario(controller, total_time=30.0):
    """
    Simulate the test scenario with the neural network controller.
    
    Parameters:
    -----------
    controller: SimpleDroneController
        Neural network controller
    total_time: float
        Total simulation time in seconds
        
    Returns:
    --------
    time_points: np.ndarray
        Time points for plotting
    states: np.ndarray
        State history
    inputs: np.ndarray
        Control input history
    references: np.ndarray
        Reference history
    """
    # Simulation parameters
    dt = 0.02  # 50 Hz
    steps = int(total_time / dt)
    
    # Create simulator
    simulator = DroneSimulator(dt=dt)
    
    # Initialize histories
    states = np.zeros((steps+1, 8))
    inputs = np.zeros((steps, 4))
    references = np.zeros((steps, 4))
    
    # Initial state (on the ground with small safety margin)
    states[0] = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Initialize controller with hover thrust
    controller.prev_u[0] = controller.u_hover
    
    # Define reference profiles
    time_points = np.arange(0, total_time, dt)
    time_points = time_points[:steps]  # Ensure correct length
    
    # Altitude: Smooth transition from 0.1m to 1.2m
    alt_ref = np.ones_like(time_points) * 0.1
    
    t_start = 1.0   # Start ascent
    t_end = 3.0     # Reach final altitude
    
    for i, t in enumerate(time_points):
        if t < t_start:
            alt_ref[i] = 0.1
        elif t < t_end:
            # Smooth transition (ramp)
            progress = (t - t_start) / (t_end - t_start)
            alt_ref[i] = 0.1 + progress * (1.2 - 0.1)
        else:
            alt_ref[i] = 1.2
    
    # Roll: Normally 0, with changes between 15-18s
    roll_ref = np.zeros_like(time_points)
    roll_ref[(time_points >= 15.0) & (time_points < 16.5)] = np.radians(3)
    roll_ref[(time_points >= 16.5) & (time_points < 18.0)] = np.radians(-3)
    
    # Pitch: Changes between 8-12s
    pitch_ref = np.zeros_like(time_points)
    pitch_ref[(time_points >= 8.0) & (time_points < 9.5)] = np.radians(15)
    pitch_ref[(time_points >= 9.5) & (time_points < 12.0)] = np.radians(-15)
    
    # Yaw: Change at 8s
    yaw_ref = np.zeros_like(time_points)
    yaw_ref[(time_points >= 8.0) & (time_points < 9.5)] = np.radians(20)
    
    # Run simulation
    for k in range(steps):
        # Get current reference
        ref = np.array([alt_ref[k], roll_ref[k], pitch_ref[k], yaw_ref[k]])
        references[k] = ref
        
        # Get control from neural network
        control = controller.predict(states[k], ref)
        inputs[k] = control
        
        # Simulate drone dynamics
        states[k+1] = simulator.step(states[k], control)
    
    # Create time points for plotting
    plot_time_points = np.linspace(0, total_time, steps+1)
    
    return plot_time_points, states, inputs, references


def plot_results(time_points, states, inputs, references):
    """
    Plot detailed graphs of drone behavior.
    
    Parameters:
    -----------
    time_points: np.ndarray
        Time points for plotting
    states: np.ndarray
        State history
    inputs: np.ndarray
        Control input history
    references: np.ndarray
        Reference history
    """
    # Adjust reference time points to match length
    ref_time_points = time_points[:-1]
    
    # Create subfigures
    fig, axs = plt.subplots(5, 1, figsize=(12, 15))
    
    # Convert radians to degrees for angles
    roll_deg = np.degrees(states[:, 2])
    pitch_deg = np.degrees(states[:, 4])
    yaw_deg = np.degrees(states[:, 6])
    
    roll_ref_deg = np.degrees(references[:, 1])
    pitch_ref_deg = np.degrees(references[:, 2])
    yaw_ref_deg = np.degrees(references[:, 3])
    
    # Altitude
    axs[0].plot(time_points, states[:, 0], 'blue', linewidth=2, label='Actual')
    axs[0].plot(ref_time_points, references[:, 0], 'blue', linestyle='--', linewidth=1, label='Reference')
    axs[0].set_title('Neural Network Controller - Altitude', color='blue')
    axs[0].set_ylim(-0.1, 1.3)
    axs[0].grid(True)
    axs[0].legend()
    
    # Vertical velocity
    axs[1].plot(time_points, states[:, 1], 'green', linewidth=2)
    axs[1].set_title('Vertical Velocity (dz)', color='green')
    axs[1].set_ylim(-1.5, 1.5)
    axs[1].grid(True)
    
    # Roll
    axs[2].plot(time_points, roll_deg, 'purple', linewidth=2)
    axs[2].plot(ref_time_points, roll_ref_deg, 'purple', linestyle='--', linewidth=1)
    axs[2].set_title('Roll Angle', color='purple')
    axs[2].set_ylim(-5, 5)
    axs[2].grid(True)
    
    # Pitch
    axs[3].plot(time_points, pitch_deg, 'red', linewidth=2)
    axs[3].plot(ref_time_points, pitch_ref_deg, 'red', linestyle='--', linewidth=1)
    axs[3].set_title('Pitch Angle', color='red')
    axs[3].set_ylim(-20, 20)
    axs[3].grid(True)
    
    # Yaw
    axs[4].plot(time_points, yaw_deg, 'magenta', linewidth=2)
    axs[4].plot(ref_time_points, yaw_ref_deg, 'magenta', linestyle='--', linewidth=1)
    axs[4].set_title('Yaw Angle', color='magenta')
    axs[4].set_ylim(-5, 25)
    axs[4].grid(True)
    
    axs[4].set_xlabel('Time (sec)')
    
    plt.tight_layout()
    plt.savefig('nn_drone_stabilization.png')
    plt.close()
    
    # Plot control inputs
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
    plt.savefig('nn_drone_control_inputs.png')
    plt.close()


def run_simple_nn_controller():
    """
    Main function to train and test the simple neural network controller.
    This version avoids the recursion errors.
    """
    print("Starting drone neural network controller training and testing...")
    
    # 1. Generate training data (using PD controller instead of MPC to avoid recursion issues)
    try:
        print("Attempting to load pre-generated training data...")
        X = np.load('drone_training_inputs.npy')
        y = np.load('drone_training_outputs.npy')
        print("Successfully loaded existing training data.")
    except FileNotFoundError:
        print("Generating new training data...")
        X, y = generate_training_data_from_pd(n_samples=20000)
        
        # Save data for future use
        np.save('drone_training_inputs.npy', X)
        np.save('drone_training_outputs.npy', y)
    
    # 2. Create and train the neural network controller
    controller = SimpleDroneController()
    history = controller.train(X, y, epochs=30, batch_size=64)
    
    # 3. Plot the training history
    plot_training_history(history)
    
    # 4. Save the trained model
    controller.save_model('drone_nn_controller.h5')
    print("Neural network controller trained and saved to 'drone_nn_controller.h5'")
    
    # 5. Test the controller on the reference scenario
    print("Testing neural network controller...")
    time_points, states, inputs, references = simulate_test_scenario(controller)
    
    # 6. Plot the results
    plot_results(time_points, states, inputs, references)
    
    print("Testing completed. Results saved to:")
    print("- nn_training_history.png")
    print("- nn_drone_stabilization.png")
    print("- nn_drone_control_inputs.png")
    

if __name__ == "__main__":
    # Use this simpler implementation to avoid recursion errors
    run_simple_nn_controller()