import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns


class QKDSimulation:
    """
    Simulation of a multiplexed QKD system with classical and quantum channels
    sharing the same optical fiber using WDM technology.
    """

    def __init__(self, debug=False):
        # Debug mode flag
        self.debug = debug

        # System Constants
        self.h = 6.626e-34  # Planck's constant [J·s]
        self.c = 3e8  # Speed of light [m/s]

        # Fiber Parameters
        self.alpha_db = 0.2  # Fiber attenuation [dB/km]
        self.alpha = self.alpha_db / (10 * np.log10(np.e))  # Linear attenuation coefficient [1/km]

        # QKD Protocol Parameters
        self.mu = 0.5  # Mean photon number
        self.q = 0.5  # Sifting factor (for BB84)
        self.error_correction_factor = 1.16  # Error correction inefficiency
        self.privacy_amplification_factor = 1.1  # Privacy amplification overhead

        # Detector Parameters
        self.detector_types = {
            'InGaAs': {
                'efficiency': 0.25,  # Quantum efficiency
                'dark_count': 1e-6,  # Dark count probability per detection window
                'time_window': 1e-9  # Detection window [s]
            },
            'SSPD': {
                'efficiency': 0.85,  # Higher quantum efficiency
                'dark_count': 1e-9,  # Lower dark count
                'time_window': 1e-9  # Detection window [s]
            }
        }
        self.current_detector = 'InGaAs'  # Default detector

        # Default WDM Parameter Settings
        self.wavelength_quantum = 1550  # Quantum channel wavelength [nm]
        self.wavelength_classical = 1570  # Classical channel wavelength [nm]
        self.filter_width = 0.8  # Quantum channel filter width [nm]
        self.classical_power = 1.0  # Classical channel power [mW]
        self.channel_spacing = 20  # Spacing between quantum and classical [nm]

        if self.debug:
            print(f"Initialized QKD Simulation with:\n"
                  f"- Alpha: {self.alpha:.6f} 1/km\n"
                  f"- Mean photon number: {self.mu}\n"
                  f"- Default filter width: {self.filter_width} nm\n"
                  f"- Channel spacing: {self.channel_spacing} nm")

    def transmission(self, distance):
        """Calculate optical transmission over fiber distance"""
        trans = np.exp(-self.alpha * distance)
        if self.debug and (distance in [10, 50, 100]):
            loss_db = -10 * np.log10(trans)
            print(f"Transmission at {distance} km: {trans:.4e} ({loss_db:.1f} dB loss)")
        return trans

    def calculate_raman_noise(self, distance, classical_power, filter_width):
        """
        Calculate Raman scattering noise photon count rate
        Args:
            distance: Fiber length in km
            classical_power: Classical signal power in mW
            filter_width: Quantum channel filter width in nm
        Returns:
            Raman noise count probability per detection window
        """
        # Constants for Raman noise model - adjusted for more realistic values
        beta = 7e-9  # Raman coefficient [(count/s)/(nm·mW·km)]

        # Calculate wavelength difference
        delta_lambda = abs(self.wavelength_classical - self.wavelength_quantum)

        # Raman noise scales with distance, classical power, and filter width
        # but reduces with channel spacing - exponential decay with spacing
        raman_factor = np.exp(-0.04 * delta_lambda)

        # Total count rate from Raman scattering
        raman_counts = beta * classical_power * distance * filter_width * raman_factor

        # Convert to probability within detection window
        detector = self.detector_types[self.current_detector]
        raman_probability = raman_counts * detector['time_window']

        if self.debug and (distance in [10, 50]) and (classical_power in [1, 5]):
            print(f"Raman noise at {distance} km, {classical_power} mW: {raman_probability:.2e}")

        return raman_probability

    def calculate_qber(self, distance, classical_power, filter_width):
        """Calculate Quantum Bit Error Rate"""
        # Transmission loss
        t = self.transmission(distance)

        # Signal detection probability (per sent qubit)
        detector = self.detector_types[self.current_detector]
        signal_prob = self.mu * t * detector['efficiency']

        # Noise sources
        dark_count_prob = detector['dark_count']
        raman_prob = self.calculate_raman_noise(distance, classical_power, filter_width)
        total_noise_prob = dark_count_prob + raman_prob

        # Quantum Bit Error Rate calculation
        # QBER = (0.5 × noise) / (signal + noise)
        # 0.5 factor because noise contributes random bits (50% error)
        if signal_prob + total_noise_prob <= 0:
            qber = 0.5  # Maximum QBER if no signal
        else:
            qber = (0.5 * total_noise_prob) / (signal_prob + total_noise_prob)

        if self.debug and (distance in [10, 50]) and (classical_power in [1, 5]):
            print(f"At {distance} km, {classical_power} mW:")
            print(f"  Signal: {signal_prob:.2e}, Noise: {total_noise_prob:.2e}, QBER: {qber:.4f}")

        return qber, signal_prob, total_noise_prob

    def calculate_key_rate(self, distance, classical_power, filter_width):
        """Calculate secure key rate"""
        qber, signal_prob, noise_prob = self.calculate_qber(distance, classical_power, filter_width)

        # No key possible if QBER is too high
        if qber >= 0.11:  # Approximate threshold for BB84
            return 0, qber

        # Raw key rate (bits per sent pulse)
        raw_rate = signal_prob + noise_prob

        # Simplified secure key fraction formula based on BB84 with decoy states
        h_binary = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) if 0 < x < 1 else 0
        secure_fraction = max(0, 1 - 2 * h_binary(qber))

        # Apply inefficiency factors
        secure_fraction /= self.error_correction_factor

        # Final key rate
        key_rate = self.q * raw_rate * secure_fraction

        if self.debug and (distance in [10, 50, 100]) and (classical_power in [0, 5]):
            print(f"Key rate at {distance} km, {classical_power} mW: {key_rate:.2e}")

        return max(0, key_rate), qber

    def set_detector(self, detector_type):
        """Change detector type"""
        if detector_type in self.detector_types:
            self.current_detector = detector_type
            if self.debug:
                det = self.detector_types[detector_type]
                print(f"Changed to {detector_type} detector (eff: {det['efficiency']}, dark: {det['dark_count']})")
            return True
        return False

    def plot_key_rate_vs_distance(self, distances=None, classical_powers=None):
        """Plot key rate vs. distance for different classical powers"""
        if self.debug:
            print("\n=== Plotting Key Rate vs Distance ===")

        if distances is None:
            distances = np.linspace(0, 100, 100)  # 0 to 100 km

        if classical_powers is None:
            classical_powers = [0, 1, 5, 10]  # Different classical powers in mW

        plt.figure(figsize=(10, 6))

        for power in classical_powers:
            key_rates = []
            qbers = []

            for d in distances:
                key_rate, qber = self.calculate_key_rate(d, power, self.filter_width)
                key_rates.append(key_rate)
                qbers.append(qber)

                # Debug output at specific points
                if self.debug and d in [10, 50, 80] and power == classical_powers[0]:
                    print(f"  At d={d} km, power={power} mW: Key Rate = {key_rate:.2e}, QBER = {qber:.4f}")

            plt.semilogy(distances, key_rates, label=f'Power: {power} mW')

            # Check if we're getting all zeros
            if np.all(np.array(key_rates) < 1e-15):
                print(f"WARNING: Key rates for power={power} mW are all effectively zero!")

        plt.xlabel('Distance (km)')
        plt.ylabel('Secure Key Rate (per pulse)')
        plt.title('Key Rate vs. Distance for Different Classical Powers')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()

    def plot_qber_vs_spacing(self, spacings=None, distances=None):
        """Plot QBER vs. channel spacing for different distances"""
        if self.debug:
            print("\n=== Plotting QBER vs Channel Spacing ===")

        if spacings is None:
            spacings = np.linspace(10, 100, 50)  # 10 to 100 nm

        if distances is None:
            distances = [10, 30, 50]  # km

        plt.figure(figsize=(10, 6))

        original_spacing = self.channel_spacing
        original_wavelength = self.wavelength_classical

        for d in distances:
            qbers = []

            for spacing in spacings:
                # Update wavelength to reflect spacing
                self.wavelength_classical = self.wavelength_quantum + spacing
                self.channel_spacing = spacing

                # Calculate QBER with fixed classical power
                fixed_power = 5  # mW
                qber, _, _ = self.calculate_qber(d, fixed_power, self.filter_width)
                qbers.append(qber)

                # Debug output at specific points
                if self.debug and spacing in [10, 50, 90] and d == distances[0]:
                    print(f"  At spacing={spacing} nm, d={d} km: QBER = {qber:.4f}")

            plt.plot(spacings, qbers, label=f'Distance: {d} km')

        # Reset to original values
        self.wavelength_classical = original_wavelength
        self.channel_spacing = original_spacing

        plt.xlabel('Channel Spacing (nm)')
        plt.ylabel('QBER')
        plt.title('QBER vs. Channel Spacing for Different Distances')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return plt.gcf()

    def plot_raman_vs_power(self, powers=None, distances=None):
        """Plot Raman noise vs. classical power for different distances"""
        if self.debug:
            print("\n=== Plotting Raman Noise vs Classical Power ===")

        if powers is None:
            powers = np.linspace(0, 20, 50)  # 0 to 20 mW

        if distances is None:
            distances = [10, 30, 50]  # km

        plt.figure(figsize=(10, 6))

        for d in distances:
            raman_noise = []

            for power in powers:
                noise = self.calculate_raman_noise(d, power, self.filter_width)
                raman_noise.append(noise)

                # Debug output at specific points
                if self.debug and power in [1, 10, 20] and d == distances[0]:
                    print(f"  At power={power} mW, d={d} km: Raman noise = {noise:.2e}")

            plt.plot(powers, raman_noise, label=f'Distance: {d} km')

        plt.xlabel('Classical Power (mW)')
        plt.ylabel('Raman Noise (probability per detection window)')
        plt.title('Raman Noise vs. Classical Power for Different Distances')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return plt.gcf()

    def plot_filter_vs_keyrate(self, filter_widths=None, distances=None):
        """Plot key rate vs. filter width for different distances"""
        if self.debug:
            print("\n=== Plotting Key Rate vs Filter Width ===")

        if filter_widths is None:
            filter_widths = np.linspace(0.1, 2.0, 50)  # 0.1 to 2.0 nm

        if distances is None:
            distances = [10, 30, 50]  # km

        plt.figure(figsize=(10, 6))

        original_filter_width = self.filter_width
        fixed_power = 5  # mW

        for d in distances:
            key_rates = []
            optimal_width = None
            max_key_rate = 0

            for width in filter_widths:
                self.filter_width = width
                key_rate, _ = self.calculate_key_rate(d, fixed_power, width)
                key_rates.append(key_rate)

                if key_rate > max_key_rate:
                    max_key_rate = key_rate
                    optimal_width = width

                # Debug output at specific points
                if self.debug and width in [0.2, 0.8, 1.5] and d == distances[0]:
                    print(f"  At width={width} nm, d={d} km: Key rate = {key_rate:.2e}")

            plt.plot(filter_widths, key_rates, label=f'Distance: {d} km')

            if optimal_width:
                plt.axvline(x=optimal_width, linestyle='--', color='gray', alpha=0.5)
                plt.text(optimal_width, max_key_rate, f'  Optimal: {optimal_width:.2f} nm',
                         verticalalignment='bottom')

        # Reset filter width
        self.filter_width = original_filter_width

        plt.xlabel('Filter Width (nm)')
        plt.ylabel('Secure Key Rate (per pulse)')
        plt.title('Key Rate vs. Filter Width for Different Distances')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return plt.gcf()

    def plot_detector_comparison(self, distances=None):
        """Compare detector performance (InGaAs vs SSPD) at different distances"""
        if self.debug:
            print("\n=== Plotting Detector Comparison ===")

        if distances is None:
            distances = np.linspace(0, 100, 100)  # 0 to 100 km

        plt.figure(figsize=(10, 6))
        fixed_power = 5  # mW

        for detector_type in ['InGaAs', 'SSPD']:
            key_rates = []

            # Set detector type
            self.set_detector(detector_type)

            for d in distances:
                key_rate, _ = self.calculate_key_rate(d, fixed_power, self.filter_width)
                key_rates.append(key_rate)

                # Debug output at specific points
                if self.debug and d in [10, 50, 80] and detector_type == 'InGaAs':
                    print(f"  At d={d} km, detector={detector_type}: Key rate = {key_rate:.2e}")

            plt.semilogy(distances, key_rates, label=f'Detector: {detector_type}')

        plt.xlabel('Distance (km)')
        plt.ylabel('Secure Key Rate (per pulse)')
        plt.title('Detector Performance Comparison (InGaAs vs SSPD)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()

    def generate_training_data(self, samples=1000):
        """Generate training data for ML optimization"""
        if self.debug:
            print(f"\n=== Generating {samples} Training Samples ===")

        data = []

        for i in range(samples):
            # Random parameters within realistic ranges
            distance = np.random.uniform(5, 80)  # 5-80 km
            classical_power = np.random.uniform(0.1, 15)  # 0.1-15 mW (avoid zero power)
            filter_width = np.random.uniform(0.1, 2.0)  # 0.1-2.0 nm
            spacing = np.random.uniform(15, 80)  # 15-80 nm wavelength spacing

            # Set spacing
            original_wavelength = self.wavelength_classical
            self.wavelength_classical = self.wavelength_quantum + spacing

            # Calculate key rate
            key_rate, qber = self.calculate_key_rate(distance, classical_power, filter_width)

            # Store parameters and results
            data.append({
                'distance': distance,
                'classical_power': classical_power,
                'filter_width': filter_width,
                'spacing': spacing,
                'key_rate': key_rate,
                'qber': qber
            })

            # Debug output occasionally
            if self.debug and i % 200 == 0:
                print(f"  Sample {i}: d={distance:.1f}, p={classical_power:.1f}, " +
                      f"w={filter_width:.2f}, key_rate={key_rate:.2e}, qber={qber:.4f}")

            # Reset wavelength
            self.wavelength_classical = original_wavelength

        df = pd.DataFrame(data)

        # Check for all zeros
        if self.debug:
            zero_rates = (df['key_rate'] < 1e-15).sum()
            if zero_rates > 0:
                print(f"WARNING: {zero_rates} samples have effectively zero key rate!")

        return df

    def train_ml_model(self, df=None):
        """Train ML model to predict optimal filter width"""
        if self.debug:
            print("\n=== Training ML Model ===")

        if df is None:
            df = self.generate_training_data()

        # Generate optimized dataset
        # For each distance/power/spacing combination, find filter width that maximizes key rate
        param_combinations = []
        optimal_filters = []

        # Sample key parameter space with grid
        distances = np.linspace(10, 80, 8)
        powers = np.linspace(0.1, 15, 6)
        spacings = np.linspace(20, 80, 4)
        filter_widths = np.linspace(0.1, 2.0, 20)

        if self.debug:
            print("Finding optimal filter widths across parameter space...")

        for d in distances:
            for p in powers:
                for s in spacings:
                    # Set spacing
                    original_wavelength = self.wavelength_classical
                    self.wavelength_classical = self.wavelength_quantum + s

                    # Try different filter widths to find optimum
                    best_rate = 0
                    best_width = 0.1

                    for fw in filter_widths:
                        key_rate, _ = self.calculate_key_rate(d, p, fw)
                        if key_rate > best_rate:
                            best_rate = key_rate
                            best_width = fw

                    # Store optimal result if key rate is non-zero
                    if best_rate > 1e-15:
                        param_combinations.append([d, p, s])
                        optimal_filters.append(best_width)

                    # Reset wavelength
                    self.wavelength_classical = original_wavelength

        # Convert to arrays
        X = np.array(param_combinations)
        y = np.array(optimal_filters)

        if self.debug:
            print(f"Training data shape: {X.shape}, Target shape: {y.shape}")
            print(f"Filter width range: {y.min():.2f} - {y.max():.2f} nm")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model (Random Forest for its ability to capture complex relationships)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        if self.debug:
            print(f"Model RMSE: {rmse:.6f}")

            # Feature importance
            importances = model.feature_importances_
            feature_names = ['Distance', 'Classical Power', 'Channel Spacing']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            print("Feature importance:")
            for _, row in importance_df.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")

        return model, rmse

    def validate_ml_predictions(self, model, test_samples=20):
        """Validate ML predictions by comparing key rates"""
        if self.debug:
            print(f"\n=== Validating ML Model with {test_samples} Samples ===")

        # Generate test points
        distances = np.linspace(10, 70, test_samples)
        classical_powers = np.linspace(1, 10, test_samples)
        spacings = np.linspace(20, 60, test_samples)

        results = []

        for i in range(test_samples):
            distance = distances[i]
            classical_power = classical_powers[i]
            spacing = spacings[i]

            # Set wavelength spacing
            original_wavelength = self.wavelength_classical
            self.wavelength_classical = self.wavelength_quantum + spacing

            # Find baseline key rate with standard filter width
            baseline_filter = 0.8  # Standard filter width
            key_rate_baseline, _ = self.calculate_key_rate(distance, classical_power, baseline_filter)

            # Predict optimal filter width using ML model
            features = np.array([[distance, classical_power, spacing]])
            predicted_filter = model.predict(features)[0]

            # Calculate key rate with predicted filter width
            key_rate_predicted, _ = self.calculate_key_rate(distance, classical_power, predicted_filter)

            # Find true optimal filter width by scanning
            filter_widths = np.linspace(0.1, 2.0, 30)
            best_rate = 0
            optimal_filter = 0.1

            for fw in filter_widths:
                key_rate, _ = self.calculate_key_rate(distance, classical_power, fw)
                if key_rate > best_rate:
                    best_rate = key_rate
                    optimal_filter = fw

            results.append({
                'distance': distance,
                'classical_power': classical_power,
                'spacing': spacing,
                'baseline_filter': baseline_filter,
                'predicted_filter': predicted_filter,
                'optimal_filter': optimal_filter,
                'key_rate_baseline': key_rate_baseline,
                'key_rate_predicted': key_rate_predicted,
                'key_rate_optimal': best_rate,
                'ml_improvement': (key_rate_predicted / key_rate_baseline if key_rate_baseline > 0 else 1.0),
                'ml_accuracy': (key_rate_predicted / best_rate if best_rate > 0 else 0.0)
            })

            # Debug output
            if self.debug and i % 5 == 0:
                print(f"  Test {i + 1}: d={distance:.1f}, p={classical_power:.1f}")
                print(f"    Baseline filter: {baseline_filter} nm → Rate: {key_rate_baseline:.2e}")
                print(f"    ML filter: {predicted_filter:.2f} nm → Rate: {key_rate_predicted:.2e}")
                print(f"    Optimal filter: {optimal_filter:.2f} nm → Rate: {best_rate:.2e}")
                print(f"    Improvement over baseline: {results[-1]['ml_improvement']:.2f}x")

            # Reset wavelength
            self.wavelength_classical = original_wavelength

        return pd.DataFrame(results)

    def plot_ml_vs_keyrate(self, model=None, rmse=None):
        """Plot key rate improvement with ML-optimized filter width"""
        if self.debug:
            print("\n=== Evaluating ML Filter Optimization Performance ===")

        if model is None:
            model, rmse = self.train_ml_model()

        results = self.validate_ml_predictions(model, test_samples=30)

        # Calculate statistics
        avg_improvement = results['ml_improvement'].mean()
        max_improvement = results['ml_improvement'].max()
        avg_accuracy = results['ml_accuracy'].mean() * 100  # as percentage

        if self.debug:
            print(f"ML performance metrics:")
            print(f"  Average improvement over fixed filter: {avg_improvement:.2f}x")
            print(f"  Maximum improvement: {max_improvement:.2f}x")
            print(f"  Average accuracy (% of optimal): {avg_accuracy:.1f}%")
            print(f"  Filter width RMSE: {rmse:.4f} nm")

        # Plot 1: Improvement factor vs distance with power as color
        plt.figure(figsize=(12, 8))

        plt.scatter(results['distance'], results['ml_improvement'],
                    c=results['classical_power'], cmap='viridis',
                    alpha=0.7, s=50)

        plt.colorbar(label='Classical Power (mW)')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5,
                    label='Baseline (no improvement)')

        plt.xlabel('Distance (km)')
        plt.ylabel('Key Rate Improvement Factor')
        plt.title(
            f'ML Filter Optimization Results\nAvg Improvement: {avg_improvement:.2f}x, Max: {max_improvement:.2f}x')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot 2: Filter width comparison
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(results['distance'], results['baseline_filter'], alpha=0.5, label='Fixed (0.8 nm)')
        plt.scatter(results['distance'], results['predicted_filter'], alpha=0.5, label='ML Predicted')
        plt.scatter(results['distance'], results['optimal_filter'], alpha=0.5, label='True Optimal')
        plt.xlabel('Distance (km)')
        plt.ylabel('Filter Width (nm)')
        plt.title('Filter Width Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        sns.kdeplot(results['baseline_filter'], label='Fixed', fill=True, alpha=0.3)
        sns.kdeplot(results['predicted_filter'], label='ML Predicted', fill=True, alpha=0.3)
        sns.kdeplot(results['optimal_filter'], label='True Optimal', fill=True, alpha=0.3)
        plt.xlabel('Filter Width (nm)')
        plt.ylabel('Density')
        plt.title('Filter Width Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        return results


# Example usage
if __name__ == "__main__":
    # Create simulation with debugging enabled
    sim = QKDSimulation(debug=True)

    print("Running QKD Simulation with debugging output:")

    # 1. Key Rate vs Distance
    print("\nGenerating Key Rate vs Distance plot...")
    sim.plot_key_rate_vs_distance(distances=np.linspace(0, 100, 50))
    # plt.savefig('key_rate_vs_distance.png')

    # 2. QBER vs Channel Spacing
    print("\nGenerating QBER vs Channel Spacing plot...")
    sim.plot_qber_vs_spacing(spacings=np.linspace(10, 100, 30))
    # plt.savefig('qber_vs_spacing.png')

    # 3. Raman Noise vs Classical Power
    print("\nGenerating Raman Noise vs Classical Power plot...")
    sim.plot_raman_vs_power(powers=np.linspace(0, 20, 30))
    # plt.savefig('raman_vs_power.png')

    # 4. Key Rate vs Filter Width
    print("\nGenerating Key Rate vs Filter Width plot...")
    sim.plot_filter_vs_keyrate(filter_widths=np.linspace(0.1, 2.0, 30))
    # plt.savefig('filter_vs_keyrate.png')

    # 5. Detector Comparison
    print("\nGenerating Detector Comparison plot...")
    sim.plot_detector_comparison(distances=np.linspace(0, 100, 50))
    # plt.savefig('detector_comparison.png')

    # 6. ML Optimization
    print("\nTraining ML model for filter width optimization...")
    model, rmse = sim.train_ml_model()
    print(f"ML Filter BW Prediction RMSE: {rmse}")

    # 7. ML predictions vs actual
    print("\nEvaluating ML model performance...")
    results = sim.plot_ml_vs_keyrate(model, rmse)
    # plt.savefig('ml_improvements.png')