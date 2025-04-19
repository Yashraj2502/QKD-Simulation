import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from collections import Counter


class QKDSimulation:
    """
    Simulation of a multiplexed QKD system with classical and quantum channels
    sharing the same optical fiber using WDM technology
    """

    def __init__(self):
        # System Constants
        self.h = 6.626e-34  # Planck's constant [J·s]
        self.c = 3e8  # Speed of light [m/s]

        # Fiber Parameters
        self.alpha_db = 0.2  # Fiber attenuation [dB/km]
        self.alpha = self.alpha_db / (10 * np.log10(np.e))  # Linear attenuation coefficient [1/km]
        self.raman_coeff = 7e-9 # Raman coefficient [(counts/s)/(mW·nm·km)]

        # QKD Protocol Parameters
        self.mu = 0.5  # Mean photon number
        self.q = 0.5  # Sifting factor (for BB84)
        self.error_correction_factor = 1.15  # Error correction inefficiency
        self.privacy_amplification_factor = 1.1  # Privacy amplification overhead
        self.fec_efficiency = 0.95

        # Detector Parameters
        self.detector_types = {
            'InGaAs': {
                'efficiency': 0.25,  # Quantum efficiency
                'dark_count': 1e-6,  # Dark count probability
                'time_window': 1e-9  # Detection window [s]
            },
            'SSPD': {
                'efficiency': 0.85,  # Higher quantum efficiency
                'dark_count': 5e-8,  # Lower dark count
                'jitter': 50e-12  # Detection window [s]
            }
        }
        self.current_detector = 'SSPD'  # Default detector

        # Default WDM Parameter Settings
        self.wavelength_quantum = 1550  # Quantum channel wavelength [nm]
        self.wavelength_classical = 1560  # Classical channel wavelength [nm]
        self.filter_width = 0.5  # Quantum channel filter width [nm]
        self.classical_power = 0  # Classical channel power [mW]
        self.channel_spacing = 20  # Spacing between quantum and classical [nm]

    def transmission(self, distance):
        """Calculate optical transmission over fiber distance"""
        return 10 ** (-self.alpha_db * distance / 10)

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
        delta_lambda = abs(self.wavelength_classical - self.wavelength_quantum)
        spectral_decay = np.exp(-delta_lambda / 25)  # Empirical decay

        return (self.raman_coeff * (classical_power * 1e-3) * distance *
                filter_width * spectral_decay * 1e9)  # Scaled to detection window

    def calculate_qber(self, distance, classical_power, filter_width):
        """Calculate Quantum Bit Error Rate"""
        # Transmission loss
        t = self.transmission(distance)

        # Signal detection probability (per sent qubit)
        detector = self.detector_types[self.current_detector]

        # Signal with bandwidth-dependent efficiency
        signal = self.mu * t * detector['efficiency'] * (filter_width / 0.5) ** 0.75

        # Noise components
        dark = detector['dark_count']
        raman = self.calculate_raman_noise(distance, classical_power, filter_width)
        total_noise = dark + raman

        # QBER with 1% baseline and cap at 0.5
        qber_val = min(0.01 + (0.5 * total_noise) / (signal + total_noise), 0.5)
        return qber_val, signal, total_noise


    def calculate_key_rate(self, distance, classical_power, filter_width):
        """Calculate secure key rate"""
        qber, signal_prob, noise_prob = self.calculate_qber(distance, classical_power, filter_width)

        if qber >= 0.11 or signal_prob < 1e-10:
            return 0.0, qber

        # Complete key rate components
        sifted_rate = (signal_prob + noise_prob) * 0.5  # Basis sifting
        error_correction = self.fec_efficiency * (1 - 1.16 * qber)
        privacy_amp = 1 - 2 * qber * np.log2(qber) - 2 * (1 - qber) * np.log2(1 - qber)

        return max(1e-10, sifted_rate * error_correction * privacy_amp), qber

    def set_detector(self, detector_type):
        """Change detector type"""
        if detector_type in self.detector_types:
            self.current_detector = detector_type
            return True
        return False

    def plot_key_rate_vs_distance(self, distances=None, classical_powers=None):
        """Plot key rate vs. distance for different classical powers"""
        if distances is None:
            distances = np.linspace(0, 100, 100)  # 0 to 100 km

        if classical_powers is None:
            classical_powers = [0, 1, 5, 10]  # Different classical powers in mW

        plt.figure(figsize=(10, 6))

        for power in classical_powers:
            rates = [self.calculate_key_rate(d, power, self.filter_width)[0]
                     for d in distances]
            plt.semilogy(distances, rates, label=f'{power} mW', linewidth=2)

        plt.xlabel('Distance (km)')
        plt.ylabel('Secure Key Rate (bits/pulse)')
        plt.title('Key Rate vs Distance with Classical Interference')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.ylim(1e-5, 1e0)  # Adjusted y-axis limits
        plt.tight_layout()
        # plt.savefig('key_rate_vs_distance.png', dpi=300)
        return plt.gcf()

    def plot_qber_vs_spacing(self, spacings=None, distances=None):
        """Plot QBER vs. channel spacing for different distances"""
        if spacings is None:
            spacings = np.linspace(0.4, 2.0, 50)  # 10 to 100 nm

        if distances is None:
            distances = [20, 50, 80]  # km

        plt.figure(figsize=(10, 6))

        original_spacing = self.channel_spacing
        original_wavelength = self.wavelength_classical

        for d in distances:
            qbers = []

            for spacing in spacings:
                # Update wavelength to reflect spacing
                self.wavelength_classical = self.wavelength_quantum + spacing
                # self.channel_spacing = spacing

                # Calculate QBER with fixed classical power
                fixed_power = 5  # mW
                qber, val, _ = self.calculate_qber(d, fixed_power, self.filter_width)
                qbers.append(qber)

            plt.plot(spacings, qbers, label=f'Distance: {d} km', linewidth=2)

        # Reset to original values
        self.wavelength_classical = original_wavelength
        self.channel_spacing = original_spacing

        plt.axhline(y=0.11, color='r', linestyle='--', label='QBER threshold')
        plt.xlabel('Channel Spacing (nm)')
        plt.ylabel('QBER')
        plt.title('QBER vs Channel Spacing (5 mW Classical Power)')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 0.5)
        plt.tight_layout()
        return plt.gcf()

    def plot_raman_vs_power(self, powers=None, distances=None):
        """Plot Raman noise vs. classical power for different distances"""
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
        if filter_widths is None:
            filter_widths = np.linspace(0.1, 2.0, 50)  # 0.1 to 2.0 nm

        if distances is None:
            distances = [20, 50, 80]  # km

        plt.figure(figsize=(10, 6))

        original_filter_width = self.filter_width
        fixed_power = 5  # mW

        for d in distances:
            key_rates = []
            optimal_width = None
            max_key_rate = 0

            for width in filter_widths:
                self.filter_width = width
                key_rate, qber = self.calculate_key_rate(d, fixed_power, width)
                key_rates.append(key_rate)

                if key_rate > max_key_rate:
                    max_key_rate = key_rate
                    optimal_width = width

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
        data = []
        optimal_widths = []  # For diagnostics

        for _ in range(samples):
            # Random parameters within realistic ranges
            distance = np.random.uniform(5, 80)  # 5-80 km
            classical_power = np.random.uniform(0, 15)  # 0-15 mW
            spacing = np.random.uniform(15, 80)  # 15-80 nm wavelength spacing

            # Set spacing
            original_wavelength = self.wavelength_classical
            self.wavelength_classical = self.wavelength_quantum + spacing

            # Find optimal filter width for these parameters
            filter_widths = np.linspace(0.1, 2.0, 20)  # Test 20 different widths
            best_key_rate = 0
            optimal_filter_width = 0.1

            for width in filter_widths:
                key_rate, qber = self.calculate_key_rate(distance, classical_power, width)
                if key_rate > best_key_rate:
                    best_key_rate = key_rate
                    optimal_filter_width = width

            optimal_widths.append(optimal_filter_width)  # Store for diagnostics

            # Store parameters and results
            data.append({
                'distance': distance,
                'classical_power': classical_power,
                'filter_width': optimal_filter_width,  # This is now truly optimal
                'spacing': spacing,
                'key_rate': best_key_rate,
                'qber': qber  # This will be the QBER at the optimal filter width
            })

            # Reset wavelength
            self.wavelength_classical = original_wavelength

        # Diagnostic output
        unique_widths = len(set(optimal_widths))
        print(f"Number of unique optimal filter widths: {unique_widths}")
        print(f"Min: {min(optimal_widths)}, Max: {max(optimal_widths)}")
        print(f"Distribution: {Counter(optimal_widths).most_common(5)}")

        return pd.DataFrame(data)

    def train_ml_model(self, df=None):
        """Train ML model to predict optimal filter width"""
        if df is None:
            df = self.generate_training_data()

            # Check data variance
            print(f"Filter width variance: {df['filter_width'].var()}")
            if df['filter_width'].var() < 0.001:
                print("WARNING: Very low variance in filter width data!")

            # Feature engineering
            df['power_distance'] = df['classical_power'] * df['distance']
            df['spacing_power'] = df['spacing'] / df['classical_power'].replace(0, 0.1)

            # Features and target
            X = df[['distance', 'classical_power', 'spacing', 'power_distance', 'spacing_power']]
            y = df['filter_width']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model (Random Forest for its ability to capture complex relationships)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Model MSE: {mse:.6f}")

            # Feature importance
            feat_importances = pd.Series(model.feature_importances_, index=X.columns)
            print("Feature importances:")
            print(feat_importances.sort_values(ascending=False))

            return model

    def validate_ml_predictions(self, model, test_samples=20):
        """Validate ML predictions by comparing key rates"""
        # Generate new test data with random filter widths
        test_data = []

        for _ in range(test_samples):
            distance = np.random.uniform(5, 80)  # 5-80 km
            classical_power = np.random.uniform(0, 15)  # 0-15 mW
            spacing = np.random.uniform(15, 80)  # 15-80 nm wavelength spacing

            # Random filter width (not optimal)
            random_filter = np.random.uniform(0.1, 2.0)

            test_data.append({
                'distance': distance,
                'classical_power': classical_power,
                'spacing': spacing,
                'filter_width': random_filter
            })

        test_df = pd.DataFrame(test_data)
        results = []

        for i, row in test_df.iterrows():
            distance = row['distance']
            classical_power = row['classical_power']
            spacing = row['spacing']
            random_filter = row['filter_width']

            # Set wavelength spacing
            original_wavelength = self.wavelength_classical
            self.wavelength_classical = self.wavelength_quantum + spacing

            # Calculate key rate with random filter width
            key_rate_random, _ = self.calculate_key_rate(distance, classical_power, random_filter)

            # Predict optimal filter width using ML model
            features = pd.DataFrame({
                'distance': [distance],
                'classical_power': [classical_power],
                'spacing': [spacing],
                'power_distance': [distance * classical_power],
                'spacing_power': [spacing / max(0.1, classical_power)]
            })

            predicted_filter = model.predict(features)[0]

            # Calculate key rate with predicted filter width
            key_rate_predicted, _ = self.calculate_key_rate(distance, classical_power, predicted_filter)

            # Also calculate the truly optimal filter width for comparison
            filter_widths = np.linspace(0.1, 2.0, 20)
            best_key_rate = 0
            optimal_filter = 0.1

            for width in filter_widths:
                key_rate, _ = self.calculate_key_rate(distance, classical_power, width)
                if key_rate > best_key_rate:
                    best_key_rate = key_rate
                    optimal_filter = width

            results.append({
                'distance': distance,
                'classical_power': classical_power,
                'spacing': spacing,
                'random_filter': random_filter,
                'predicted_filter': predicted_filter,
                'optimal_filter': optimal_filter,
                'key_rate_random': key_rate_random,
                'key_rate_predicted': key_rate_predicted,
                'key_rate_optimal': best_key_rate,
                'improvement_over_random': (key_rate_predicted / key_rate_random if key_rate_random > 0 else 1.0),
                'optimality_ratio': (key_rate_predicted / best_key_rate if best_key_rate > 0 else 1.0)
            })

            # Reset wavelength
            self.wavelength_classical = original_wavelength

        return pd.DataFrame(results)

    def plot_ml_vs_keyrate(self, model=None):
        """Plot key rate improvement with ML-optimized filter width"""
        if model is None:
            model = self.train_ml_model()

        results = self.validate_ml_predictions(model, test_samples=50)

        # Calculate statistics
        avg_improvement = results['improvement_over_random'].mean()
        max_improvement = results['improvement_over_random'].max()
        avg_optimality = results['optimality_ratio'].mean()

        plt.figure(figsize=(12, 8))

        # Scatter plot of improvements
        plt.scatter(results['distance'], results['improvement_over_random'],
                    c=results['classical_power'], cmap='viridis',
                    alpha=0.7, s=50)

        plt.colorbar(label='Classical Power (mW)')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)

        plt.xlabel('Distance (km)')
        plt.ylabel('Key Rate Improvement Factor (over random)')
        plt.title(
            f'ML Filter Optimization Results\nAvg Improvement: {avg_improvement:.2f}x, Max: {max_improvement:.2f}x, Optimality: {avg_optimality:.2f}')
        plt.grid(True, alpha=0.3)

        # Add a second plot for filter width comparison
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(results['distance'], results['random_filter'], alpha=0.5, label='Random')
        plt.scatter(results['distance'], results['predicted_filter'], alpha=0.5, label='ML Predicted')
        plt.scatter(results['distance'], results['optimal_filter'], alpha=0.5, label='True Optimal')
        plt.xlabel('Distance (km)')
        plt.ylabel('Filter Width (nm)')
        plt.title('Filter Width Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Use only if values have variance
        if results['random_filter'].var() > 0:
            sns.kdeplot(results['random_filter'], label='Random', fill=True, alpha=0.3)
        if results['predicted_filter'].var() > 0:
            sns.kdeplot(results['predicted_filter'], label='ML Predicted', fill=True, alpha=0.3)
        if results['optimal_filter'].var() > 0:
            sns.kdeplot(results['optimal_filter'], label='True Optimal', fill=True, alpha=0.3)
        plt.xlabel('Filter Width (nm)')
        plt.ylabel('Density')
        plt.title('Filter Width Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Report the correct metrics
        print("Results summary:")
        print(f"Average key rate improvement: {results['improvement_over_random'].mean():.2f}x")
        print(f"Maximum key rate improvement: {results['improvement_over_random'].max():.2f}x")
        print(f"ML predictions vs optimal: {results['optimality_ratio'].mean():.2f}")

        return results


# Example usage
if __name__ == "__main__":
    plt.close('all')
    sim = QKDSimulation()

    # Run all simulations
    print("Generating plots...")

    # 1. Key Rate vs Distance with different classical powers
    sim.plot_key_rate_vs_distance()
    plt.savefig('key_rate_vs_distance.png')

    # 2. QBER vs Channel Spacing
    sim.plot_qber_vs_spacing()
    plt.savefig('qber_vs_spacing.png')

    # 3. Raman Noise vs Classical Power
    sim.plot_raman_vs_power()
    plt.savefig('raman_vs_power.png')

    # 4. Key Rate vs Filter Width
    sim.plot_filter_vs_keyrate()
    plt.savefig('filter_vs_keyrate.png')

    # 5. Detector Comparison
    sim.plot_detector_comparison()
    plt.savefig('detector_comparison.png')

    # 6. Train ML model and show improvements
    print("Training ML model for filter width optimization...")
    model = sim.train_ml_model()

    # 7. ML predictions vs actual
    print("Validating ML model performance...")
    results = sim.plot_ml_vs_keyrate(model)
    plt.savefig('ml_improvements.png')

    plt.show()