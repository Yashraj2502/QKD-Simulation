import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.special import erfcinv


class QKDSimulation:
    """Complete QKD simulation with all fixes implemented"""

    def __init__(self):
        # Physical constants
        self.h = 6.626e-34  # Planck's constant
        self.c = 3e8  # Speed of light

        # Fiber parameters
        self.alpha_db = 0.2  # Attenuation [dB/km]
        self.raman_coeff = 7e-9  # Raman coefficient [(counts/s)/(mW·nm·km)]

        # Protocol parameters
        self.mu = 0.5  # Mean photon number
        self.fec_efficiency = 0.95

        # Detector parameters
        self.detectors = {
            'SNSPD': {
                'efficiency': 0.85,
                'dark_count': 5e-8,
                'jitter': 50e-12
            }
        }
        self.current_detector = 'SNSPD'

        # System configuration
        self.wavelength_q = 1550  # Quantum channel [nm]
        self.wavelength_c = 1560  # Classical channel [nm]
        self.default_filter_bw = 0.5  # Default filter bandwidth [nm]

    def transmission(self, distance):
        """Fiber transmission loss"""
        return 10 ** (-self.alpha_db * distance / 10)

    def raman_noise(self, distance, power, filter_bw):
        """Raman scattering noise calculation"""
        delta_lambda = abs(self.wavelength_c - self.wavelength_q)
        spectral_decay = np.exp(-delta_lambda / 25)  # Empirical decay

        return (self.raman_coeff * (power * 1e-3) * distance *
                filter_bw * spectral_decay * 1e9)  # Scaled to detection window

    def qber(self, distance, classical_power, filter_bw):
        """Quantum Bit Error Rate calculation"""
        t = self.transmission(distance)
        detector = self.detectors[self.current_detector]

        # Signal with bandwidth-dependent efficiency
        signal = self.mu * t * detector['efficiency'] * (filter_bw / 0.5) ** 0.75

        # Noise components
        dark = detector['dark_count']
        raman = self.raman_noise(distance, classical_power, filter_bw)
        total_noise = dark + raman

        # QBER with 1% baseline and cap at 0.5
        qber_val = min(0.01 + (0.5 * total_noise) / (signal + total_noise), 0.5)
        return qber_val, signal, total_noise

    def secret_key_rate(self, distance, classical_power, filter_bw):
        """Secure key rate calculation"""
        qber_val, signal, noise = self.qber(distance, classical_power, filter_bw)

        if qber_val >= 0.11 or signal < 1e-10:
            return 0.0, qber_val

        # Complete key rate components
        sifted_rate = (signal + noise) * 0.5  # Basis sifting
        error_correction = self.fec_efficiency * (1 - 1.16 * qber_val)
        privacy_amp = 1 - 2 * qber_val * np.log2(qber_val) - 2 * (1 - qber_val) * np.log2(1 - qber_val)

        return max(1e-10, sifted_rate * error_correction * privacy_amp), qber_val

    def plot_key_rate_vs_distance(self):
        """Plot key rate vs distance"""
        distances = np.linspace(1, 100, 50)
        powers = [0, 1, 5, 10]

        plt.figure(figsize=(10, 6))
        for power in powers:
            rates = [self.secret_key_rate(d, power, self.default_filter_bw)[0]
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
        plt.show()

    def plot_qber_vs_spacing(self):
        """Plot QBER vs channel spacing"""
        spacings = np.linspace(0.4, 2.0, 50)
        distances = [20, 50, 80]
        power = 5

        plt.figure(figsize=(10, 6))
        for d in distances:
            qbers = []
            for spacing in spacings:
                self.wavelength_c = self.wavelength_q + spacing
                qber_val, _, _ = self.qber(d, power, self.default_filter_bw)
                qbers.append(qber_val)

            plt.plot(spacings, qbers, label=f'{d} km', linewidth=2)

        plt.axhline(y=0.11, color='r', linestyle='--', label='QBER threshold')
        plt.xlabel('Channel Spacing (nm)')
        plt.ylabel('QBER')
        plt.title('QBER vs Channel Spacing (5 mW Classical Power)')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 0.5)
        plt.tight_layout()
        # plt.savefig('qber_vs_spacing.png', dpi=300)
        plt.show()

    def plot_filter_optimization(self):
        """Plot filter bandwidth optimization"""
        filter_widths = np.linspace(0.1, 2.0, 50)
        distances = [20, 50, 80]
        power = 5

        plt.figure(figsize=(10, 6))
        for d in distances:
            rates = []
            for bw in filter_widths:
                rate, _ = self.secret_key_rate(d, power, bw)
                rates.append(rate)

            opt_idx = np.argmax(rates)
            plt.plot(filter_widths, rates, label=f'{d} km', linewidth=2)
            plt.axvline(x=filter_widths[opt_idx], linestyle='--', alpha=0.3)
            plt.text(filter_widths[opt_idx], max(rates) / 2,
                     f'Opt: {filter_widths[opt_idx]:.2f}nm', rotation=90)

        plt.xlabel('Filter Bandwidth (nm)')
        plt.ylabel('Secure Key Rate (bits/pulse)')
        plt.title('Filter Bandwidth Optimization (5 mW Classical Power)')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, None)  # Ensure no negative values
        plt.tight_layout()
        # plt.savefig('filter_optimization.png', dpi=300)
        plt.show()

    def generate_ml_data(self, samples=1000):
        """Generate training data for ML model"""
        data = []
        for _ in range(samples):
            d = np.random.uniform(5, 150)
            p = np.random.uniform(0, 20)
            s = np.random.uniform(0.4, 2.0)

            self.wavelength_c = self.wavelength_q + s

            # Find optimal filter width
            widths = np.linspace(0.1, 2.0, 20)
            rates = [self.secret_key_rate(d, p, w)[0] for w in widths]
            opt_width = widths[np.argmax(rates)]

            if max(rates) > 1e-6:  # Only keep viable cases
                data.append({
                    'distance': d,
                    'power': p,
                    'spacing': s,
                    'optimal_width': opt_width,
                    'power_distance': p * d,
                    'spacing_power': s / max(0.1, p)
                })

        return pd.DataFrame(data)

    def train_ml_model(self):
        """Train ML model for filter optimization"""
        df = self.generate_ml_data(2000)
        print(f"Unique optimal widths: {df['optimal_width'].nunique()}")
        print(f"Width range: {df['optimal_width'].min():.2f}-{df['optimal_width'].max():.2f} nm")

        X = df[['distance', 'power', 'spacing', 'power_distance', 'spacing_power']]
        y = df['optimal_width']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5)
        model.fit(X_train, y_train)

        print(f"\nTest MSE: {mean_squared_error(y_test, model.predict(X_test)):.6f}")
        print("Feature importances:")
        print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

        return model


if __name__ == "__main__":
    plt.close('all')
    sim = QKDSimulation()

    print("Generating simulation plots...")
    sim.plot_key_rate_vs_distance()
    sim.plot_qber_vs_spacing()
    sim.plot_filter_optimization()

    print("\nTraining ML model...")
    model = sim.train_ml_model()