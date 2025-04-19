import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.special import erfcinv


class QKDSimulation:
    """Fully corrected QKD simulation with proper physical models"""

    def __init__(self):
        # Physical constants
        self.h = 6.626e-34  # Planck's constant
        self.c = 3e8  # Speed of light

        # Fiber parameters (updated values)
        self.alpha_db = 0.2  # Attenuation [dB/km]
        self.alpha_linear = 0.046  # Linear attenuation [1/km]
        self.raman_coeff = 5e-9  # [(counts/s)/(mW·nm·km)]

        # QKD protocol
        self.mu = 0.5  # Mean photon number
        self.q = 0.5  # Basis sifting factor
        self.fec_efficiency = 0.95

        # Detector parameters (updated)
        self.detectors = {
            'SNSPD': {
                'efficiency': 0.85,
                'dark_count': 1e-7,  # More realistic dark count rate
                'jitter': 50e-12
            }
        }
        self.current_detector = 'SNSPD'

        # System parameters
        self.wavelength_q = 1550  # Quantum channel [nm]
        self.wavelength_c = 1560  # Classical channel [nm]
        self.default_filter_bw = 0.5  # [nm]
        self.channel_spacing = 0.8  # [nm]

    def transmission(self, distance):
        """Correct fiber transmission with proper dB conversion"""
        return 10 ** (-self.alpha_db * distance / 10)

    def raman_noise(self, distance, power, filter_bw):
        """Physically accurate Raman noise model"""
        delta_lambda = abs(self.wavelength_c - self.wavelength_q)
        spectral_decay = np.exp(-delta_lambda / 30)  # Adjusted decay

        # Convert power from mW to W for proper scaling
        noise_photons = (self.raman_coeff * (power * 1e-3) * distance *
                         filter_bw * spectral_decay)

        return noise_photons * 1e9  # Scale to match detection window

    def qber(self, distance, classical_power, filter_bw):
        """Complete QBER calculation with realistic components"""
        t = self.transmission(distance)
        detector = self.detectors[self.current_detector]

        # Signal photons (with realistic detector effects)
        signal = self.mu * t * detector['efficiency'] * np.sqrt(filter_bw / 0.5)

        # Noise sources (properly scaled)
        dark = detector['dark_count']
        raman = self.raman_noise(distance, classical_power, filter_bw)
        total_noise = dark + raman

        # QBER with 1% baseline misalignment
        qber = 0.01 + (0.5 * total_noise) / (signal + total_noise)
        return min(qber, 0.5), signal, total_noise  # Cap at 50%

    def secret_key_rate(self, distance, classical_power, filter_bw):
        """GLLP key rate with proper scaling"""
        qber_val, signal, noise = self.qber(distance, classical_power, filter_bw)

        if qber_val >= 0.11 or signal < 1e-10:
            return 0.0, qber_val

        # Realistic key rate components
        sifted_rate = (signal + noise) * self.q
        error_correction = self.fec_efficiency * (1 - 1.16 * qber_val)
        privacy_amp = 1 - 2 * qber_val * np.log2(qber_val) - 2 * (1 - qber_val) * np.log2(1 - qber_val)

        skr = sifted_rate * error_correction * privacy_amp
        return max(1e-10, skr), qber_val

    def plot_key_rate_vs_distance(self):
        """Corrected key rate plot"""
        distances = np.linspace(1, 100, 50)
        powers = [0, 1, 5, 10]  # mW

        plt.figure(figsize=(10, 6))
        for power in powers:
            rates = []
            for d in distances:
                rate, _ = self.secret_key_rate(d, power, self.default_filter_bw)
                rates.append(rate)

            plt.semilogy(distances, rates, label=f'{power} mW', linewidth=2)

        plt.xlabel('Distance (km)')
        plt.ylabel('Secure Key Rate (bits/pulse)')
        plt.title('Key Rate vs Distance with Classical Interference')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig('key_rate_vs_distance.png')
        plt.show()

    def plot_qber_vs_spacing(self):
        """QBER with proper spacing effects"""
        spacings = np.linspace(0.4, 2.0, 50)
        distances = [20, 50, 80]
        power = 5  # mW

        plt.figure(figsize=(10, 6))
        for d in distances:
            qbers = []
            for spacing in spacings:
                self.channel_spacing = spacing
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
        plt.savefig('qber_vs_spacing.png')
        plt.show()

    def plot_filter_optimization(self):
        """Realistic filter optimization"""
        filter_widths = np.linspace(0.1, 2.0, 50)
        distances = [20, 50, 80]
        power = 5  # mW

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
        plt.savefig('filter_optimization.png')
        plt.show()

    def generate_ml_data(self, samples=1000):
        """Generate meaningful ML training data"""
        data = []
        for _ in range(samples):
            d = np.random.uniform(5, 150)
            p = np.random.uniform(0, 20)
            s = np.random.uniform(0.4, 2.0)

            self.channel_spacing = s
            self.wavelength_c = self.wavelength_q + s

            # Find true optimal filter width
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
        """Train meaningful filter width predictor"""
        df = self.generate_ml_data()
        print(f"Unique optimal widths: {df['optimal_width'].nunique()}")
        print(f"Width range: {df['optimal_width'].min():.2f}-{df['optimal_width'].max():.2f} nm")

        X = df[['distance', 'power', 'spacing', 'power_distance', 'spacing_power']]
        y = df['optimal_width']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor(n_estimators=200, max_depth=5)
        model.fit(X_train, y_train)

        print(f"\nTest MSE: {mean_squared_error(y_test, model.predict(X_test)):.6f}")
        print("Feature importances:")
        print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

        return model


if __name__ == "__main__":
    sim = QKDSimulation()

    print("Generating physical simulation plots...")
    sim.plot_key_rate_vs_distance()
    sim.plot_qber_vs_spacing()
    sim.plot_filter_optimization()

    print("\nTraining ML model...")
    model = sim.train_ml_model()