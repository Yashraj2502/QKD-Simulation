import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fontTools.misc.plistlib import end_real
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


''' Constants '''
h = 6.626e-34 # J*s
c = 3e8 # m/s


# Raman Efficiency Model (Synthetic for now)
def raman_efficiency(delta_lambda):
    # Eta(lambda) peak around 100 nm Strokes shift, drops on either side
    return np.exp(-((delta_lambda - 100) / 30)**2) * 1e-9   # 1/W/m

# Raman Power Calculation
def raman_power(Pp, L, alpha, delta_lambda):
    eta = raman_efficiency(delta_lambda)
    return eta * Pp * L * np.exp(-alpha * L)    # Simplified forward Raman power

''' Photon Count Estimation '''
def power_to_photon(P, lambda_nm, delta_f, tau):
    freq = c/(lambda_nm * 1e-9)
    energy = h * freq
    return (P * delta_f * tau) / energy

''' QBER & Key Rate '''
def compute_qber(signal, noise):
    return noise / (signal + noise)

def compute_key_rate(signal, noise, protocol='DPS'):
    qber = compute_qber(signal, noise)
    if qber > 0.11:
        return 0
    return signal * (1 -2 * qber)

''' Simulation Wrapper '''
def simulation_qkd(filter_bw_nm, channel_spacing_nm, P_classical_dBm):
    lambda_q = 1550 # m
    lambda_c = lambda_q + channel_spacing_nm
    delta_lambda = lambda_c - lambda_q

    P_classical = 10 ** ((P_classical_dBm - 30) / 10)   # Convert dBm to Watts
    fiber_len = 25 # km
    alpha = 0.2 / 4.343 #dB/km to 1/m
    L = fiber_len * 1000 # m

    delta_f = filter_bw_nm * 1.25e11    # 1 nm ~ 125 GHz
    tau = 300e-12   # 300 ps

    P_raman = raman_power(P_classical, L, alpha, delta_lambda)
    n_raman = power_to_photon(P_raman, lambda_q, delta_f, tau)

    # Assuming signal is 1 photon per pulse
    signal = 1
    noise = n_raman + 5e-5  # Include dark count

    qber = compute_qber(signal, noise)
    rate = compute_key_rate(signal, noise)
    return qber, rate, n_raman

''' ML- Based Filter Optimization '''
def ml_filter_optimizer():
    x = []
    y = []
    key_rates = []
    for spacing in np.linspace(20, 100, 20):
        for power in np.linspace(-40, -20, 10):
            for bw in np.linspace(0.1, 1.0, 10):
                qber, rate, _ = simulation_qkd(bw, spacing, power)
                x.append([spacing, power])
                y.append(bw)
                key_rates.append(rate)
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print("ML Filter BW Prediction RMSE:", rmse)
    return model

''' Plot 1: Key Rate v/s Distance '''
def plot_key_distance():
    distance = np.linspace(0, 100, 50)
    rates = []
    for d in distance:
        lambda_q = 1550
        lambda_c = 1600
        delta_lambda = lambda_c - lambda_q
        P = 10 ** ((-30 - 30) / 10)
        alpha = 0.2 / 4.343
        delta_f = 0.5 * 1.25e11
        tau = 300e-12

        P_raman = raman_power(P, d*1000, alpha, delta_lambda)
        n_raman = power_to_photon(P_raman, lambda_q, delta_f, tau)
        rate = compute_key_rate(1, n_raman + 5e-5)
        rates.append(rate)

    plt.figure()
    sns.lineplot(x=distance, y=rates)
    plt.xlabel("Distance (km)")
    plt.ylabel("Final Key Rate (a.u.)")
    plt.title("Key Rate v/s Distance")
    plt.grid(True)
    # plt.savefig("key_rate_vs_distance.png")
    plt.show()

''' Plot 2: QBER vs Channel Spacing '''
def plot_qber_vs_channel():
    spacing = np.linspace(10, 100, 50)
    qbers = [simulation_qkd(0.5, s, -30)[0] for s in spacing]
    plt.figure()
    sns.lineplot(x=spacing, y=qbers)
    plt.xlabel("Channel Spacing (nm)")
    plt.ylabel("QBER")
    plt.title("QBER vs Channel Spacing")
    plt.grid(True)
    # plt.savefig("qber_vs_channel.png")
    plt.show()

''' Plot 3: Raman Noise vs Classical Power '''
def plot_raman_power():
    powers = np.linspace(-40, 20, 50)
    noise = [simulation_qkd(0.5, 5, p)[2] for p in powers]
    plt.figure()
    sns.lineplot(x=powers, y=noise)
    plt.xlabel("Classical Power (dBm)")
    plt.ylabel("Raman Noise (Photons)")
    plt.title("Raman Noise vs Classical Power")
    plt.grid(True)
    # plt.savefig("raman_power.png")
    plt.show()

''' Plot 4: Filter Width vs Key Rate '''
def plot_filter_width_vs_key_rate():
    bws = np.linspace(0.1, 1.0, 20)
    rates = [simulation_qkd(bw, 50, -30)[1] for bw in bws]
    plt.figure()
    sns.lineplot(x=bws, y=rates)
    plt.xlabel("Filter Width (nm)")
    plt.ylabel("Key Rate (a.u.)")
    plt.title("Filter Width vs Key Rate")
    plt.grid(True)
    # plt.savefig("filter_width_vs_key_rate.png")
    plt.show()

''' Plot 5: InGaAs vs SSPD Comparison '''
def plot_detector_comparison():
    distances = np.linspace(0, 100, 50)
    ingaas = []
    sspd = []
    for d in distances:
        n1 = simulation_qkd(0.5, 50, -30)[2] + 5e-5
        ingaas.append(compute_key_rate(1, n1))
        n2 = simulation_qkd(0.5, 50, -30)[2] + 2.5e-9
        sspd.append(compute_key_rate(1, n2))

    plt.figure()
    plt.plot(distances, ingaas, label = 'InGaAs')
    plt.plot(distances, sspd, label = 'SSPD')
    plt.xlabel("Distance (km)")
    plt.ylabel("Key Rate (a.u.)")
    plt.title("InGaAs vs SSPD Comparison")
    plt.grid(True)
    plt.legend()
    # plt.savefig("detector_comparison.png")
    plt.show()

''' Plot 6: ML Output vs Key Rate Boost '''
def plot_ml_optimization():
    model = ml_filter_optimizer()
    text_x = np.array([[50, -30], [70, -35], [90, -40]])
    predicitons = model.predict(text_x)
    base_rate = [simulation_qkd(0.5, x[0], x[1])[1] for x in text_x]
    optimized_rate = [simulation_qkd(p, x[0], x[1])[1] for p, x in zip(predicitons, text_x)]

    plt.figure()
    indices = range(len(text_x))
    plt.bar(indices, base_rate, width=0.4, label='Base Rate')
    plt.bar([i + 0.4 for i in indices], optimized_rate, width=0.4, label='ML Optimized')
    plt.xticks([i + 0.2 for i in indices], [f"Case {i+1}" for i in indices])
    plt.ylabel("Key Rate (a.u.)")
    plt.title("ML Output vs Key Rate Boost")
    plt.legend()
    plt.grid(True)
    # plt.savefig("ml_keyrate_boost.png")
    plt.show()


''' Main Execution '''
if __name__ == "__main__":
    print("Running QKD Simulation:")
    plot_key_distance()
    # ml_model = ml_filter_optimizer()
    plot_qber_vs_channel()
    plot_raman_power()
    plot_filter_width_vs_key_rate()
    plot_detector_comparison()
    plot_ml_optimization()







