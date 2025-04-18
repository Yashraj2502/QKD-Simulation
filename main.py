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
    for spacing in np.linspace(20, 100, 20):
        for power in np.linspace(-40, -20, 10):
            for bw in np.linspace(0.1, 1.0, 10):
                qber, rate, _ = simulation_qkd(bw, spacing, power)
                x.append([spacing, power])
                y.append(bw)
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

''' Main Execution '''
if __name__ == "__main__":
    print("Running QKD Simulation:")
    plot_key_distance()
    ml_model = ml_filter_optimizer()








