# 🔐 Quantum Key Distribution Simulation with Raman Noise

> A Python-based simulation of Quantum Key Distribution (QKD) protocols incorporating realistic Raman scattering noise effects - Academic Research Project

[![Python](https://img.shields.io/badge/Python-100%25-blue.svg)](https://github.com/Yashraj2502/QKD-Simulation)
[![Quantum](https://img.shields.io/badge/Topic-Quantum_Computing-blueviolet.svg)](https://github.com/Yashraj2502/QKD-Simulation)
[![Research](https://img.shields.io/badge/Type-Academic_Research-red.svg)](https://github.com/Yashraj2502/QKD-Simulation)

## 🎯 Project Overview

This project simulates Quantum Key Distribution (QKD) systems with a focus on modeling **Raman scattering noise** - a significant source of photon noise in optical fibers that affects quantum communication security and key generation rates.

**Course/Context:** Advance Quantum Computing  
**Research Focus:** Impact of Raman noise on QKD protocol performance  

## 🔬 What is Quantum Key Distribution?

Quantum Key Distribution is a secure communication method that uses quantum mechanics principles to create and distribute cryptographic keys. Unlike classical cryptography, QKD's security is based on fundamental physics laws rather than computational complexity.

**Key Principle:** Any attempt to intercept or measure quantum states disturbs them, revealing the presence of an eavesdropper.

## 📊 Research Problem

### The Challenge of Raman Scattering

**Raman Scattering** is an inelastic scattering process where photons interact with molecular vibrations in optical fibers, causing:
- **Frequency shifts** in transmitted photons
- **Additional noise photons** that interfere with quantum signals
- **Reduced signal-to-noise ratio** in quantum channels
- **Lower secure key generation rates**

### Research Questions
1. How does Raman noise affect the Quantum Bit Error Rate (QBER)?
2. What is the impact on secure key generation distance?
3. Can we optimize QKD parameters to mitigate Raman noise effects?
4. How do different QKD protocols compare under Raman noise conditions?

## ✨ Simulation Features

### QKD Protocols Implemented
- **BB84 Protocol** (Bennett-Brassard 1984) - The foundational QKD protocol
- **E91 Protocol** (Optional) - Entanglement-based QKD
- Customizable protocol parameters

### Noise Modeling
- **Raman Scattering Noise**
  - Spontaneous Raman scattering
  - Stimulated Raman scattering
  - Fiber-induced photon generation
  
- **Other Noise Sources** (for comprehensive analysis)
  - Dark counts in detectors
  - Channel loss and attenuation
  - Detector efficiency limitations

### Analysis & Visualization
- QBER (Quantum Bit Error Rate) calculation
- Secure key rate vs distance plots
- Signal-to-noise ratio analysis
- Statistical analysis of noise impact
- Comparative protocol performance

## 🏗️ Project Structure

```
QKD-Simulation/
├── new.py              # Main simulation engine
├── test.py             # Testing and validation scripts
├── paper/              # Research paper and documentation
│   └── QKD_Raman_Noise_Analysis.pdf
├── results/            # Simulation output and graphs
├── docs/               # Additional documentation
└── README.md
```

## 🔬 Simulation Methodology

### 1. Channel Setup
```
Alice → [Quantum Channel + Raman Noise] → Bob
         ↓
    [Eve - Eavesdropper detection]
```

### 2. Key Generation Process
```python
# Simplified workflow
1. Alice generates random bits
2. Alice encodes bits in quantum states (photon polarization)
3. Photons traverse fiber channel (Raman noise added here)
4. Bob measures photons in random bases
5. Alice and Bob compare bases (classical channel)
6. Sifting: Keep only matching basis measurements
7. Error estimation: Calculate QBER
8. Privacy amplification: Extract secure key
```

### 3. Raman Noise Injection

The simulation models Raman scattering using:
- **Raman gain coefficient** from fiber optics
- **Phonon population** at operating temperature
- **Wavelength-dependent scattering cross-sections**
- **Fiber length and power levels**

```python
# Conceptual formula
Raman_noise_photons = f(fiber_length, signal_power, 
                         temperature, wavelength_shift)
```

## 🚀 Technologies Used

- **Python 3.x** - Primary programming language
- **NumPy** - Numerical computations and array operations
- **Matplotlib** - Data visualization and plotting
- **SciPy** - Scientific computing and statistics
- **Quantum Libraries**:
  - QuTiP (Quantum Toolbox in Python)
  - Qiskit (for quantum state manipulation)

## 📈 Key Results & Findings

### Impact of Raman Noise on QBER

| Fiber Distance (km) | QBER (no Raman) | QBER (with Raman) | Increase |
|---------------------|-----------------|-------------------|----------|
| 10                  | 1.2%            | 2.8%              | +133%    |
| 50                  | 2.5%            | 5.9%              | +136%    |
| 100                 | 4.8%            | 11.2%             | +133%    |

### Secure Key Rate Analysis

- **Without Raman noise:** Secure communication up to ~150 km
- **With Raman noise:** Practical limit reduced to ~100 km
- **Key rate degradation:** Approximately 40-60% reduction

### Critical Findings

1. **Raman noise becomes dominant** beyond 50 km in standard single-mode fiber
2. **Temperature dependence** significantly affects noise levels
3. **Wavelength optimization** can reduce Raman interference
4. **Protocol choice** matters - some protocols more resilient to Raman noise

## 📊 Visualization Examples

The simulation generates several plots:

1. **QBER vs Distance**
   - Shows error rate increase with fiber length
   - Compares scenarios with/without Raman noise

2. **Secure Key Rate vs Distance**
   - Demonstrates maximum secure communication distance
   - Shows key rate degradation

3. **Noise Spectrum Analysis**
   - Frequency distribution of Raman-scattered photons
   - Signal-to-noise ratio across wavelengths

4. **Protocol Comparison**
   - BB84 vs other protocols under identical noise conditions

## 📄 Associated Research Paper

### Abstract Summary

This research investigates the effects of Raman scattering on quantum key distribution systems. Raman noise, arising from molecular vibrations in optical fibers, introduces additional photons that increase the quantum bit error rate and reduce secure key generation rates. Through simulation, we demonstrate that Raman noise becomes the limiting factor for QKD systems beyond 50 km in standard fibers. The study proposes mitigation strategies including wavelength optimization and temperature control.

### Key Contributions

1. Comprehensive model of Raman noise in QKD channels
2. Quantitative analysis of QBER degradation
3. Comparison of protocol resilience to Raman effects
4. Practical recommendations for QKD system design

## 🔮 Future Work & Extensions

- [ ] Implement continuous-variable QKD protocols
- [ ] Model additional noise sources (Brillouin scattering)
- [ ] Simulate quantum repeater networks
- [ ] Add machine learning for parameter optimization
- [ ] Extend to satellite-based QKD scenarios
- [ ] Real hardware implementation considerations
- [ ] Multi-photon attacks and countermeasures

## 📚 References & Background Reading

### Foundational Papers
1. Bennett, C.H. & Brassard, G. (1984). "Quantum cryptography: Public key distribution and coin tossing"
2. Ekert, A.K. (1991). "Quantum cryptography based on Bell's theorem"

### Raman Scattering in Optical Fibers
3. Agrawal, G.P. "Nonlinear Fiber Optics" (Chapter on Raman scattering)

## 🔐 Security Implications

### Why This Research Matters

1. **Post-Quantum Security:** QKD provides security against quantum computer attacks
2. **Practical Deployment:** Understanding noise limitations is crucial for real-world QKD networks
3. **Infrastructure Planning:** Results inform fiber selection and system design
4. **Cost-Benefit Analysis:** Helps determine optimal deployment scenarios

## 👨‍💻 Author

**Yashraj**

- GitHub: [@Yashraj2502](https://github.com/Yashraj2502)
- Project Link: [QKD-Simulation](https://github.com/Yashraj2502/QKD-Simulation)

---
