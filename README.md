# fin0

Various financial mathematics codes in Python for testing purposes.  
See comment in files for what each does, but explanation is very limited.  
Includes:
- Analytical pricing of European options under Black-Scholes-Merton, with Greeks
- Implied volatility calculation
- Monte Carlo simulation (and European option pricing) for
  - Black-Scholes-Merton (with and without importance sampling)
  - Cox-Ingersoll-Ross
  - Heston (with and without importance sampling and antithetic variates)
  - gamma
  - variance gamma
  - bilateral gamma
  - Poisson
- American option pricing
  - binomial tree
  - Longstaff-Schwartz
  - de-Americanization
- Carr-Madan FFT pricing of European options for
  - Black-Scholes-Merton
  - variance gamma
  - bilateral gamma
- PDE pricing
  - Black-Scholes PDE

## Some notes

While not necessary, full paths are always generated in Monte Carlo simulations for pricing, so that path-dependent options may be priced also.  
Scripts should be ran from within this directory.  
Plots are written to the out folder in this directory.

## Some plots

### Implied volatility calculation

![BSM_E_vol](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/BSM_E_vol.png)

### American option pricing under Black-Scholes-Merton via the binomial tree

![BT_BSM_AE](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/BT_BSM_AE.png)

### Black-Scholes-Merton Monte Carlo simulation

![MC_BSM](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/MC_BSM.png)

### Cox-Ingersoll-Ross Monte Carlo simulation

![MC_CIR](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/MC_CIR.png)

### Heston Monte Carlo simulation

![MC_Heston_1](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/MC_Heston_1.png)
![MC_Heston_2](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/MC_Heston_2.png)

### Variance gamma Monte Carlo simulation

![MC_VG_2](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/MC_VG_2.png)

### Bilateral gamma Monte Carlo simulation

![MC_BG_2](https://raw.githubusercontent.com/olafx/fin0/refs/heads/main/out/MC_BG_2.png)
