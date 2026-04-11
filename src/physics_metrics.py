import numpy as np


def effective_capacitance(df):
    z = df["zreal"].values + 1j * df["zimag"].values
    omega = df["omega"].values

    mask = (omega > 0) & (np.abs(z) > 1e-6)
    z = z[mask]
    omega = omega[mask]

    if len(z) == 0:
        return np.array([])

    c_eff = -np.imag(z) / (omega * np.abs(z) ** 2)
    # Filtrar valores fisicamente razoáveis
    c_eff = c_eff[(c_eff > 1e-15) & (c_eff < 1e-2)]

    return c_eff


def series_resistance(df, n=5):
    # Rs é o valor em alta frequência (primeiras linhas após sort desc)
    rs_vals = df.head(n)["zreal"].values
    # Usar mediana para robustez contra outliers
    return np.median(rs_vals) if len(rs_vals) > 0 else np.nan


def polarization_resistance(df, n=5):
    # Rp = Z(0Hz) - Z(inf)
    hf = np.median(df.head(n)["zreal"].values)
    lf = np.median(df.tail(n)["zreal"].values)
    rp = lf - hf
    # Garantir que Rp seja positivo e razoável
    return max(rp, 0.0)


def stored_energy(c_eff, voltage=1.0):
    if len(c_eff) == 0:
        return np.array([])
    return 0.5 * c_eff * voltage**2


def dominant_tau(rp, c_eff):
    if len(c_eff) == 0 or np.isnan(rp):
        return np.nan
    return rp * np.median(c_eff)


def dispersion_index(c_eff):
    if len(c_eff) < 2:
        return np.nan
    return np.std(np.log10(c_eff))


def extract_features(df):
    c_eff = effective_capacitance(df)
    rs = series_resistance(df)
    rp = polarization_resistance(df)

    # Capacitância na menor frequência disponível (após ordenação desc em preprocess)
    c_low = np.nan
    if len(c_eff) > 0:
        c_low = c_eff[-1]

    return {
        "Rs": rs,
        "Rp": rp,
        "C_mean": np.nan if len(c_eff) == 0 else np.mean(c_eff),
        "C_max": np.nan if len(c_eff) == 0 else np.max(c_eff),
        "C_lowfreq": c_low,
        "Energy_mean": (np.nan if len(c_eff) == 0 else np.mean(stored_energy(c_eff))),
        "Tau": dominant_tau(rp, c_eff),
        "Dispersion": dispersion_index(c_eff),
    }
