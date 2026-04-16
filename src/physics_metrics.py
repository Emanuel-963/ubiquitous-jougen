import numpy as np


def effective_capacitance(df):
    """Compute the effective capacitance from impedance data.

    Derives frequency-dependent capacitance from the imaginary part of
    the impedance, filtering out non-physical values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ``zreal``, ``zimag``, and ``omega``.

    Returns
    -------
    numpy.ndarray
        Array of effective capacitance values (F). May be empty if no
        valid data points remain after filtering.
    """
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
    """Estimate the series (ohmic) resistance from high-frequency impedance.

    Uses the median of the first *n* real-impedance values (highest
    frequencies) for robustness against outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a ``zreal`` column, sorted by descending frequency.
    n : int, optional
        Number of high-frequency points to consider (default 5).

    Returns
    -------
    float
        Estimated series resistance (Ω), or ``np.nan`` if no data.
    """
    # Rs é o valor em alta frequência (primeiras linhas após sort desc)
    rs_vals = df.head(n)["zreal"].values
    # Usar mediana para robustez contra outliers
    return np.median(rs_vals) if len(rs_vals) > 0 else np.nan


def polarization_resistance(df, n=5):
    """Estimate the polarization resistance from impedance endpoints.

    Computes Rp as the difference between the low-frequency and
    high-frequency real impedance medians, clamped to zero.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a ``zreal`` column, sorted by descending frequency.
    n : int, optional
        Number of points at each frequency extreme to consider (default 5).

    Returns
    -------
    float
        Estimated polarization resistance (Ω), non-negative.
    """
    # Rp = Z(0Hz) - Z(inf)
    hf = np.median(df.head(n)["zreal"].values)
    lf = np.median(df.tail(n)["zreal"].values)
    rp = lf - hf
    # Garantir que Rp seja positivo e razoável
    return max(rp, 0.0)


def stored_energy(c_eff, voltage=1.0):
    """Calculate the energy stored in a capacitor array.

    Uses the relation E = 0.5 · C · V².

    Parameters
    ----------
    c_eff : numpy.ndarray
        Effective capacitance values (F).
    voltage : float, optional
        Applied voltage (V). Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Stored energy values (J). Empty array if *c_eff* is empty.
    """
    if len(c_eff) == 0:
        return np.array([])
    return 0.5 * c_eff * voltage**2


def dominant_tau(rp, c_eff):
    """Compute the dominant time constant from Rp and capacitance.

    Defined as τ = Rp × median(C_eff).

    Parameters
    ----------
    rp : float
        Polarization resistance (Ω).
    c_eff : numpy.ndarray
        Effective capacitance values (F).

    Returns
    -------
    float
        Dominant time constant (s), or ``np.nan`` if inputs are invalid.
    """
    if len(c_eff) == 0 or np.isnan(rp):
        return np.nan
    return rp * np.median(c_eff)


def dispersion_index(c_eff):
    """Compute the dispersion index of the effective capacitance.

    Measures the spread of capacitance values in log-space as the
    standard deviation of log10(C_eff).

    Parameters
    ----------
    c_eff : numpy.ndarray
        Effective capacitance values (F).

    Returns
    -------
    float
        Standard deviation of log10(C_eff), or ``np.nan`` if fewer
        than 2 values are available.
    """
    if len(c_eff) < 2:
        return np.nan
    return np.std(np.log10(c_eff))


def extract_features(df):
    """Extract a dictionary of physics-based features from EIS data.

    Aggregates series resistance, polarization resistance, capacitance
    statistics, stored energy, dominant time constant, and dispersion
    index into a single feature dictionary.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ``zreal``, ``zimag``, and ``omega``,
        sorted by descending frequency.

    Returns
    -------
    dict
        Feature dictionary with keys: ``Rs``, ``Rp``, ``C_mean``,
        ``C_max``, ``C_lowfreq``, ``Energy_mean``, ``Tau``,
        ``Dispersion``.
    """
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
