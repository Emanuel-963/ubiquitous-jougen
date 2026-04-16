import numpy as np
from scipy.optimize import least_squares


def cpe_impedance(omega, Q, n):
    """Compute the impedance of a Constant Phase Element (CPE).

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequency array in rad/s.
    Q : float
        CPE pseudo-capacitance parameter in S·s^n.
    n : float
        CPE exponent (0 < n <= 1). n=1 corresponds to an ideal capacitor.

    Returns
    -------
    numpy.ndarray
        Complex impedance of the CPE at each frequency.
    """
    return 1 / (Q * (1j * omega) ** n)


def warburg_impedance(omega, sigma):
    """Compute the Warburg diffusion impedance.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequency array in rad/s.
    sigma : float
        Warburg coefficient.

    Returns
    -------
    numpy.ndarray
        Complex Warburg impedance at each frequency.
    """
    return sigma / np.sqrt(1j * omega)


def model_impedance(params, omega):
    """Compute the total impedance of an Rs + (Rp || CPE) + Warburg circuit.

    Parameters
    ----------
    params : array-like
        Model parameters ``[Rs, Rp, Q, n, sigma]``.
    omega : numpy.ndarray
        Angular frequency array in rad/s.

    Returns
    -------
    numpy.ndarray
        Complex impedance of the full equivalent circuit.
    """
    Rs, Rp, Q, n, sigma = params
    Zcpe = cpe_impedance(omega, Q, n)
    Zw = warburg_impedance(omega, sigma)
    Zpar = 1 / (1 / Rp + 1 / Zcpe)
    return Rs + Zpar + Zw


def residuals(params, omega, Z_exp):
    """Compute residuals between model and experimental impedance.

    Parameters
    ----------
    params : array-like
        Model parameters ``[Rs, Rp, Q, n, sigma]``.
    omega : numpy.ndarray
        Angular frequency array in rad/s.
    Z_exp : numpy.ndarray
        Experimental complex impedance data.

    Returns
    -------
    numpy.ndarray
        Concatenated real and imaginary residuals.
    """
    Z_model = model_impedance(params, omega)
    return np.concatenate([(Z_model.real - Z_exp.real), (Z_model.imag - Z_exp.imag)])


def fit_cpe_warburg(df):
    """Fit a CPE + Warburg equivalent circuit model to EIS data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ``'omega'``, ``'zreal'``, and ``'zimag'``.

    Returns
    -------
    dict
        Fitted parameters (``Rs_fit``, ``Rp_fit``, ``Q``, ``n``, ``Sigma``),
        ``Fit_error`` (mean squared residual), and ``Converged`` flag.

    Raises
    ------
    ValueError
        If the DataFrame contains fewer than 3 data points.
    """
    omega = df["omega"].values
    Z_exp = df["zreal"].values + 1j * df["zimag"].values

    if len(df) < 3:
        raise ValueError(f"Insuficientes pontos de dados: {len(df)}")

    # Palpite inicial baseado em características dos dados
    # Rs: primeira (menor frequência) é tipicamente Rs
    # Rp: diferença entre baixa e alta frequência
    rs_initial = float(df["zreal"].iloc[-1])  # maior frequência (Rs puro)
    rp_initial = float(df["zreal"].iloc[0] - df["zreal"].iloc[-1])  # dinâmica total
    rp_initial = max(rp_initial, 0.1)  # garantir valor positivo mínimo

    p0 = [
        max(rs_initial, 0.01),  # Rs
        max(rp_initial, 0.1),  # Rp
        1e-4,  # Q
        0.85,  # n
        0.01,  # Sigma
    ]

    bounds = (
        [1e-6, 1e-6, 1e-12, 0.3, 1e-10],  # lower bounds
        [1e6, 1e8, 1.0, 1.0, 1e5],  # upper bounds
    )

    res = least_squares(
        residuals, p0, bounds=bounds, args=(omega, Z_exp), max_nfev=10000
    )

    # Checar sucesso e registrar aviso se necessário
    import logging

    logger = logging.getLogger(__name__)
    if not res.success:
        logger.warning("Ajuste CPE+Warburg não convergiu: %s", res.message)

    Rs, Rp, Q, n, sigma = res.x

    return {
        "Rs_fit": Rs,
        "Rp_fit": Rp,
        "Q": Q,
        "n": n,
        "Sigma": sigma,
        "Fit_error": float(np.mean(res.fun**2)),
        "Converged": bool(res.success),
    }
