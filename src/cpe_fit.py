import numpy as np
from scipy.optimize import least_squares


def cpe_impedance(omega, Q, n):
    return 1 / (Q * (1j * omega) ** n)


def warburg_impedance(omega, sigma):
    return sigma / np.sqrt(1j * omega)


def model_impedance(params, omega):
    Rs, Rp, Q, n, sigma = params
    Zcpe = cpe_impedance(omega, Q, n)
    Zw = warburg_impedance(omega, sigma)
    Zpar = 1 / (1 / Rp + 1 / Zcpe)
    return Rs + Zpar + Zw


def residuals(params, omega, Z_exp):
    Z_model = model_impedance(params, omega)
    return np.concatenate([(Z_model.real - Z_exp.real), (Z_model.imag - Z_exp.imag)])


def fit_cpe_warburg(df):
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
