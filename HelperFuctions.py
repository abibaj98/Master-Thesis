def r_u_pseudo_outcomes(y, w, probs, mu, eps):
    return (y - mu) / (w - probs + eps)


def dr_pseudo_outcomes(y, w, probs, mu0, mu1, eps):
    mu_w = w * mu1 + (1 - w) * mu0
    return (w - probs) / (probs * (1 - probs) + eps) * (y - mu_w) + mu1 - mu0


def ra_pseudo_outcomes(y, w, mu0, mu1):
    return w * (y - mu0) + (1 - w) * (mu1 - y)


def pw_pseudo_outcomes(y, w, probs, eps):
    return (w / (probs + eps) - (1 - w) / ((1 - probs) + eps)) * y
