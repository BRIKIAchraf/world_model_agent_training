import numpy as np


def machine_sound(
    faulty,
    sample_rate=8000,
    duration=0.5,
    distance=1.0,
    rng=None,
):
    """
    Génère le son d'une machine normale ou défectueuse.

    faulty=False :
        son grave et régulier.

    faulty=True :
        son grave + fréquence aiguë + bruit.
    """

    if rng is None:
        rng = np.random.default_rng()

    number_of_samples = int(
        sample_rate * duration
    )

    time = np.linspace(
        0,
        duration,
        number_of_samples,
        endpoint=False,
    )

    # Son normal : vibration grave
    signal = np.sin(
        2 * np.pi * 100 * time
    )

    if faulty:
        # Sifflement aigu d'une machine défectueuse
        signal += 0.6 * np.sin(
            2 * np.pi * 450 * time
        )

        # Bruit supplémentaire
        signal += 0.2 * rng.normal(
            0,
            1,
            size=number_of_samples,
        )

    # Le son diminue avec la distance
    attenuation = 1.0 / (1.0 + distance)

    signal = signal * attenuation

    # Éviter les valeurs trop grandes
    signal = np.clip(
        signal,
        -1.0,
        1.0,
    )

    return signal.astype(np.float32)