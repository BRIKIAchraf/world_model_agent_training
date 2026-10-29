import numpy as np

from envs.sound import machine_sound


normal_sound = machine_sound(
    faulty=False,
    distance=2.0,
)

faulty_sound = machine_sound(
    faulty=True,
    distance=2.0,
    rng=np.random.default_rng(0),
)

print(
    "Forme du son normal :",
    normal_sound.shape,
)

print(
    "Forme du son défectueux :",
    faulty_sound.shape,
)

print(
    "Amplitude normale :",
    round(float(np.max(np.abs(normal_sound))), 4),
)

print(
    "Amplitude défectueuse :",
    round(float(np.max(np.abs(faulty_sound))), 4),
)

print(
    "Différence moyenne :",
    round(
        float(
            np.mean(
                np.abs(
                    normal_sound - faulty_sound
                )
            )
        ),
        4,
    ),
)