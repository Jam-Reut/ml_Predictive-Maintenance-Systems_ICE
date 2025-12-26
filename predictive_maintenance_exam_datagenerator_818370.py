import numpy as np
import pandas as pd


def generate_predictive_maintenance_exam_v2(
    n_samples: int = 6000,
    random_state: int = 818370
) -> pd.DataFrame:
    """
    Erzeugt einen synthetischen Predictive-Maintenance-Datensatz
    mit physikalisch sinnvollen Zusammenhängen und ca. 10% Failures.
    """

    np.random.seed(random_state)

    # --- numerische Basis-Features ---
    ambient_temp = np.random.normal(loc=25, scale=7, size=n_samples).clip(5, 45)
    humidity = np.random.normal(loc=60, scale=15, size=n_samples).clip(20, 100)
    speed_kmh = np.random.normal(loc=230, scale=40, size=n_samples).clip(80, 320)

    distance_km_since_depot = np.random.exponential(scale=500, size=n_samples).clip(0, 3000)
    days_since_maintenance = np.random.exponential(scale=60, size=n_samples).clip(0, 365)

    axle_load_tons = np.random.normal(loc=17, scale=2, size=n_samples).clip(12, 24)
    brake_usage_pct = np.random.beta(a=2, b=5, size=n_samples) * 100

    vibration_rms = np.random.normal(loc=0.4, scale=0.2, size=n_samples).clip(0.05, 1.5)

    # Lager-Temperatur hängt von Umgebung, Vibration und Bremsen ab
    bearing_temp = (
        ambient_temp
        + np.random.normal(loc=30, scale=8, size=n_samples)
        + vibration_rms * 15
        + brake_usage_pct / 50
    )
    bearing_temp = bearing_temp.clip(30, 130)

    error_count_24h = np.random.poisson(lam=1.2, size=n_samples).clip(0, 10)
    curve_ratio = np.random.beta(a=2, b=8, size=n_samples)

    # abgeleitete Features
    temperature_diff = bearing_temp - ambient_temp
    stress_index = (
        (bearing_temp - 70) / 40
        + (vibration_rms - 0.5) / 0.5
        + (days_since_maintenance / 180)
        + (axle_load_tons - 16) / 4
        + (error_count_24h / 5)
    )
    # auf 0–1 skalieren
    stress_index = (stress_index - stress_index.min()) / (stress_index.max() - stress_index.min())

    # --- kategorische Features ---
    train_line_choices = ["ICE1", "ICE3", "ICE4"]
    shift_choices = ["day", "night"]
    environment_mode_choices = ["normal", "hot", "cold", "wet"]

    train_line = np.random.choice(train_line_choices, size=n_samples, p=[0.3, 0.4, 0.3])
    shift = np.random.choice(shift_choices, size=n_samples, p=[0.6, 0.4])
    environment_mode = np.random.choice(
        environment_mode_choices, size=n_samples, p=[0.5, 0.2, 0.15, 0.15]
    )

    # --- Risikofunktion / Label-Logik ---
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    risk = (
        0.30 * sigmoid((bearing_temp - 80) / 8)
        + 0.25 * sigmoid((vibration_rms - 0.7) / 0.2)
        + 0.20 * sigmoid((days_since_maintenance - 90) / 30)
        + 0.15 * sigmoid((error_count_24h - 2) / 1.5)
        + 0.10 * sigmoid((axle_load_tons - 18) / 2)
    )

    # Umgebungsbedingungen erhöhen Risiko
    risk += np.where(environment_mode == "hot", 0.05, 0.0)
    risk += np.where(environment_mode == "cold", 0.02, 0.0)
    risk += np.where(environment_mode == "wet", 0.03, 0.0)

    # leichtes Rauschen
    risk += np.random.normal(loc=0.0, scale=0.02, size=n_samples)

    # Threshold so wählen, dass ca. 10% Failures entstehen
    threshold = np.quantile(risk, 0.9)
    failure_within_30d = (risk > threshold).astype(int)

    df = pd.DataFrame(
        {
            "ambient_temp": ambient_temp,
            "humidity": humidity,
            "speed_kmh": speed_kmh,
            "distance_km_since_depot": distance_km_since_depot,
            "days_since_maintenance": days_since_maintenance,
            "axle_load_tons": axle_load_tons,
            "brake_usage_pct": brake_usage_pct,
            "vibration_rms": vibration_rms,
            "bearing_temp": bearing_temp,
            "error_count_24h": error_count_24h,
            "curve_ratio": curve_ratio,
            "temperature_diff": temperature_diff,
            "stress_index": stress_index,
            "train_line": train_line,
            "shift": shift,
            "environment_mode": environment_mode,
            "failure_within_30d": failure_within_30d,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_predictive_maintenance_exam_v2()
    print("Failure-Rate:", df["failure_within_30d"].mean())
    df.to_csv("predictive_maintenance_exam_v2.csv", index=False)
    print("Datei predictive_maintenance_exam_v2.csv gespeichert.")
