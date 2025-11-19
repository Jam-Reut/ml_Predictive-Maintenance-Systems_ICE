import numpy as np
import pandas as pd

def generate_dataset(path='predictive_maintenance_db.csv', n_trains=50, n_days=365):
    np.random.seed(42)
    components = ["AC", "Door", "Brake"]
    rows = []

    for train_id in range(1, n_trains + 1):
        for comp in components:
            days_since_maintenance = 0

            for day in range(1, n_days + 1):
                weekday = day % 7
                ambient_temp = np.random.normal(loc=15, scale=10)

                if comp == "Door":
                    cycles = np.random.poisson(lam=300)
                    vibration_level = np.random.normal(loc=0.5, scale=0.2)
                    operating_hours = np.random.uniform(8, 18)
                    temperature = np.random.normal(loc=25, scale=3)
                elif comp == "Brake":
                    cycles = np.random.poisson(lam=500)
                    vibration_level = np.random.normal(loc=0.7, scale=0.25)
                    operating_hours = np.random.uniform(6, 16)
                    temperature = np.random.normal(loc=80, scale=10)
                else:
                    cycles = np.random.poisson(lam=100)
                    vibration_level = np.random.normal(loc=0.4, scale=0.15)
                    operating_hours = np.random.uniform(10, 20)
                    temperature = np.random.normal(loc=40, scale=5)

                error_code_count = np.random.poisson(lam=0.3)

                base_risk = 0.001
                risk = base_risk + 0.0005 * days_since_maintenance
                risk += 0.0002 * max(0, temperature - 60)
                risk += 0.0003 * error_code_count
                risk += 0.0004 * (vibration_level - 0.5)
                risk = max(0.0001, min(0.2, risk))

                failure_within_30d = np.random.binomial(1, risk)

                if failure_within_30d == 1:
                    days_since_maintenance = 0
                else:
                    days_since_maintenance += 1

                rows.append({
                    "train_id": train_id,
                    "component_type": comp,
                    "day_index": day,
                    "operating_hours": operating_hours,
                    "cycles": cycles,
                    "temperature": temperature,
                    "vibration_level": vibration_level,
                    "error_code_count": error_code_count,
                    "ambient_temp": ambient_temp,
                    "weekday": weekday,
                    "days_since_last_maintenance": days_since_maintenance,
                    "failure_within_30d": failure_within_30d
                })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Datensatz gespeichert unter: {path}")

if __name__ == '__main__':
    generate_dataset()
