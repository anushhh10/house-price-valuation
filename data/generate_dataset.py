"""
Synthetic Real Estate Dataset Generator with Localized Societal Metrics (INR Scale).

Generates realistic housing data combining core structural specifications
with domain-specific localized indicators (Infrastructure, Safety, Environment, Growth)
priced in Indian Rupees (INR - ₹).
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_housing_data(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic real estate dataset with realistic structural
    and localized societal metrics priced in Indian Rupees (INR - ₹).

    Parameters:
    -----------
    n_samples : int
        Number of synthetic property records to generate.
    random_seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        Synthetic real estate dataset containing structural, localized metrics,
        and simulated market price in INR.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Structural Features
    bhk = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.40, 0.30, 0.10, 0.05])
    
    # SqFt scales logically with BHK plus variance
    sqft_base = bhk * 450 + rng.normal(loc=150, scale=80, size=n_samples)
    sqft = np.clip(sqft_base.astype(int), 400, 4500)

    # Bathrooms roughly equal to or slightly less/more than BHK
    bathrooms = np.clip(bhk + rng.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.7, 0.1]), 1, 5)

    property_age = np.clip(rng.exponential(scale=8.0, size=n_samples).astype(int), 0, 40)
    total_floors = rng.integers(3, 25, size=n_samples)
    floor_number = np.array([rng.integers(1, max_fl + 1) for max_fl in total_floors])
    parking_spaces = np.clip(rng.poisson(lam=bhk * 0.4, size=n_samples), 0, 3)

    # 2. Localized Societal Metrics
    # Infrastructure
    school_score = np.round(np.clip(rng.normal(loc=6.8, scale=1.8, size=n_samples), 1.0, 10.0), 1)
    hospital_distance_km = np.round(np.clip(rng.exponential(scale=3.5, size=n_samples), 0.3, 15.0), 2)
    transit_score = np.round(np.clip(rng.normal(loc=6.5, scale=2.0, size=n_samples), 1.0, 10.0), 1)

    # Safety
    crime_rate = np.round(np.clip(rng.normal(loc=42.0, scale=18.0, size=n_samples), 5.0, 95.0), 1)
    police_proximity_km = np.round(np.clip(rng.exponential(scale=2.8, size=n_samples), 0.2, 12.0), 2)

    # Environment
    aqi = np.clip(rng.normal(loc=95.0, scale=35.0, size=n_samples).astype(int), 25, 280)
    green_cover_pct = np.round(np.clip(rng.normal(loc=28.0, scale=12.0, size=n_samples), 3.0, 65.0), 1)
    noise_level_db = np.round(np.clip(rng.normal(loc=58.0, scale=10.0, size=n_samples), 35.0, 85.0), 1)

    # Growth & Economy
    growth_index = np.round(np.clip(rng.normal(loc=55.0, scale=18.0, size=n_samples), 5.0, 98.0), 1)
    commercial_hub_distance_km = np.round(np.clip(rng.exponential(scale=6.0, size=n_samples), 0.5, 25.0), 2)

    # 3. Realistic Valuation Formula in Indian Rupees (INR - ₹)
    # Base structural component: ₹6,500/sqft base rate, ₹1.5L/bathroom, ₹2.5L/parking space
    base_sqft_rate = 6800.0
    structural_value = (sqft * base_sqft_rate) + (bathrooms * 150000) + (parking_spaces * 250000)
    age_depreciation = np.maximum(0.58, 1.0 - (property_age * 0.0125))

    # Multi-Agent Factor Multipliers
    infra_factor = 1.0 + ((school_score - 5.0) * 0.035) + ((transit_score - 5.0) * 0.030) - (hospital_distance_km * 0.010)
    safety_factor = 1.0 - ((crime_rate - 35.0) * 0.0035) - (police_proximity_km * 0.008)
    env_factor = 1.0 - ((aqi - 80) * 0.0009) + ((green_cover_pct - 20) * 0.0025) - ((noise_level_db - 55) * 0.0018)
    growth_factor = 1.0 + ((growth_index - 50.0) * 0.0038) - (commercial_hub_distance_km * 0.007)

    # Composite locational multiplier
    locational_multiplier = np.clip(infra_factor * safety_factor * env_factor * growth_factor, 0.55, 1.75)

    # Final Price computation with market stochastic noise (4.5% std dev)
    raw_price = structural_value * age_depreciation * locational_multiplier
    market_noise = rng.normal(loc=1.0, scale=0.045, size=n_samples)
    final_price = np.round(np.clip(raw_price * market_noise, 2500000, 65000000), -3)

    df = pd.DataFrame({
        "sqft": sqft,
        "bhk": bhk,
        "bathrooms": bathrooms,
        "property_age": property_age,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "parking_spaces": parking_spaces,
        "school_score": school_score,
        "hospital_distance_km": hospital_distance_km,
        "transit_score": transit_score,
        "crime_rate": crime_rate,
        "police_proximity_km": police_proximity_km,
        "aqi": aqi,
        "green_cover_pct": green_cover_pct,
        "noise_level_db": noise_level_db,
        "growth_index": growth_index,
        "commercial_hub_distance_km": commercial_hub_distance_km,
        "price": final_price,
    })

    return df


def format_inr_display(val: float) -> str:
    """Format an INR number to Lakhs or Crores."""
    if val >= 10000000:
        return f"INR {val / 10000000:.2f} Cr"
    elif val >= 100000:
        return f"INR {val / 100000:.2f} L"
    return f"INR {val:,.0f}"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic real estate dataset in INR.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/house_prices.csv",
        help="Target output CSV file path.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Generating {args.samples} synthetic housing records (INR Valuation)...")
    df = generate_housing_data(n_samples=args.samples, random_seed=args.seed)

    df.to_csv(args.output, index=False)
    print(f"Dataset successfully saved to: {args.output}")
    print(f"Shape: {df.shape}")
    print("\nSummary Statistics of Generated Price in INR:")
    stats = df["price"].describe()
    for k, v in stats.items():
        print(f" - {k:<6}: {format_inr_display(v)} (INR {v:,.0f})")


if __name__ == "__main__":
    main()
