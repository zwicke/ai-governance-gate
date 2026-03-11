from typing import List
from pydantic import BaseModel

# --- HARDWARE ENVIRONMENTAL REGISTRY ---
# carbon_g_hr: Grams of CO2 per hour of peak operation.
# Factors in typical power draw and data center PUE (Power Usage Effectiveness).
HARDWARE_REGISTRY = {
    "H100 SXM": {"bandwidth_gbs": 3350, "carbon_g_hr": 280}, 
    "A100 SXM": {"bandwidth_gbs": 2039, "carbon_g_hr": 350}, 
    "TPU v6e": {"bandwidth_gbs": 820, "carbon_g_hr": 45},   # High-efficiency specialized AI chip
    "Shared": {"bandwidth_gbs": 150, "carbon_g_hr": 600}    # Generic legacy cloud hardware
}

class InferenceConfig(BaseModel):
    model_name: str
    provider: str
    hardware: str
    quality: int
    billing_type: str 
    input_price: float 
    output_price: float
    parameters_billions: float 
    quantization_bits: int

    def calculate_normalized_cost(self) -> float:
        """Translates pricing into a universal USD per 1M blended tokens."""
        if self.billing_type == "token":
            return (self.input_price + self.output_price) / 2
        # For instance-based, we assume a standard volume of 50 tokens/sec
        tokens_per_hour = 50 * 3600 
        return (self.output_price / tokens_per_hour) * 1_000_000

    def get_roofline_latency(self) -> float:
        """Calculates physics-based Response Time (ms) via Memory Bandwidth."""
        hw = HARDWARE_REGISTRY.get(self.hardware, HARDWARE_REGISTRY["Shared"])
        model_size_gb = (self.parameters_billions * self.quantization_bits) / 8
        return (model_size_gb / hw["bandwidth_gbs"]) * 1000 * 1.2

    def get_carbon_footprint(self) -> float:
        """Surfaces environmental debt (grams of CO2 per 1M tokens)."""
        hw = HARDWARE_REGISTRY.get(self.hardware, HARDWARE_REGISTRY["Shared"])
        # Compute time required to produce 1M tokens
        hours_per_1m = (self.get_roofline_latency() * 1000000) / 3600000
        return hours_per_1m * hw["carbon_g_hr"]

def get_pareto_frontier(configs: List[InferenceConfig]):
    """Identifies 'Market Leaders'—models where no better trade-off exists."""
    frontier = []
    data_list = [
        {"c": c, "cost": c.calculate_normalized_cost(), "lat": c.get_roofline_latency(), "carbon": c.get_carbon_footprint()} 
        for c in configs
    ]
    for a in data_list:
        is_dominated = any(
            (b["cost"] <= a["cost"] and b["lat"] <= a["lat"] and b["c"].quality >= a["c"].quality and b["carbon"] <= a["carbon"]) and
            (b["cost"] < a["cost"] or b["lat"] < a["lat"] or b["c"].quality > a["c"].quality or b["carbon"] < a["carbon"])
            for b in data_list
        )
        if not is_dominated: frontier.append(a["c"])
    return frontier