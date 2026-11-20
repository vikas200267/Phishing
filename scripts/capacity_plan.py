from __future__ import annotations
import argparse, math, json

def effective_rps(measured_rps: float, headroom: float) -> float:
    if measured_rps <= 0:
        raise ValueError('--measured-rps must be > 0')
    if headroom <= 0:
        raise ValueError('--headroom must be > 0')
    # Accept both a 1.30 capacity factor and the older 0.30 slack form.
    # 1.30 means retain 1/1.30 of the measured capacity; 0.30 means
    # retain 70% of the measured capacity.
    if headroom < 1.0:
        return measured_rps * (1.0 - headroom)
    return measured_rps / headroom

def main():
    p=argparse.ArgumentParser(description='Estimate replica/load-generator counts from an actual measured benchmark. This is capacity math, not a capacity claim.')
    p.add_argument('--measured-rps', type=float, required=True)
    p.add_argument('--target-rps', type=float, default=1_000_000)
    p.add_argument('--headroom', type=float, default=1.30, help='Capacity factor (1.30 = 30% headroom) or legacy slack fraction (0.30 = retain 70%).')
    p.add_argument('--generators', type=int, default=8)
    a=p.parse_args()
    if a.target_rps <= 0: raise ValueError('--target-rps must be > 0')
    if a.generators <= 0: raise ValueError('--generators must be > 0')
    effective=effective_rps(a.measured_rps, a.headroom)
    replicas=math.ceil(a.target_rps/effective)
    per_gen=math.ceil(a.target_rps/a.generators)
    print(json.dumps({
        'measured_rps':a.measured_rps,
        'headroom':a.headroom,
        'headroom_interpretation':'factor' if a.headroom>=1 else 'slack_fraction',
        'effective_rps_per_replica':effective,
        'target_rps':a.target_rps,
        'required_api_replicas':replicas,
        'load_generators':a.generators,
        'target_rps_per_generator':per_gen
    },indent=2))
if __name__=='__main__':main()
