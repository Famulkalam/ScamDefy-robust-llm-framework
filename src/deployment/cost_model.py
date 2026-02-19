def estimate_cost(throughput_rps: float):
    """
    Estimates cloud infrastructure cost for the fraud detection service.
    Detailed assumption: 
    - Deployment on AWS EC2 t3.xlarge (4 vCPUs, 16GB RAM)
    - On-demand price approx $0.1664/hour (us-east-1)
    """
    HOURLY_COST_AWS_T3_XLARGE = 0.1664
    
    # Capacity planning (safe utilization @ 70%)
    safe_throughput = throughput_rps * 0.7
    
    # Messages per hour per instance
    messages_per_hour = safe_throughput * 3600
    
    # Millions of messages per month
    cost_per_million = (HOURLY_COST_AWS_T3_XLARGE / messages_per_hour) * 1_000_000
    
    print("-" * 30)
    print("Financial Cost Model")
    print("-" * 30)
    print(f"Instance Type: AWS t3.xlarge")
    print(f"Est. Throughput: {safe_throughput:.1f} msgs/sec (safe)")
    print(f"Cost per 1M messages: ${cost_per_million:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    # Example usage - would normally import the benchmark result
    estimate_cost(50.0) # Dummy 50 RPS
