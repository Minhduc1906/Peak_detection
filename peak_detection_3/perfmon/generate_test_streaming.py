import pandas as pd
import json
import os
import random
from typing import List, Dict, Any

# Set experiment duration to 300 seconds
EXPERIMENT_DURATION = 300

def parse_multi_flow_tcp(value: str) -> List[Dict[str, Any]]:
    """Parse TCP traffic configuration strings."""
    if pd.isna(value) or value == '-':
        return []
    
    flow_configs = []
    if '+' in value:
        flow_specs = value.split('+')
        for spec in flow_specs:
            spec = spec.strip()
            if '@' in spec:
                flows_str, rate_str = spec.split('@')
                flows = int(flows_str.strip())
                rate = float(rate_str.split()[0])
                flow_configs.append({
                    'type': 'tcp_limited',
                    'count': flows,
                    'rate': rate
                })
    else:
        if '@' in value:
            flows_str, rate_str = value.split('@')
            flows = int(flows_str.strip())
            rate = float(rate_str.split()[0])
            flow_configs.append({
                'type': 'tcp_limited',
                'count': flows,
                'rate': rate
            })
    
    return flow_configs

def parse_cbr_traffic(value: str) -> List[Dict[str, Any]]:
    """Parse CBR traffic configurations."""
    if pd.isna(value) or value == '-':
        return []
    
    cbr_configs = []
    if 'CBR UDP' in value:
        rates_str = value.split('Mb/s')[0]
        rates = [float(x.strip()) for x in rates_str.split(',')]
        for rate in rates:
            cbr_configs.append({
                'type': 'cbr',
                'rate': rate
            })
    
    return cbr_configs

def parse_stochastic_config(value: str) -> List[Dict[str, Any]]:
    """Parse stochastic traffic configuration strings."""
    if pd.isna(value) or value == '-':
        return []
    
    stochastic_flows = []
    if '+' in value:
        flow_specs = value.split('+')
        for spec in flow_specs:
            spec = spec.strip()
            stochastic_flows.extend(parse_single_stochastic_config(spec))
    else:
        stochastic_flows.extend(parse_single_stochastic_config(value))
    
    return stochastic_flows

def parse_single_stochastic_config(flow: str) -> List[Dict[str, Any]]:
    """Parse a single stochastic traffic configuration."""
    stochastic_flows = []
    
    if 'VoIP' in flow:
        params = flow[flow.index('(')+1:flow.index(')')].split(';')
        stochastic_flows.append({
            'type': 'VoIP',
            'min_size': int(params[0]),
            'max_size': int(params[1]),
            'rate': float(params[2])
        })
    elif 'Gaming' in flow:
        params = flow[flow.index('(')+1:flow.index(')')].split(';')
        stochastic_flows.append({
            'type': 'Gaming',
            'min_size': int(params[0]),
            'max_size': int(params[1]),
            'rate': float(params[2])
        })
    elif 'Video' in flow:
        params = flow[flow.index('(')+1:flow.index(')')].split(';')
        stochastic_flows.append({
            'type': 'Video',
            'min_size': int(params[0]),
            'max_size': int(params[1]),
            'distribution': params[2],
            'rate': float(params[3])
        })
    
    return stochastic_flows

def create_streaming_flow(client_index: int, rate: float, base_start_time: int, quality_profile: str = "standard") -> List[Dict[str, Any]]:
    """Create a realistic D-ITG TCP command for a streaming flow with exponential distribution."""
    base_port = 21000 + (client_index - 1) * 100
    avg_packet_size = 1000  # Average of 500–1500 bytes
    mean_packet_rate = (rate * 1000000) / (avg_packet_size * 8)  # packets/s
    
    cmd = (
        f"/usr/bin/ITGSend -T TCP -a client{client_index} "
        f"-U {base_port} {base_port+100} -t {EXPERIMENT_DURATION * 1000} "
        f"-E {mean_packet_rate:.2f} -u 500 1500 "
        f"-B E {mean_packet_rate*2:.2f} C 1000 "
        f"-m rttm -D"
    )
    
    return [{
        "testname": f"D-ITG Streaming {quality_profile.upper()} #{client_index}",
        "host2": {
            "name": f"server{client_index}-mgmt",
            "cmd": cmd,
            "persist": False,
            "start_time": base_start_time
        }
    }]

def create_cbr_flow(client_index: int, rate: float, base_start_time: int) -> List[Dict[str, Any]]:
    """Create a realistic D-ITG UDP command for a CBR flow with Pareto distribution."""
    base_port = 30000 + client_index * 100
    packet_size = 1200
    mean_packet_rate = (rate * 1000000) / (packet_size * 8)  # packets/s
    pareto_shape = 1.5
    pareto_scale = mean_packet_rate * (pareto_shape - 1) / pareto_shape  # Adjust scale for rate
    
    cmd = (
        f"/usr/bin/ITGSend -T UDP -a client{client_index} "
        f"-U {base_port} {base_port+100} -t {EXPERIMENT_DURATION * 1000} "
        f"-V {pareto_shape} {pareto_scale:.2f} -c {packet_size} "
        f"-B E {mean_packet_rate*2:.2f} C 2000 "
        f"-m rttm"
    )
    
    return [{
        "testname": f"D-ITG UDP CBR #{client_index}",
        "host2": {
            "name": f"server{client_index}-mgmt",
            "cmd": cmd,
            "persist": False,
            "start_time": base_start_time
        }
    }]

def create_stochastic_flow(client_index: int, flow: Dict[str, Any], base_start_time: int) -> List[Dict[str, Any]]:
    """Create a realistic D-ITG UDP command for a stochastic flow."""
    base_port = 30000 + client_index * 100
    flow_type = flow['type']
    
    if flow_type == 'Gaming':
        cmd = (
            f"/usr/bin/ITGSend -T UDP -a client{client_index} "
            f"-U {base_port} {base_port+100} -t {EXPERIMENT_DURATION * 1000} "
            f"-x csa "
            f"-m rttm"
        )
    elif flow_type == 'VoIP':
        cmd = (
            f"/usr/bin/ITGSend -T UDP -a client{client_index} "
            f"-U {base_port} {base_port+100} -t {EXPERIMENT_DURATION * 1000} "
            f"-x voip g729.2 -VAD "
            f"-m rttm"
        )
    elif flow_type == 'Video':
        mean_packet_size = (flow['min_size'] + flow['max_size']) // 2
        mean_packet_rate = (flow['rate'] * 1000000) / (mean_packet_size * 8)  # packets/s
        std_dev_rate = mean_packet_rate * 0.2  # 20% of mean
        std_dev_size = mean_packet_size * 0.2
        
        cmd = (
            f"/usr/bin/ITGSend -T UDP -a client{client_index} "
            f"-U {base_port} {base_port+100} -t {EXPERIMENT_DURATION * 1000} "
            f"-N {mean_packet_rate:.2f} {std_dev_rate:.2f} "
            f"-n {mean_packet_size} {std_dev_size:.2f} "
            f"-m rttm"
        )
    
    return [{
        "testname": f"D-ITG {flow_type} #{client_index}",
        "host2": {
            "name": f"server{client_index}-mgmt",
            "cmd": cmd,
            "persist": False,
            "start_time": base_start_time
        }
    }]

def generate_test_files(csv_file: str, output_dir: str = "test_configs", randomize: bool = True, 
                      total_experiment_duration: int = 300):
    """Generate test files for D-ITG with realistic traffic distributions."""
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    if not randomize:
        random.seed(42)
    
    for idx, row in df.iterrows():
        scenario = row['Scenario']
        base_scenario_name = scenario.replace(" ", "_").replace(",", "").lower()
        
        limited_tcp_configs = parse_multi_flow_tcp(row['TCP flows (rate limited)'])
        cbr_configs = parse_cbr_traffic(row['CBR traffic'])
        stochastic_flows = parse_stochastic_config(row['Stochastic traffic'])
        
        tests = []
        client_index = 1
        base_start_time = 0
        
        streaming_profiles = []
        for cfg in limited_tcp_configs:
            if "4K" in scenario or "Ultra" in scenario or cfg['rate'] >= 20:
                streaming_profiles.append("4k")
            elif cfg['rate'] >= 5:
                streaming_profiles.append("standard")
            else:
                streaming_profiles.append("low")
        
        for i, tcp_config in enumerate(limited_tcp_configs):
            profile = streaming_profiles[i % len(streaming_profiles)] if streaming_profiles else "standard"
            for _ in range(tcp_config['count']):
                tests.extend(create_streaming_flow(
                    client_index, tcp_config['rate'], base_start_time, profile
                ))
                client_index += 1
        
        for cbr_config in cbr_configs:
            tests.extend(create_cbr_flow(
                client_index, cbr_config['rate'], base_start_time
            ))
            client_index += 1
        
        for flow in stochastic_flows:
            tests.extend(create_stochastic_flow(
                client_index, flow, base_start_time
            ))
            client_index += 1
        
        total_bw = sum(config['rate'] * config['count'] for config in limited_tcp_configs)
        total_bw += sum(config['rate'] for config in cbr_configs)
        total_bw += sum(flow['rate'] for flow in stochastic_flows)
        
        if total_bw > 120:
            bw_category = "ultra_high_load"
        elif total_bw > 80:
            bw_category = "high_load"
        elif total_bw > 40:
            bw_category = "medium_load"
        else:
            bw_category = "normal_load"
        
        if tests:
            rand_suffix = "_dynamic" if randomize else ""
            filename = os.path.join(
                output_dir, 
                f"test_{base_scenario_name}_{bw_category}{rand_suffix}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump(tests, f, indent=2)
            
            print(f"Generated {filename} with {len(tests)} traffic configurations")
            print(f"  - {len(limited_tcp_configs)} streaming profiles")
            print(f"  - {len(cbr_configs)} CBR traffic flows")
            print(f"  - {len(stochastic_flows)} stochastic traffic patterns")
            print(f"  - Total estimated bandwidth: {total_bw:.1f} Mbps ({bw_category})")
    
    print(f"\nAll test configurations have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    csv_filename = "network_test_scenarios.csv"
    output_dir = "test_configs"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    if not os.path.exists(csv_filename):
        print(f"Error: CSV file '{csv_filename}' not found.")
        print("Please ensure this file exists with your streaming scenarios.")
        print("Exiting script.")
        exit(1)
    
    experiment_duration = EXPERIMENT_DURATION
    
    print(f"\nUsing CSV file: {csv_filename}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment duration: {experiment_duration} seconds")
    print("\nGenerating test configurations...")
    
    generate_test_files(csv_filename,
                      output_dir=output_dir,
                      randomize=True,
                      total_experiment_duration=experiment_duration)