import pandas as pd
import json
import os
import random
from typing import List, Dict, Any, Union, Tuple

def parse_traffic_config(value: str) -> Tuple[List[Union[int, float]], str, float]:
    """Parse traffic configuration strings into flow counts and rates."""
    if pd.isna(value) or value == '-':
        return [], '', 0.0
    
    if '@' in value:  # TCP flows with rate limit (e.g., "2@15 Mb/s/flow")
        flows_str, rate_str = value.split('@')
        # Extract the number of flows
        flows = [int(flows_str.strip().split()[0])]
        # Extract the rate value
        rate = float(rate_str.split()[0])
        return flows, 'tcp', rate
    elif 'CBR UDP' in value:  # CBR traffic (e.g., "20 Mb/s CBR UDP")
        # Extract the rate value
        rate_str = value.split('Mb/s')[0].strip()
        rates = [float(rate_str)]
        return rates, 'cbr', 0.0
    elif 'TCP flow' in value:  # Unlimited TCP flows (e.g., "0 TCP flow" or "2 TCP flows")
        # Extract the number of flows
        flows_count = int(value.split('TCP')[0].strip())
        flows = [flows_count] if flows_count > 0 else []
        return flows, 'tcp_unlimited', float('inf')
    
    return [], '', 0.0

def parse_stochastic_config(value: str) -> List[Dict[str, Any]]:
    """Parse stochastic traffic configuration strings."""
    if pd.isna(value) or value == '-':
        return []
    
    stochastic_flows = []
    flow_configs = value.split('+')
    
    for flow in flow_configs:
        if 'VoIP' in flow:
            params = flow[flow.index('(')+1:flow.index(')')].split(';')
            stochastic_flows.append({
                'type': 'VoIP',
                'min_size': int(params[0]),
                'max_size': int(params[1]),
                'variance': float(params[2])
            })
        elif 'Gaming' in flow:
            params = flow[flow.index('(')+1:flow.index(')')].split(';')
            stochastic_flows.append({
                'type': 'Gaming',
                'min_size': int(params[0]),
                'max_size': int(params[1]),
                'variance': float(params[2])
            })
        elif 'Video' in flow:
            params = flow[flow.index('(')+1:flow.index(')')].split(';')
            if len(params) == 4:
                stochastic_flows.append({
                    'type': 'Video',
                    'min_size': int(params[0]),
                    'max_size': int(params[1]),
                    'distribution': params[2],
                    'variance': float(params[3])
                })
            else:
                # Handle format in the provided CSV: Video(800;1200;g;0.1)
                stochastic_flows.append({
                    'type': 'Video',
                    'min_size': int(params[0]),
                    'max_size': int(params[1]),
                    'distribution': params[2],
                    'variance': float(params[3])
                })
    
    return stochastic_flows

def generate_streaming_segments(total_duration: int, stream_type: str = 'netflix_hd') -> List[Dict[str, Any]]:
    """
    Generate streaming-specific segments that mimic the buffering behavior of streaming services.
    
    Args:
        total_duration: Total streaming duration in seconds
        stream_type: Type of streaming service
        
    Returns:
        List of dicts with 'type', 'rate_multiplier', 'duration', and 'start_time' keys
    """
    segments = []
    remaining_duration = total_duration
    current_time = 0
    
    # Netflix streaming patterns
    if stream_type == 'netflix_sd':
        initial_buffer_time = random.randint(3, 7)
        initial_buffer_multiplier = 1.8
        steady_state_multiplier = 0.65
        rebuffer_interval_min = 40
        rebuffer_interval_max = 120
        rebuffer_duration = random.randint(1, 3)
        rebuffer_multiplier = 1.4
    
    elif stream_type == 'netflix_hd':
        # Netflix HD uses larger initial buffer and maintains steadier playback
        initial_buffer_time = random.randint(5, 10)
        initial_buffer_multiplier = 2.0
        steady_state_multiplier = 0.7
        rebuffer_interval_min = 45
        rebuffer_interval_max = 120
        rebuffer_duration = random.randint(2, 4)
        rebuffer_multiplier = 1.5
    
    elif stream_type == 'netflix_4k':
        # Netflix 4K needs even larger buffers
        initial_buffer_time = random.randint(10, 15)
        initial_buffer_multiplier = 2.2
        steady_state_multiplier = 0.75
        rebuffer_interval_min = 40
        rebuffer_interval_max = 100
        rebuffer_duration = random.randint(3, 6)
        rebuffer_multiplier = 1.8
    
    elif stream_type == 'netflix_hdr':
        # HDR content requires higher bitrates
        initial_buffer_time = random.randint(8, 12)
        initial_buffer_multiplier = 2.1
        steady_state_multiplier = 0.78
        rebuffer_interval_min = 35
        rebuffer_interval_max = 90
        rebuffer_duration = random.randint(3, 5)
        rebuffer_multiplier = 1.7
    
    # YouTube streaming patterns
    elif stream_type == 'youtube_sd':
        # YouTube SD tends to have smaller buffers
        initial_buffer_time = random.randint(3, 6)
        initial_buffer_multiplier = 1.7
        steady_state_multiplier = 0.6
        rebuffer_interval_min = 30
        rebuffer_interval_max = 100
        rebuffer_duration = random.randint(1, 3)
        rebuffer_multiplier = 1.4
    
    elif stream_type == 'youtube_hd':
        # YouTube HD has more aggressive buffering strategy
        initial_buffer_time = random.randint(7, 12)
        initial_buffer_multiplier = 2.2
        steady_state_multiplier = 0.6
        rebuffer_interval_min = 25
        rebuffer_interval_max = 90
        rebuffer_duration = random.randint(2, 4)
        rebuffer_multiplier = 1.6
    
    elif stream_type == 'youtube_4k':
        # YouTube 4K tends to use larger initial buffers
        initial_buffer_time = random.randint(12, 18)
        initial_buffer_multiplier = 2.8
        steady_state_multiplier = 0.65
        rebuffer_interval_min = 20
        rebuffer_interval_max = 70
        rebuffer_duration = random.randint(3, 7)
        rebuffer_multiplier = 2.0
    
    elif stream_type == 'youtube_live':
        # YouTube Live has more variable buffering due to live nature
        initial_buffer_time = random.randint(5, 8)
        initial_buffer_multiplier = 1.9
        steady_state_multiplier = 0.75
        rebuffer_interval_min = 15
        rebuffer_interval_max = 50
        rebuffer_duration = random.randint(2, 4)
        rebuffer_multiplier = 1.7
    
    else:  # Default to Netflix HD settings
        initial_buffer_time = random.randint(5, 10)
        initial_buffer_multiplier = 2.0
        steady_state_multiplier = 0.7
        rebuffer_interval_min = 45
        rebuffer_interval_max = 120
        rebuffer_duration = random.randint(2, 4)
        rebuffer_multiplier = 1.5
    
    # Ensure the initial buffer time is not longer than the total duration
    initial_buffer_time = min(initial_buffer_time, remaining_duration)
    
    # Add initial buffering segment
    segments.append({
        'type': 'initial_buffer',
        'rate_multiplier': initial_buffer_multiplier,
        'duration': initial_buffer_time,
        'start_time': current_time
    })
    
    remaining_duration -= initial_buffer_time
    current_time += initial_buffer_time
    
    # Add steady state streaming with periodic rebuffering
    while remaining_duration > 0:
        # Determine next rebuffer time
        next_rebuffer_interval = random.randint(rebuffer_interval_min, rebuffer_interval_max)
        
        # Ensure we don't exceed the remaining duration
        steady_duration = min(next_rebuffer_interval, remaining_duration)
        
        # Add steady state segment
        segments.append({
            'type': 'steady_state',
            'rate_multiplier': steady_state_multiplier,
            'duration': steady_duration,
            'start_time': current_time
        })
        
        remaining_duration -= steady_duration
        current_time += steady_duration
        
        # Add rebuffering segment if we still have time
        if remaining_duration > 0:
            rebuffer_time = min(rebuffer_duration, remaining_duration)
            
            segments.append({
                'type': 'rebuffer',
                'rate_multiplier': rebuffer_multiplier,
                'duration': rebuffer_time,
                'start_time': current_time
            })
            
            remaining_duration -= rebuffer_time
            current_time += rebuffer_time
    
    return segments

def create_streaming_tcp_config(index: int, base_rate: float, start_time: int, 
                               segments: List[Dict[str, Any]], stream_type: str) -> List[Dict[str, Any]]:
    """Create multiple TCP test configurations that mimic streaming behavior."""
    configs = []
    
    for segment in segments:
        # Calculate the effective rate for this segment
        effective_rate = base_rate * segment['rate_multiplier']
        
        # Create appropriate naming based on segment type
        segment_type_name = segment['type'].replace('_', ' ').title()
        
        configs.append({
            "testname": f"{stream_type.replace('_', ' ').title()} {segment_type_name} #{index}",
            "host1": {
                "name": f"client{index}-mgmt",
                "cmd": f"iperf3 -c server{index} -C reno -R -b {effective_rate}M -t {segment['duration']}",
                "persist": False,
                "start_time": start_time + segment['start_time']
            }
        })
    
    return configs

def create_cbr_streaming_config(index: int, rate: float, start_time: int, duration: int, 
                               stream_type: str) -> Dict[str, Any]:
    """
    Create a CBR UDP test configuration that simulates a streaming service with constant bit rate.
    
    For streaming services that use CBR (like some live streams), this creates a more consistent
    traffic pattern with small variations to simulate network adjustments.
    """
    # Small variation around the target rate (±5%)
    jitter_percent = 5
    
    return {
        "testname": f"{stream_type.replace('_', ' ').title()} CBR Streaming #{index}",
        "host1": {
            "name": f"client{index}-mgmt",
            "cmd": f"iperf3 -c server{index} -R -u -b {rate}M -t {duration} --pacing-timer 1000 --fq-rate {rate * (100 - jitter_percent) / 100}M",
            "persist": False,
            "start_time": start_time
        }
    }

def create_stochastic_config(index: int, flow: Dict[str, Any], start_time: int, duration_seconds: int = 300) -> Dict[str, Any]:
    """Create a stochastic traffic configuration with an adjustable duration (default 5 minutes)."""
    # Convert seconds to milliseconds for D-ITG
    duration_ms = duration_seconds * 1000
    
    base_config = {
        "testname": f"{flow['type']} Traffic Simulation #{index}",
        "host1": {
            "name": f"server{index}-mgmt",
            "cmd": "ITGRecv",
            "persist": True,
            "start_time": 0
        },
        "host2": {
            "name": f"client{index}-mgmt",
            "persist": False,
            "start_time": start_time
        }
    }
    
    if flow['type'] == 'VoIP':
        base_config["host2"]["cmd"] = (
            f"ITGSend -T UDP -a server{index} "
            f"-C {flow['max_size']} -c {flow['min_size']} "
            f"-t {duration_ms} -x voip_recv_log_{index} -l voip_send_log_{index} "
            f"-e 100 -E 100 -k 20 -K 20 -v {flow['variance']}"
        )
    elif flow['type'] == 'Gaming':
        base_config["host2"]["cmd"] = (
            f"ITGSend -T UDP -a server{index} "
            f"-C {flow['max_size']} -c {flow['min_size']} "
            f"-t {duration_ms} -x game_recv_log_{index} -l game_send_log_{index} "
            f"-e 50 -E 50 -k 10 -K 10 -v {flow['variance']}"
        )
    elif flow['type'] == 'Video':
        base_config["host2"]["cmd"] = (
            f"ITGSend -T UDP -a server{index} "
            f"-C {flow['max_size']} -c {flow['min_size']} "
            f"-t {duration_ms} -x video_recv_log_{index} -l video_send_log_{index} "
            f"-e 1000 -E 1000 -k 30 -K 30 -d {flow['distribution']} -v {flow['variance']}"
        )
    
    return base_config

def calculate_total_bandwidth(limited_flows, limited_rate, unlimited_flows, cbr_rates):
    """Calculate total bandwidth required for a scenario."""
    total = 0
    
    # Rate-limited TCP flows
    if limited_flows and limited_rate != float('inf'):
        total += limited_flows * limited_rate
    
    # Unlimited TCP flows (assume 80 Mbps each as an estimate)
    if unlimited_flows:
        total += unlimited_flows * 80
    
    # CBR traffic
    if cbr_rates:
        total += sum(cbr_rates)
    
    return total

def determine_stream_type(scenario_name, index):
    """Determine the streaming service type based on scenario name and index."""
    scenario_lower = scenario_name.lower()
    
    # For Netflix streaming scenarios
    if 'netflix' in scenario_lower:
        if '4k' in scenario_lower:
            return 'netflix_4k'
        elif 'hdr' in scenario_lower:
            return 'netflix_hdr'
        elif 'hd' in scenario_lower:
            return 'netflix_hd'
        elif 'sd' in scenario_lower:
            return 'netflix_sd'
        else:
            return 'netflix_hd'  # Default to HD if not specified
    
    # For YouTube streaming scenarios
    elif 'youtube' in scenario_lower:
        if '4k' in scenario_lower:
            return 'youtube_4k'
        elif 'live' in scenario_lower:
            return 'youtube_live'
        elif 'hd' in scenario_lower:
            return 'youtube_hd'
        elif 'sd' in scenario_lower:
            return 'youtube_sd'
        else:
            return 'youtube_hd'  # Default to HD if not specified
    
    # For mixed scenarios, use the index to alternate between services
    elif 'dual netflix' in scenario_lower:
        return 'netflix_hd'
    elif 'dual youtube' in scenario_lower:
        return 'youtube_hd'
    elif 'multi-quality netflix' in scenario_lower:
        return ['netflix_hd', 'netflix_4k'][index % 2]
    elif 'multi-quality youtube' in scenario_lower:
        return ['youtube_hd', 'youtube_4k'][index % 2]
    elif 'netflix + youtube' in scenario_lower:
        if index == 0:
            return 'netflix_hd'
        else:
            return 'youtube_hd'
    elif 'netflix 4k + youtube' in scenario_lower:
        if index == 0:
            return 'netflix_4k'
        else:
            return 'youtube_hd'
    elif 'netflix hd + youtube 4k' in scenario_lower:
        if index == 0:
            return 'netflix_hd'
        else:
            return 'youtube_4k'
    
    # Default case
    return 'netflix_hd'

def generate_streaming_test_files(csv_file: str, output_dir: str = "test_configs", 
                             total_experiment_duration: int = 600):
    """
    Generate test.json files for streaming scenarios with realistic buffering patterns.
    
    Args:
        csv_file: Path to the CSV file with experiment specifications
        output_dir: Directory to save the generated JSON files
        total_experiment_duration: Target duration for the entire experiment in seconds (default: 600s = 10 minutes)
    """
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility if needed
    random.seed(42)
    
    for idx, row in df.iterrows():
        scenario = row['Scenario']
        base_scenario_name = scenario.replace(" ", "_").replace(",", "").lower()
        
        # Parse all traffic types
        limited_flows_list, limited_type, limited_rate = parse_traffic_config(row['TCP flows (rate limited)'])
        unlimited_flows_list, unlimited_type, _ = parse_traffic_config(row['TCP flows (unlimited)'])
        cbr_rates, cbr_type, _ = parse_traffic_config(row['CBR traffic'])
        stochastic_flows = parse_stochastic_config(row['Stochastic traffic'])
        
        # Generate configurations for all possible combinations
        for limited_flows in (limited_flows_list or [0]):
            for unlimited_flows in (unlimited_flows_list or [0]):
                # Plan the test timeline
                client_index = 1
                start_time = 0
                start_times = []
                
                # Add 10 second spacing between different stream starts
                if limited_flows > 0:
                    for i in range(limited_flows):
                        random_offset = random.randint(0, 3)  # Small random offset
                        start_times.append(start_time + random_offset)
                        client_index += 1
                        start_time += 10  # Base spacing of 10 seconds
                
                if unlimited_flows > 0:
                    for i in range(unlimited_flows):
                        random_offset = random.randint(0, 3)  # Small random offset
                        start_times.append(start_time + random_offset)
                        client_index += 1
                        start_time += 10  # Base spacing of 10 seconds
                
                # Add CBR traffic start times
                cbr_start_times = []
                if cbr_rates:
                    for i in range(len(cbr_rates)):
                        random_offset = random.randint(0, 3)
                        cbr_start_times.append(start_time + random_offset)
                        client_index += 1
                        start_time += 10
                
                # Add stochastic traffic
                for flow in stochastic_flows:
                    start_times.append(start_time)
                    client_index += 1
                    start_time += 10
                
                # Calculate the total test duration based on target experiment duration
                all_start_times = start_times + cbr_start_times
                total_test_duration = 0
                if all_start_times:
                    min_required_duration = max(all_start_times) + 60  # Minimum required duration
                    total_test_duration = max(min_required_duration, total_experiment_duration)
                
                # Now create the actual tests with streaming-specific segments
                tests = []
                client_index = 1
                
                # Calculate total bandwidth
                total_bw = calculate_total_bandwidth(limited_flows, limited_rate, unlimited_flows, cbr_rates or [])
                bw_category = "high_load" if total_bw > 90 else "normal_load"
                
                # Add rate-limited TCP flows (streaming services)
                if limited_flows > 0:
                    for i in range(limited_flows):
                        flow_start = start_times[i]
                        flow_duration = total_test_duration - flow_start
                        
                        # Determine stream type based on scenario name and index
                        stream_type = determine_stream_type(scenario, i)
                        
                        # Generate streaming-specific segments
                        segments = generate_streaming_segments(flow_duration, stream_type)
                        test_configs = create_streaming_tcp_config(
                            client_index, limited_rate, flow_start, segments, stream_type
                        )
                        tests.extend(test_configs)
                        
                        client_index += 1
                
                # Add unlimited TCP flows (other streaming services)
                if unlimited_flows > 0:
                    for i in range(unlimited_flows):
                        flow_start = start_times[limited_flows + i]
                        flow_duration = total_test_duration - flow_start
                        
                        # For unlimited streams, choose appropriate type based on scenario
                        stream_type = determine_stream_type(scenario, limited_flows + i)
                        
                        # Generate streaming-specific segments - using base rate of 80 for unlimited 
                        segments = generate_streaming_segments(flow_duration, stream_type)
                        test_configs = create_streaming_tcp_config(
                            client_index, 80.0, flow_start, segments, stream_type
                        )
                        tests.extend(test_configs)
                        
                        client_index += 1
                
                # Add CBR traffic for streaming services that use constant bit rate
                if cbr_rates:
                    for i, rate in enumerate(cbr_rates):
                        flow_start = cbr_start_times[i]
                        flow_duration = total_test_duration - flow_start
                        
                        # Determine stream type for this CBR flow
                        stream_type = determine_stream_type(scenario, i)
                        
                        # For CBR traffic, we have two options:
                        # 1. Use pure CBR UDP traffic (more realistic for certain streaming types)
                        # 2. Create a modified pattern that still has small variations (more realistic for adaptive CBR)
                        
                        # Check if we want a pure CBR stream or one with some variations
                        if "cbr" in scenario.lower():
                            # Pure CBR streaming
                            tests.append(create_cbr_streaming_config(
                                client_index, rate, flow_start, flow_duration, stream_type
                            ))
                        else:
                            # CBR with small variations to mimic adaptive streaming
                            # Break into segments with small rate changes (±10% around the target rate)
                            segment_duration = 30  # 30 second segments
                            segments = []
                            
                            for seg_start in range(0, flow_duration, segment_duration):
                                seg_duration = min(segment_duration, flow_duration - seg_start)
                                # Vary the rate slightly around the target rate
                                rate_variation = random.uniform(0.9, 1.1)
                                
                                segments.append({
                                    'type': 'cbr_segment',
                                    'rate_multiplier': rate_variation,
                                    'duration': seg_duration,
                                    'start_time': seg_start
                                })
                            
                            test_configs = create_streaming_tcp_config(
                                client_index, rate, flow_start, segments, f"{stream_type}_cbr"
                            )
                            tests.extend(test_configs)
                        
                        client_index += 1
                
                # Add stochastic traffic (background traffic or video effects)
                for j, flow in enumerate(stochastic_flows):
                    if start_times:
                        flow_start = start_times[limited_flows + unlimited_flows + j]
                        tests.append(create_stochastic_config(client_index, flow, flow_start, duration_seconds=300))
                        client_index += 1
                
                # Only save if there are any tests to run
                if tests:
                    # Format filename to indicate streaming simulation
                    cbr_suffix = "_cbr" if cbr_rates else ""
                    filename = os.path.join(
                        output_dir, 
                        f"streaming_{base_scenario_name}_{limited_flows}limited_{unlimited_flows}unlimited{cbr_suffix}_{bw_category}.json"
                    )
                    with open(filename, 'w') as f:
                        json.dump(tests, f, indent=2)
                    
                    print(f"Generated {filename}")

if __name__ == "__main__":
    # Set experiment duration to 10 minutes (600 seconds)
    experiment_duration = 600  # 10 minutes in seconds
    
    generate_streaming_test_files("Experiments_streaming.csv", 
                             total_experiment_duration=experiment_duration)