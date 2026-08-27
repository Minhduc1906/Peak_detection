import json
import random
import os
import argparse
import csv
import math

# Define quality profiles (FHD and higher only)
QUALITY_PROFILES = {
    "FHD": {  # Full HD (1080p)
        "idle_traffic": 1.2,        # Mbps
        "initial_burst_min": 14000,  # Kbps
        "initial_burst_max": 16000,  # Kbps
        "high_bitrate": 7000,        # Kbps
        "medium_bitrate": 5000,      # Kbps
        "packet_size": 512,          # bytes
        "display_name": "Full HD (1080p)"
    },
    "UHD": {  # Ultra HD (4K)
        "idle_traffic": 2.5,        # Mbps
        "initial_burst_min": 21000,  # Kbps
        "initial_burst_max": 25000,  # Kbps
        "high_bitrate": 15000,       # Kbps
        "medium_bitrate": 10000,     # Kbps
        "packet_size": 1024,         # bytes
        "display_name": "UHD (4K)"
    },
    "HDR": {  # High Dynamic Range (4K HDR)
        "idle_traffic": 3.0,        # Mbps
        "initial_burst_min": 25000,  # Kbps
        "initial_burst_max": 30000,  # Kbps
        "high_bitrate": 18000,       # Kbps
        "medium_bitrate": 12000,     # Kbps
        "packet_size": 1024,         # bytes
        "display_name": "HDR (4K HDR)"
    }
}

def generate_netflix_streams(num_streams, quality="FHD", num_clients=20, experiment_duration=300, 
                             mixed_quality=False, quality_distribution=None):
    """Generate Netflix traffic with specified quality"""
    commands = []
    
    # Validate quality selection
    if not mixed_quality and quality not in QUALITY_PROFILES:
        raise ValueError(f"Invalid quality '{quality}'. Choose from: {', '.join(QUALITY_PROFILES.keys())}")
    
    # Setup quality distribution if using mixed quality
    if mixed_quality:
        if not quality_distribution:
            quality_distribution = {"FHD": 50, "UHD": 30, "HDR": 20}
        
        # Convert percentages to counts
        quality_list = []
        remaining = num_streams
        
        for q, percentage in quality_distribution.items():
            if q not in QUALITY_PROFILES:
                continue
            
            if list(quality_distribution.keys())[-1] != q:
                count = int(num_streams * percentage / 100)
                remaining -= count
                quality_list.extend([q] * count)
            else:
                quality_list.extend([q] * remaining)
        
        random.shuffle(quality_list)
    else:
        quality_list = [quality] * num_streams
    
    # Assign clients to streams
    available_clients = list(range(1, num_clients + 1))
    random.shuffle(available_clients)
    client_assignments = available_clients[:num_streams]
    
    # Add background traffic
    if num_streams < num_clients:
        background_client = random.choice([c for c in available_clients if c not in client_assignments[:num_streams]])
        commands.append({
            "testname": "Background Traffic",
            "host2": {
                "name": "server1-mgmt",
                "cmd": f"/usr/bin/ITGSend -T UDP -a client{background_client} -V 1 0.5 -c 512 -t {experiment_duration * 1000}",
                "persist": False,
                "start_time": 0
            }
        })
    
    # Calculate start times to ensure coverage
    stream_duration = 225  # ~3.75 minutes per stream
    start_times = [0]  # First stream starts at 0
    
    # Stagger remaining streams
    for i in range(1, num_streams):
        ideal_time = (i * 60) % (experiment_duration - stream_duration)
        variation = random.randint(-10, 10)
        start_time = max(0, min(experiment_duration - stream_duration, ideal_time + variation))
        start_times.append(start_time)
    
    # Generate commands for each stream
    for i, start_time in enumerate(start_times):
        if i >= num_streams:
            break
            
        stream_id = f"stream{i+1}"
        client_num = client_assignments[i]
        stream_quality = quality_list[i]
        profile = QUALITY_PROFILES[stream_quality]
        
        # Add commands for this stream
        commands.extend([
            {
                "testname": f"Netflix {profile['display_name']} - Idle traffic (Stream {stream_id})",
                "host2": {
                    "name": "server1-mgmt",
                    "cmd": f"/usr/bin/ITGSend -T UDP -a client{client_num} -V 1 {profile['idle_traffic']} -c {profile['packet_size']} -t {experiment_duration * 1000}",
                    "persist": False,
                    "start_time": start_time
                }
            },
            {
                "testname": f"Netflix {profile['display_name']} - Initial Burst Phase (Stream {stream_id})",
                "host2": {
                    "name": "server1-mgmt",
                    "cmd": f"/usr/bin/ITGSend -T UDP -a client{client_num} -U {profile['initial_burst_min']} {profile['initial_burst_max']} -c {profile['packet_size']} -j 0 -B U 140 750 U 0.1 150 -t 5700",
                    "persist": False,
                    "start_time": start_time
                }
            },
            {
                "testname": f"Netflix {profile['display_name']} - High Bitrate (Stream {stream_id})",
                "host2": {
                    "name": "server1-mgmt",
                    "cmd": f"/usr/bin/ITGSend -T UDP -a client{client_num} -O {profile['high_bitrate']} -c {profile['packet_size']} -t 225000 -d 15000 -B U 50 400 C 5000",
                    "persist": False,
                    "start_time": start_time
                }
            },
            {
                "testname": f"Netflix {profile['display_name']} - Medium Bitrate (Stream {stream_id})",
                "host2": {
                    "name": "server1-mgmt",
                    "cmd": f"/usr/bin/ITGSend -T UDP -a client{client_num} -O {profile['medium_bitrate']} -c {profile['packet_size']} -t 225000 -d 15000 -B U 50 400 C 5000",
                    "persist": False,
                    "start_time": start_time
                }
            }
        ])
    
    return commands

def estimate_bandwidth(quality, num_streams, mixed_quality=False, quality_distribution=None):
    """Estimate total bandwidth in Mbps"""
    if mixed_quality and quality_distribution:
        total_bandwidth = 0
        for q, percentage in quality_distribution.items():
            if q not in QUALITY_PROFILES:
                continue
            stream_count = math.ceil(num_streams * percentage / 100)
            total_bandwidth += QUALITY_PROFILES[q]['high_bitrate'] * stream_count / 1000
        return total_bandwidth
    else:
        return (QUALITY_PROFILES[quality]['high_bitrate'] * num_streams) / 1000

def generate_from_csv(csv_file_path):
    """Generate test configurations from a CSV file"""
    os.makedirs("test_configs", exist_ok=True)
    test_summary = []
    
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            test_name = row.get('test_name', 'test')
            num_streams = int(row.get('num_streams', 5))
            quality = row.get('quality', 'FHD')
            num_clients = int(row.get('num_clients', 20))
            mixed = row.get('mixed', '').lower() in ['true', 'yes', '1', 'y']
            duration = int(row.get('duration', 300))
            
            # Process quality distribution if mixed
            quality_distribution = None
            if mixed:
                quality_distribution = {
                    "FHD": int(row.get('fhd_pct', 50)),
                    "UHD": int(row.get('uhd_pct', 30)),
                    "HDR": int(row.get('hdr_pct', 20))
                }
            
            # Generate commands
            if mixed:
                print(f"Generating test case '{test_name}' with mixed quality...")
                commands = generate_netflix_streams(num_streams, num_clients=num_clients, 
                                                  experiment_duration=duration,
                                                  mixed_quality=True, 
                                                  quality_distribution=quality_distribution)
                quality_desc = "mixed"
            else:
                print(f"Generating test case '{test_name}' with {QUALITY_PROFILES[quality]['display_name']} quality...")
                commands = generate_netflix_streams(num_streams, quality=quality, 
                                                  num_clients=num_clients,
                                                  experiment_duration=duration)
                quality_desc = quality
            
            # Save to file
            output_file = f"test_configs/{test_name}.json"
            with open(output_file, 'w') as f:
                json.dump(commands, f, indent=2)
            
            # Estimate bandwidth
            estimated_bandwidth = estimate_bandwidth(quality, num_streams, mixed, quality_distribution)
            
            # Add to summary
            test_info = {
                'test_name': test_name,
                'streams': num_streams,
                'quality': quality_desc,
                'clients': num_clients,
                'estimated_bandwidth': f"{estimated_bandwidth:.2f} Mbps"
            }
            
            test_summary.append(test_info)
            print(f"  - Estimated bandwidth: {estimated_bandwidth:.2f} Mbps")
            print(f"  - Output saved to '{output_file}'")
    
    # Generate summary CSV
    if test_summary:
        with open("test_configs/test_summary.csv", 'w', newline='') as csvfile:
            fieldnames = list(test_summary[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_summary)

def generate_test_cases_csv(output_file="test_cases.csv"):
    """Generate CSV with test cases for 50Mbps and 100Mbps networks"""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['test_name', 'num_streams', 'quality', 'num_clients', 'mixed', 
                      'duration', 'fhd_pct', 'uhd_pct', 'hdr_pct', 'target_network']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Test cases for 50Mbps network
        writer.writerow({
            'test_name': '50mbps_fhd_normal',
            'num_streams': 7,
            'quality': 'FHD',
            'num_clients': 10,
            'mixed': 'no',
            'duration': 300,
            'target_network': '50Mbps'
        })
        
        writer.writerow({
            'test_name': '50mbps_fhd_limit',
            'num_streams': 7,
            'quality': 'FHD',
            'num_clients': 10,
            'mixed': 'no',
            'duration': 300,
            'target_network': '50Mbps'
        })
        
        writer.writerow({
            'test_name': '50mbps_mixed',
            'num_streams': 5,
            'quality': 'FHD',
            'num_clients': 8,
            'mixed': 'yes',
            'duration': 300,
            'fhd_pct': 60,
            'uhd_pct': 30,
            'hdr_pct': 10,
            'target_network': '50Mbps'
        })
        
        writer.writerow({
            'test_name': '50mbps_4k',
            'num_streams': 3,
            'quality': 'UHD',
            'num_clients': 5,
            'mixed': 'no',
            'duration': 300,
            'target_network': '50Mbps'
        })
        
        writer.writerow({
            'test_name': '50mbps_overload',
            'num_streams': 4,
            'quality': 'UHD',
            'num_clients': 5,
            'mixed': 'no',
            'duration': 300,
            'target_network': '50Mbps'
        })
        
        # Test cases for 100Mbps network
        writer.writerow({
            'test_name': '100mbps_fhd',
            'num_streams': 13,
            'quality': 'FHD',
            'num_clients': 15,
            'mixed': 'no',
            'duration': 300,
            'target_network': '100Mbps'
        })
        
        writer.writerow({
            'test_name': '100mbps_4k',
            'num_streams': 6,
            'quality': 'UHD',
            'num_clients': 8,
            'mixed': 'no',
            'duration': 300,
            'target_network': '100Mbps'
        })
        
        writer.writerow({
            'test_name': '100mbps_mixed',
            'num_streams': 10,
            'quality': 'FHD',
            'num_clients': 12,
            'mixed': 'yes',
            'duration': 300,
            'fhd_pct': 50,
            'uhd_pct': 30,
            'hdr_pct': 20,
            'target_network': '100Mbps'
        })
        
        writer.writerow({
            'test_name': '100mbps_hdr',
            'num_streams': 5,
            'quality': 'HDR',
            'num_clients': 6,
            'mixed': 'no',
            'duration': 300,
            'target_network': '100Mbps'
        })
        
        writer.writerow({
            'test_name': '100mbps_overload',
            'num_streams': 12,
            'quality': 'UHD',
            'num_clients': 15,
            'mixed': 'yes',
            'duration': 300,
            'fhd_pct': 40,
            'uhd_pct': 40,
            'hdr_pct': 20,
            'target_network': '100Mbps'
        })
    
    print(f"Sample test cases CSV created at '{output_file}'")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate Netflix traffic simulation for D-iTG')
    parser.add_argument('--streams', type=int, default=5, help='Number of Netflix streams')
    parser.add_argument('--clients', type=int, default=20, help='Total available clients')
    parser.add_argument('--quality', type=str, default='FHD', choices=QUALITY_PROFILES.keys(), help='Video quality')
    parser.add_argument('--mixed', action='store_true', help='Use mixed quality distribution')
    parser.add_argument('--duration', type=int, default=300, help='Experiment duration (seconds)')
    parser.add_argument('--output', type=str, default=None, help='Output filename (without .json)')
    parser.add_argument('--csv', type=str, default=None, help='CSV file with test configurations')
    parser.add_argument('--generate-csv', action='store_true', help='Generate sample test cases CSV')
    
    args = parser.parse_args()
    os.makedirs("test_configs", exist_ok=True)
    
    if args.generate_csv:
        csv_file = generate_test_cases_csv()
        print(f"You can now run: python {os.path.basename(__file__)} --csv {csv_file}")
        return
    
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV file '{args.csv}' not found")
            return
        generate_from_csv(args.csv)
        return
    
    # Generate a single test case from command line arguments
    quality_distribution = {"FHD": 50, "UHD": 30, "HDR": 20} if args.mixed else None
    commands = generate_netflix_streams(
        args.streams, 
        quality=args.quality, 
        num_clients=args.clients,
        experiment_duration=args.duration,
        mixed_quality=args.mixed, 
        quality_distribution=quality_distribution
    )
    
    # Determine output filename
    quality_desc = "mixed" if args.mixed else args.quality
    output_file = f"test_configs/{args.output or f'netflix_{quality_desc}_{args.streams}streams'}.json"
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(commands, f, indent=2)
    
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(args.quality, args.streams, args.mixed, quality_distribution)
    
    print(f"Generated Netflix traffic simulation with {args.streams} streams")
    print(f"Estimated bandwidth: {bandwidth:.2f} Mbps")
    print(f"Output saved to '{output_file}'")

if __name__ == "__main__":
    main()