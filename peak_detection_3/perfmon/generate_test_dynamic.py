#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Tuple

IPERF_BASE_PORT = 5201
IPERF_PORTS_PER_SERVER = 8


def parse_traffic_config(value: str) -> Tuple[List[float], str, float]:
    """Parse the traffic definition used in Experiments_dynamic_v1.csv."""
    if not value or value == "-":
        return [], "", 0.0

    if "@" in value:
        flows_str, rate_str = value.split("@", 1)
        flows = [float(item.strip()) for item in flows_str.split(",") if item.strip()]
        rate = float(rate_str.split()[0])
        return flows, "tcp", rate

    if "CBR UDP" in value:
        rates_str = value.split("Mb/s")[0]
        rates = [float(item.strip()) for item in rates_str.split(",") if item.strip()]
        return rates, "cbr", 0.0

    if "TCP flow" in value or "TCP flows" in value:
        flows_str = value.split("TCP")[0]
        flows = [float(item.strip()) for item in flows_str.split(",") if item.strip()]
        return flows, "tcp_unlimited", float("inf")

    return [], "", 0.0


def parse_stochastic_config(value: str) -> List[Dict[str, Any]]:
    """Parse stochastic traffic definitions such as Video(...) or VoIP(...)."""
    if not value or value == "-":
        return []

    stochastic_flows: List[Dict[str, Any]] = []
    for flow in value.split("+"):
        flow = flow.strip()
        if "VoIP" in flow:
            params = flow[flow.index("(") + 1:flow.index(")")].split(";")
            stochastic_flows.append({
                "type": "VoIP",
                "min_size": int(params[0]),
                "max_size": int(params[1]),
                "variance": float(params[2]),
            })
        elif "Gaming" in flow:
            params = flow[flow.index("(") + 1:flow.index(")")].split(";")
            stochastic_flows.append({
                "type": "Gaming",
                "min_size": int(params[0]),
                "max_size": int(params[1]),
                "variance": float(params[2]),
            })
        elif "Video" in flow:
            params = flow[flow.index("(") + 1:flow.index(")")].split(";")
            stochastic_flows.append({
                "type": "Video",
                "min_size": int(params[0]),
                "max_size": int(params[1]),
                "distribution": params[2],
                "variance": float(params[3]),
            })

    return stochastic_flows


def create_tcp_config(client_index: int, server_index: int, server_port: int, rate: float, start_time: int, duration: int, flow_label: str) -> Dict[str, Any]:
    command = f"iperf3 -c server{server_index} -p {server_port} -C reno -R -t {duration}"
    if rate != float("inf"):
        command = f"iperf3 -c server{server_index} -p {server_port} -C reno -R -b {rate}M -t {duration}"

    return {
        "testname": f"TCP download {flow_label}",
        "host1": {
            "name": f"client{client_index}-mgmt",
            "cmd": command,
            "persist": False,
            "start_time": start_time,
        },
    }


def create_udp_config(client_index: int, server_index: int, server_port: int, rate: float, start_time: int, duration: int, flow_label: str) -> Dict[str, Any]:
    return {
        "testname": f"UDP download {flow_label}",
        "host1": {
            "name": f"client{client_index}-mgmt",
            "cmd": f"iperf3 -c server{server_index} -p {server_port} -R -u -b {rate}M -t {duration}",
            "persist": False,
            "start_time": start_time,
        },
    }


def create_stochastic_config(client_index: int, server_index: int, flow: Dict[str, Any], start_time: int, duration_seconds: int, flow_label: str) -> Dict[str, Any]:
    duration_ms = duration_seconds * 1000
    config = {
        "testname": f"{flow['type']} traffic {flow_label}",
        "host1": {
            "name": f"server{server_index}-mgmt",
            "persist": False,
            "start_time": start_time,
        },
    }

    if flow["type"] == "VoIP":
        config["host1"]["cmd"] = (
            f"ITGSend -T UDP -a client{client_index} "
            f"-C {flow['max_size']} -c {flow['min_size']} "
            f"-t {duration_ms} -x voip_recv_log_{server_index}_{client_index} "
            f"-l voip_send_log_{server_index}_{client_index} -e 100 -E 100 -k 20 -K 20 -v {flow['variance']}"
        )
    elif flow["type"] == "Gaming":
        config["host1"]["cmd"] = (
            f"ITGSend -T UDP -a client{client_index} "
            f"-C {flow['max_size']} -c {flow['min_size']} "
            f"-t {duration_ms} -x game_recv_log_{server_index}_{client_index} "
            f"-l game_send_log_{server_index}_{client_index} -e 50 -E 50 -k 10 -K 10 -v {flow['variance']}"
        )
    elif flow["type"] == "Video":
        config["host1"]["cmd"] = (
            f"ITGSend -T UDP -a client{client_index} "
            f"-C {flow['max_size']} -c {flow['min_size']} "
            f"-t {duration_ms} -x video_recv_log_{server_index}_{client_index} "
            f"-l video_send_log_{server_index}_{client_index} -e 1000 -E 1000 -k 30 -K 30 "
            f"-d {flow['distribution']} -v {flow['variance']}"
        )

    return config


def generate_download_segments(total_duration: int, min_segment: int = 10, max_segment: int = 60,
                               min_pause: int = 5, max_pause: int = 30) -> List[Dict[str, int]]:
    segments: List[Dict[str, int]] = []
    remaining_duration = total_duration
    current_time = 0

    if remaining_duration < min_segment:
        return [{"type": "download", "duration": remaining_duration, "start_time": 0}]

    while remaining_duration > 0:
        if remaining_duration < min_segment + min_pause:
            segments.append({"type": "download", "duration": remaining_duration, "start_time": current_time})
            break

        segment_duration = min(random.randint(min_segment, max_segment), remaining_duration)
        segments.append({"type": "download", "duration": segment_duration, "start_time": current_time})
        remaining_duration -= segment_duration
        current_time += segment_duration

        if remaining_duration > min_segment:
            pause_duration = min(random.randint(min_pause, max_pause), remaining_duration - min_segment)
            segments.append({"type": "pause", "duration": pause_duration, "start_time": current_time})
            remaining_duration -= pause_duration
            current_time += pause_duration

    return segments


def load_dynamic_scenarios(csv_path: str) -> Dict[str, Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["Scenario"]: row for row in reader}


def load_household_profiles(csv_path: str) -> Dict[str, Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["Profile Name"]: row for row in reader}


def next_server_index(counter: Dict[str, int], pool_size: int) -> int:
    counter["value"] += 1
    return ((counter["value"] - 1) % pool_size) + 1


def next_iperf_endpoint(counter: Dict[str, int], pool_size: int, ports_per_server: int) -> Tuple[int, int]:
    counter["value"] += 1
    zero_based = counter["value"] - 1
    server_index = (zero_based % pool_size) + 1
    port_offset = (zero_based // pool_size) % ports_per_server
    return server_index, IPERF_BASE_PORT + port_offset


def append_client_scenario_tests(
    tests: List[Dict[str, Any]],
    scenario_name: str,
    client_index: int,
    start_time: int,
    duration: int,
    dynamic_scenarios: Dict[str, Dict[str, str]],
    server_counter: Dict[str, int],
    server_pool_size: int,
    iperf_ports_per_server: int,
    randomize: bool,
    flow_prefix: str,
) -> None:
    scenario = dynamic_scenarios[scenario_name]
    limited_flows_list, _, limited_rate = parse_traffic_config(scenario["TCP flows (rate limited)"])
    unlimited_flows_list, _, _ = parse_traffic_config(scenario["TCP flows (unlimited)"])
    cbr_rates, _, _ = parse_traffic_config(scenario["CBR traffic"])
    stochastic_flows = parse_stochastic_config(scenario["Stochastic traffic"])

    limited_flows = int(limited_flows_list[0]) if limited_flows_list else 0
    unlimited_flows = int(unlimited_flows_list[0]) if unlimited_flows_list else 0
    cbr_rate = cbr_rates[0] if cbr_rates else None

    def build_tcp(rate: float, flow_name: str, offset: int = 0) -> None:
        server_index, server_port = next_iperf_endpoint(server_counter, server_pool_size, iperf_ports_per_server)
        if randomize:
            for segment_index, segment in enumerate(generate_download_segments(duration), start=1):
                if segment["type"] != "download":
                    continue
                tests.append(
                    create_tcp_config(
                        client_index,
                        server_index,
                        server_port,
                        rate,
                        start_time + segment["start_time"] + offset,
                        segment["duration"],
                        f"{flow_name}-{segment_index}",
                    )
                )
        else:
            tests.append(create_tcp_config(client_index, server_index, server_port, rate, start_time + offset, duration, flow_name))

    def build_udp(rate: float, flow_name: str, offset: int = 0) -> None:
        server_index, server_port = next_iperf_endpoint(server_counter, server_pool_size, iperf_ports_per_server)
        if randomize:
            for segment_index, segment in enumerate(generate_download_segments(duration), start=1):
                if segment["type"] != "download":
                    continue
                tests.append(
                    create_udp_config(
                        client_index,
                        server_index,
                        server_port,
                        rate,
                        start_time + segment["start_time"] + offset,
                        segment["duration"],
                        f"{flow_name}-{segment_index}",
                    )
                )
        else:
            tests.append(create_udp_config(client_index, server_index, server_port, rate, start_time + offset, duration, flow_name))

    for flow_number in range(limited_flows):
        build_tcp(limited_rate, f"{flow_prefix}-tcp-limited-{flow_number + 1}")

    for flow_number in range(unlimited_flows):
        build_tcp(float("inf"), f"{flow_prefix}-tcp-unlimited-{flow_number + 1}")

    if cbr_rate is not None:
        build_udp(cbr_rate, f"{flow_prefix}-udp-cbr")

    for flow_number, flow in enumerate(stochastic_flows, start=1):
        server_index = next_server_index(server_counter, server_pool_size)
        tests.append(
            create_stochastic_config(
                client_index,
                server_index,
                flow,
                start_time,
                duration,
                f"{flow_prefix}-stochastic-{flow_number}",
            )
        )


def generate_household_profile_tests(
    row: Dict[str, str],
    dynamic_scenarios: Dict[str, Dict[str, str]],
    output_dir: str,
    total_experiment_duration: int,
    randomize: bool,
) -> str:
    households = int(row.get("Households", 8))
    clients_per_household = int(row.get("Clients per household", 2))
    if clients_per_household < 2:
        raise ValueError("This profile mode requires at least 2 clients per household")

    first_start = int(row.get("First household start (s)", 5))
    start_interval = int(row.get("Household start interval (s)", 5))
    client_1_scenario = row["Client 1 Scenario"]
    client_2_scenario = row["Client 2 Scenario"]
    server_pool_size = int(row.get("Server pool size", 16))
    iperf_ports_per_server = int(row.get("Iperf ports per server", IPERF_PORTS_PER_SERVER))

    tests: List[Dict[str, Any]] = []
    server_counter = {"value": 0}

    for household_index in range(households):
        household_start = first_start + (household_index * start_interval)
        flow_duration = max(1, total_experiment_duration - household_start)

        client_1_index = (household_index * clients_per_household) + 1
        client_2_index = client_1_index + 1

        append_client_scenario_tests(
            tests,
            client_1_scenario,
            client_1_index,
            household_start,
            flow_duration,
            dynamic_scenarios,
            server_counter,
            server_pool_size,
            iperf_ports_per_server,
            randomize,
            f"house{household_index + 1}-client1",
        )
        append_client_scenario_tests(
            tests,
            client_2_scenario,
            client_2_index,
            household_start,
            flow_duration,
            dynamic_scenarios,
            server_counter,
            server_pool_size,
            iperf_ports_per_server,
            randomize,
            f"house{household_index + 1}-client2",
        )

    scenario_slug = row["Scenario"].replace(" ", "_").replace(",", "").lower()
    filename = os.path.join(output_dir, f"test_{scenario_slug}.json")
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(tests, handle, indent=2)

    return filename


def get_profile_name_for_house(row: Dict[str, str], household_index: int) -> str:
    house_number = household_index + 1
    value = row.get(f"House {house_number} Profile", "").strip()
    if not value:
        raise ValueError(
            f"Missing profile assignment for house {house_number}. "
            f"Expected column 'House {house_number} Profile'."
        )
    return value


def generate_mixed_household_profile_tests(
    row: Dict[str, str],
    dynamic_scenarios: Dict[str, Dict[str, str]],
    household_profiles: Dict[str, Dict[str, str]],
    output_dir: str,
    total_experiment_duration: int,
    randomize: bool,
) -> str:
    households = int(row.get("Households", 8))
    clients_per_household = int(row.get("Clients per household", 2))
    if clients_per_household != 2:
        raise ValueError("Mixed household profile mode currently requires exactly 2 clients per household")

    first_start = int(row.get("First household start (s)", 5))
    start_interval = int(row.get("Household start interval (s)", 5))
    server_pool_size = int(row.get("Server pool size", 16))
    iperf_ports_per_server = int(row.get("Iperf ports per server", IPERF_PORTS_PER_SERVER))

    tests: List[Dict[str, Any]] = []
    server_counter = {"value": 0}

    for household_index in range(households):
        household_start = first_start + (household_index * start_interval)
        flow_duration = max(1, total_experiment_duration - household_start)
        profile_name = get_profile_name_for_house(row, household_index)

        if profile_name not in household_profiles:
            raise ValueError(f"Unknown household profile '{profile_name}' for house {household_index + 1}")

        profile = household_profiles[profile_name]
        client_1_scenario = profile["Client 1 Scenario"]
        client_2_scenario = profile["Client 2 Scenario"]

        client_1_index = (household_index * clients_per_household) + 1
        client_2_index = client_1_index + 1

        append_client_scenario_tests(
            tests,
            client_1_scenario,
            client_1_index,
            household_start,
            flow_duration,
            dynamic_scenarios,
            server_counter,
            server_pool_size,
            iperf_ports_per_server,
            randomize,
            f"house{household_index + 1}-{profile_name}-client1",
        )
        append_client_scenario_tests(
            tests,
            client_2_scenario,
            client_2_index,
            household_start,
            flow_duration,
            dynamic_scenarios,
            server_counter,
            server_pool_size,
            iperf_ports_per_server,
            randomize,
            f"house{household_index + 1}-{profile_name}-client2",
        )

    scenario_slug = row["Scenario"].replace(" ", "_").replace(",", "").lower()
    filename = os.path.join(output_dir, f"test_{scenario_slug}.json")
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(tests, handle, indent=2)

    return filename


def generate_test_files(csv_file: str, output_dir: str = "test_configs", randomize: bool = True, total_experiment_duration: int = 120) -> None:
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    with open(csv_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        mode = row.get("Mode", "").strip().lower()
        source_csv = row.get("Source scenario CSV", "Experiments_dynamic_v1.csv")
        if not os.path.isabs(source_csv):
            source_csv = os.path.join(os.path.dirname(csv_file), source_csv)
        dynamic_scenarios = load_dynamic_scenarios(source_csv)

        if mode == "household_profiles":
            output_file = generate_household_profile_tests(
                row=row,
                dynamic_scenarios=dynamic_scenarios,
                output_dir=output_dir,
                total_experiment_duration=total_experiment_duration,
                randomize=randomize,
            )
        elif mode == "mixed_household_profiles":
            profile_csv = row.get("Source profile CSV", "Experiments_dynamic_household_profile_library.csv")
            if not os.path.isabs(profile_csv):
                profile_csv = os.path.join(os.path.dirname(csv_file), profile_csv)
            household_profiles = load_household_profiles(profile_csv)

            output_file = generate_mixed_household_profile_tests(
                row=row,
                dynamic_scenarios=dynamic_scenarios,
                household_profiles=household_profiles,
                output_dir=output_dir,
                total_experiment_duration=total_experiment_duration,
                randomize=randomize,
            )
        else:
            raise ValueError(f"Unsupported mode '{mode}' in {csv_file}")

        print(f"Generated {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate household-based test configs from dynamic scenario profiles.")
    parser.add_argument("--csv", default="Experiments_dynamic_household_profiles.csv", help="Input CSV describing the household scenario")
    parser.add_argument("--output-dir", default="test_configs", help="Directory for generated JSON files")
    parser.add_argument("--duration", type=int, default=120, help="Total experiment duration in seconds")
    parser.add_argument("--no-randomize", action="store_true", help="Disable random segmentation and use continuous flows")
    args = parser.parse_args()

    generate_test_files(
        csv_file=args.csv,
        output_dir=args.output_dir,
        randomize=not args.no_randomize,
        total_experiment_duration=args.duration,
    )


if __name__ == "__main__":
    main()
