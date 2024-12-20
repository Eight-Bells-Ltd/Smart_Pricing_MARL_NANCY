import cv2
import matplotlib.pyplot as plt
import csv
import os
import uuid

auction_data = {}

def render_env(num_agents, round_number, bids, agent_list, output_folder):

    plt.figure(figsize=(10, 6))

    # Plot bids
    [plt.plot(i + 1, bids[i], marker='o', label=f"{agent_list[i]} Bid") for i in range(num_agents)]
    
    # Plot settings
    plt.xlabel("Agents")
    plt.ylabel("Bid Value")
    plt.ylim(0, 100)
    plt.title(f"Bids of Each Agent in Round {round_number}")
    plt.legend()
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(output_folder, f"round_{round_number}_bids.png"))
    plt.close()

    # Append bid data for the round to a CSV file
    with open(os.path.join(output_folder, f"out.csv"), mode='a', newline='') as file:
        writer = csv.writer(file)
        for i, bid in enumerate(bids):
            writer.writerow([round_number, agent_list[i], bid])

def render_final_plot(agent_list, data_file, output_folder):
    # Read data from the CSV file
    rounds = {}
    with open(data_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            round_number, agent, bid = int(row[0]), row[1], float(row[2])
            if agent not in rounds:
                rounds[agent] = []
            rounds[agent].append((round_number, bid))

    # Plotting the data for each agent across rounds
    plt.figure(figsize=(12, 8))
    for agent in agent_list:
        rounds_sorted = sorted(rounds[agent])  # Sort by round number
        round_nums, bids = zip(*rounds_sorted)
        plt.plot(round_nums, bids, marker='o', label=f"{agent}")

    # Plot settings
    plt.xlabel("Round Number")
    plt.ylabel("Bid Value")
    plt.title("Bids of Each Agent Throughout the Auction")
    plt.legend()
    plt.grid(True)

    # Save the final comprehensive plot
    plt.savefig(os.path.join(output_folder, "full_auction_bids.png"))
    plt.close()
    print(f"Plot saved to {output_folder}/full_auction_bids.png")

def create_video_from_pngs(images_folder):

    output_video = "outputs/output_video.mp4"

    # Get sorted list of PNG file paths in the folder by creation date
    image_files = sorted(
        (os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(".png")),
        key=lambda f: os.path.getmtime(f)
    )

    if not image_files:
        print("No PNG images found in the specified folder.")
        return

    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), 0.8, (width, height))

    # Write each image to the video
    for image_file in image_files:
        video.write(cv2.imread(image_file))

    video.release()

    print(f"Video created successfully: {output_video}")

def plot_all_rounds(my_bid_history, output_folder, agent_name_mapping, min_limit_bid, max_limit_bid):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Plot data for each agent
    for agent, bid_history in my_bid_history.items():
        rounds = range(1, len(bid_history) + 1)  # Rounds are 1-based indexing
        agent_index = agent_name_mapping[agent]  # Map agent to its index

        # Plot the bid history
        line, = plt.plot(rounds, bid_history, marker='o', label=f"{agent}")

        # Get the color of the current line
        color = line.get_color()

        # Plot min and max limit lines
        plt.axhline(y=min_limit_bid[agent_index], color=color, linestyle='--', linewidth=1, alpha=0.4, label='_nolegend_')
        plt.axhline(y=max_limit_bid[agent_index], color=color, linestyle='--', linewidth=1, alpha=0.4, label='_nolegend_')

    # Plot settings
    plt.xlabel("Round Number")
    plt.ylabel("Bid Value")
    plt.title("Bids of Each Agent Throughout All Rounds")
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_file = os.path.join(output_folder, "all_rounds_bids.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to {output_file}")

def save_data(my_bid_history, output_folder, agent_name_mapping, min_limit_bid, max_limit_bid, auction_id=None):
    # Generate a unique auction ID if not provided
    if auction_id is None:
        auction_id = str(uuid.uuid4())

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define file name with auction ID to ensure uniqueness
    file_path = os.path.join(output_folder, f"auction_{auction_id}.csv")

    # Define headers for the CSV file
    headers = ["Auction_ID", "Agent", "Round", "My_Bid", "My_Max", "My_Min"]

    # Write the data to a new file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers

        # Iterate through each agent and write their data
        for agent, agent_id in agent_name_mapping.items():
            for round_num in range(len(my_bid_history[agent])):
                writer.writerow([
                    auction_id,  # Unique Auction ID
                    agent,  # Agent name
                    round_num + 1,  # Round number
                    my_bid_history[agent][round_num],  # My Bid history
                    max_limit_bid[agent_id],  # My Max bid
                    min_limit_bid[agent_id]  # My Min bid
                ])

    # print(f"Data successfully saved to {file_path}.")

