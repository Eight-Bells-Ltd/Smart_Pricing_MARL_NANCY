import os
import cv2
import matplotlib.pyplot as plt
import csv

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