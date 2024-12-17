import argparse
import yaml
from environment.reverse_auction_env import ReverseAuctionEnv
from training.train import train
from evaluation.evaluate import evaluate
from render.render import create_video_from_pngs
import ray


def main():
    parser = argparse.ArgumentParser(description="Reverse Auction Simulation")
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='evaluate')
    parser.add_argument('--config', default='config.yml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    env_kwargs = config['environment']
    env_fn = ReverseAuctionEnv


    ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=True)
    print(ray.available_resources())

    if args.mode == 'train':
        train_config = config['training']
        train(env_fn, steps=train_config['steps'], learning_rate=train_config['learning_rate'],
              batch_size=train_config['batch_size'], model_path=config['model_path'], **env_kwargs)
    elif args.mode == 'evaluate':
        eval_config = config['evaluation']
        evaluate(env_fn, num_games=eval_config['num_games'],
                 model_path=config['model_path'], render_mode=eval_config['render_mode'], **env_kwargs)
        # create_video_from_pngs("outputs/pngs")

    ray.shutdown()

if __name__ == "__main__":
    main()
    #test
