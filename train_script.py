import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="data")
    return parser.parse_args()



def main():
    args = parse_args()
    print(args.config_path)
    
    
    model = get_model(args)
    train_loader, val_loader, test_loader = get_dataloaders(args)
    
    train()
    test()
    



if __name__ == "__main__":
    main()