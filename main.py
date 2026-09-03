from DataLoader.DataLoader import DataLoader

def main():
    try:
        loader = DataLoader()
        print(f"here is path to files: {loader.path}")
    except Exception as e:
        print(e)



if __name__ == "__main__":
    main()
