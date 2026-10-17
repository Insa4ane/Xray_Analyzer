import logging
import torch as tc
from DataLoader.Loader import DataLoader
from agent.XrayAgent import XrayAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("sanity_check.log")]
)

def main():
    tc.manual_seed(42)
    loader = DataLoader()
    X_train, X_test, y_train, y_test, X_val, y_val = loader.split_data()

    agent = XrayAgent()
    agent.eval()

    X = X_train[:32].to(agent.device)
    y = y_train[:32].to(agent.device).view(-1, 1)
    logging.info(f"Batch: {len(y)} zdjec, klasa 1: {int(y.sum().item())}, klasa 0: {int((1 - y).sum().item())}")

    criterion = tc.nn.BCEWithLogitsLoss()
    optimizer = tc.optim.Adam(agent.parameters(), lr=1e-3)

    for step in range(300):
        optimizer.zero_grad()
        logits = agent(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 299:
            acc = ((tc.sigmoid(logits) > 0.5).float() == y).float().mean().item() * 100
            logging.info(f"step {step:3d} | loss {loss.item():.4f} | acc {acc:.1f}%")

if __name__ == "__main__":
    main()